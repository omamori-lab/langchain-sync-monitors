"""LangSmith feedback: one request per score, on the step's run, in its project, without extension.

The fake LangSmith answers as the REST API does: `404` for a run not ingested
yet, `409` for a feedback id it has already stored, and `429` with
`Retry-After` when rate-limited.
"""

from __future__ import annotations

from uuid import UUID, uuid5

import httpx
import pytest

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langsmith_scores import (
    LANGSMITH_ENDPOINT,
    LangSmithFeedbackSender,
    build_langsmith_sender,
    read_langsmith_credentials,
)
from langchain_sync_monitors.scores import SCORE_ID_NAMESPACE, PendingScore, Tracer
from tests.support.score_services import (
    PROJECT_ID,
    PROJECT_NAME,
    FakeLangSmith,
    build_step_id,
    read_request_json,
)


def build_score(*, project: str = PROJECT_NAME, value: float = 0.9) -> PendingScore:
    return PendingScore(
        step_id=build_step_id(),
        name="monitor_suspicion",
        value=value,
        tracer=Tracer.LANGSMITH,
        project=project,
        queued_at=0.0,
    )


def build_sender(service: FakeLangSmith) -> LangSmithFeedbackSender:
    return LangSmithFeedbackSender(http_client=service.build_client())


def build_sender_knowing_the_project(service: FakeLangSmith) -> LangSmithFeedbackSender:
    """Return a sender that has already looked up the project, so it asks for none."""
    sender = build_sender(service)
    sender.project_ids[PROJECT_NAME] = PROJECT_ID
    return sender


def test_the_feedback_is_a_model_score_on_the_step_run_in_its_project_without_extension() -> None:
    # Arrange
    service = FakeLangSmith()
    score = build_score(value=0.75)

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.written == [score]
    [post] = service.find_requests("POST", "/feedback")
    assert read_request_json(post) == {
        "id": str(
            uuid5(SCORE_ID_NAMESPACE, f"langsmith/{PROJECT_NAME}/{score.step_id}/monitor_suspicion")
        ),
        "run_id": str(score.step_id),
        "session_id": PROJECT_ID,
        "key": "monitor_suspicion",
        "score": 0.75,
        "feedback_source": {"type": "model"},
        "extend_trace_retention": False,
    }


def test_the_project_is_looked_up_once_for_many_scores_and_kept() -> None:
    # Arrange
    service = FakeLangSmith()
    sender = build_sender(service)

    # Act
    sender.send([build_score(), build_score(), build_score()])
    sender.send([build_score()])

    # Assert
    [lookup] = service.find_requests("GET", "/sessions")
    assert lookup.url.params["name"] == PROJECT_NAME
    assert len(service.find_requests("POST", "/feedback")) == 4


def test_a_project_not_found_yet_leaves_its_scores_waiting_until_it_is() -> None:
    # Arrange
    service = FakeLangSmith(projects={})
    sender = build_sender(service)
    score = build_score()

    # Act
    first = sender.send([score])
    posts_before_the_project = len(service.find_requests("POST", "/feedback"))
    service.projects[PROJECT_NAME] = PROJECT_ID
    second = sender.send([score])

    # Assert
    assert first.waiting == [score]
    assert posts_before_the_project == 0
    assert second.written == [score]
    assert len(service.find_requests("GET", "/sessions")) == 2


def test_a_run_not_ingested_yet_waits_and_is_written_later() -> None:
    # Arrange
    service = FakeLangSmith()
    sender = build_sender(service)
    score = build_score()
    service.unknown_runs.add(str(score.step_id))

    # Act
    first = sender.send([score])
    service.unknown_runs.clear()
    second = sender.send([score])

    # Assert
    assert first.waiting == [score]
    assert first.refused == []
    assert second.written == [score]


def test_a_score_sent_again_is_stored_once() -> None:
    # Arrange
    score = build_score()
    service = FakeLangSmith()
    sender = build_sender(service)

    # Act
    first = sender.send([score])
    second = sender.send([score])

    # Assert: the second post meets the first's fixed id, which counts as written
    assert first.written == [score]
    assert second.written == [score]
    assert len(service.feedback) == 1
    ids = {read_request_json(post)["id"] for post in service.find_requests("POST", "/feedback")}
    assert len(ids) == 1


def test_a_rate_limit_pauses_the_tool_and_leaves_the_rest_waiting() -> None:
    # Arrange
    service = FakeLangSmith(
        queued_answers=[
            httpx.Response(200, json={}),
            httpx.Response(429, headers={"Retry-After": "12"}),
        ],
    )
    sender = build_sender_knowing_the_project(service)
    scores = [build_score(), build_score(), build_score()]

    # Act
    report = sender.send(scores)

    # Assert: the third score is never posted
    assert report.written == scores[:1]
    assert report.waiting == scores[1:]
    assert report.pause_seconds == 12.0
    assert len(service.find_requests("POST", "/feedback")) == 2


def test_a_rate_limited_project_lookup_pauses_every_score() -> None:
    # Arrange
    service = FakeLangSmith(queued_answers=[httpx.Response(429, headers={"Retry-After": "40"})])
    scores = [build_score(), build_score()]

    # Act
    report = build_sender(service).send(scores)

    # Assert
    assert report.waiting == scores
    assert report.pause_seconds == 40.0
    assert service.find_requests("POST", "/feedback") == []


def test_a_service_that_stays_down_leaves_every_score_waiting_without_a_pause() -> None:
    # Arrange
    service = FakeLangSmith(queued_answers=[httpx.Response(503), httpx.Response(503)])
    sender = build_sender_knowing_the_project(service)
    scores = [build_score(), build_score()]

    # Act
    report = sender.send(scores)

    # Assert: one score retried once, and the second not tried in this window
    assert report.waiting == scores
    assert report.pause_seconds is None
    assert len(service.find_requests("POST", "/feedback")) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_a_client_error_refuses_the_score_with_the_status(status: int) -> None:
    # Arrange
    service = FakeLangSmith(queued_answers=[httpx.Response(status, json={"detail": "no"})])
    sender = build_sender_knowing_the_project(service)
    score = build_score()

    # Act
    report = sender.send([score])

    # Assert
    assert report.refused == [score]
    assert report.refusal == f"LangSmith answered HTTP {status}"


def test_a_project_lookup_in_an_unknown_shape_leaves_the_scores_waiting() -> None:
    # Arrange
    service = FakeLangSmith(queued_answers=[httpx.Response(200, json={"unexpected": True})])
    score = build_score()

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.waiting == [score]


def test_the_credentials_come_from_the_variables_langsmith_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-key")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://smith.example.test/api/v1/")
    monkeypatch.setenv("LANGCHAIN_WORKSPACE_ID", "workspace-7")

    # Act
    credentials = read_langsmith_credentials()

    # Assert
    assert credentials is not None
    assert credentials.api_key.get_secret_value() == "lc-key"
    assert credentials.endpoint == "https://smith.example.test/api/v1"
    assert credentials.workspace_id == "workspace-7"
    assert "lc-key" not in repr(credentials)


def test_without_a_key_there_are_no_credentials_and_the_endpoint_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    for name in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    # Act
    without_key = read_langsmith_credentials()
    monkeypatch.setenv("LANGSMITH_API_KEY", "key")
    with_key = read_langsmith_credentials()

    # Assert
    assert without_key is None
    assert with_key is not None
    assert with_key.endpoint == LANGSMITH_ENDPOINT


def test_a_key_no_header_may_carry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("LANGSMITH_API_KEY", "key\nwith a line break")

    # Act, Assert
    with pytest.raises(ConfigurationError, match="LANGSMITH_API_KEY"):
        read_langsmith_credentials()


@pytest.mark.parametrize(
    ("workspace_id", "expected"), [("workspace-7", "workspace-7"), (None, None)]
)
def test_the_sender_authenticates_with_the_key_and_any_workspace(
    monkeypatch: pytest.MonkeyPatch,
    workspace_id: str | None,
    expected: str | None,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGSMITH_API_KEY", "key-for-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://smith.example.test/api/v1")
    for name in ("LANGSMITH_WORKSPACE_ID", "LANGCHAIN_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)
    if workspace_id is not None:
        monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", workspace_id)
    credentials = read_langsmith_credentials()
    assert credentials is not None

    # Act
    sender = build_langsmith_sender(credentials)

    # Assert: the feedback path is relative to the endpoint, as the SDK posts it
    request = sender.http_client.build_request("POST", "/feedback")
    assert str(request.url) == "https://smith.example.test/api/v1/feedback"
    assert request.headers["x-api-key"] == "key-for-test"
    assert request.headers.get("X-Tenant-Id") == expected
    sender.close()


def test_the_score_id_is_fixed_by_the_step_the_name_the_tool_and_the_project() -> None:
    # Arrange
    step_id = UUID("01a0f41a-6ff7-7212-81f7-f2164b013f97")
    score = PendingScore(
        step_id=step_id,
        name="outer_suspicion",
        value=0.2,
        tracer=Tracer.LANGSMITH,
        project="p",
        queued_at=5.0,
    )

    # Act
    ids = {
        score.score_id,
        PendingScore(
            step_id=step_id,
            name="outer_suspicion",
            value=0.9,
            tracer=Tracer.LANGSMITH,
            project="p",
            queued_at=99.0,
        ).score_id,
    }
    others = {
        PendingScore(
            step_id=step_id,
            name="inner_suspicion",
            value=0.2,
            tracer=Tracer.LANGSMITH,
            project="p",
            queued_at=5.0,
        ).score_id,
        PendingScore(
            step_id=step_id,
            name="outer_suspicion",
            value=0.2,
            tracer=Tracer.LANGFUSE,
            project=None,
            queued_at=5.0,
        ).score_id,
    }

    # Assert: value and queue time never change the id; name and tool do
    assert ids == {uuid5(SCORE_ID_NAMESPACE, f"langsmith/p/{step_id}/outer_suspicion")}
    assert not others & ids
    assert len(others) == 2
