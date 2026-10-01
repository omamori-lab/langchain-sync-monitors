"""LangSmith feedback: one request per score, on the step's run, in its project, without extension.

The fake LangSmith answers as the REST API does: `404` for a run not ingested
yet, `409` for a feedback id it has already stored, and `429` with
`Retry-After` when rate-limited. Each score carries the connection its
tracer's client sends through, so the feedback goes where the trace went.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock
from uuid import UUID, uuid5

import httpx
import pytest
from pydantic import SecretStr

from langchain_sync_monitors import request_pool
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langsmith_scores import (
    LANGSMITH_ENDPOINT,
    MAX_CLIENTS,
    LangSmithFeedbackSender,
    build_langsmith_client,
    read_client_connection,
    read_langsmith_credentials,
)
from langchain_sync_monitors.score_requests import REQUEST_TIMEOUT_SECONDS
from langchain_sync_monitors.scores import (
    SCORE_ID_NAMESPACE,
    LangSmithCredentials,
    PendingScore,
    Tracer,
)
from tests.support.score_services import (
    CONNECTION,
    LANGSMITH_KEY,
    PROJECT_ID,
    PROJECT_NAME,
    FakeLangSmith,
    build_step_id,
    read_request_json,
    set_score_credentials,
)

OWN_CONNECTION = LangSmithCredentials(
    api_key=SecretStr("own-tracer-key"),
    endpoint="https://own-smith.test/api/v1",
    workspace_id="own-workspace",
)


@dataclass(frozen=True)
class StubClient:
    """Stands in for LangSmith's `Client`, with the public attributes the sender reads."""

    api_url: object = None
    api_key: object = None
    workspace_id: object = None


def build_score(
    *,
    project: str = PROJECT_NAME,
    value: float = 0.9,
    connection: LangSmithCredentials | None = CONNECTION,
) -> PendingScore:
    return PendingScore(
        step_id=build_step_id(),
        name="monitor_suspicion",
        value=value,
        tracer=Tracer.LANGSMITH,
        project=project,
        queued_at=0.0,
        connection=connection,
    )


def build_sender(service: FakeLangSmith) -> LangSmithFeedbackSender:
    return LangSmithFeedbackSender(build_client=service.build_client, posts_in_flight=1)


def build_sender_knowing_the_project(service: FakeLangSmith) -> LangSmithFeedbackSender:
    """Return a sender that has already looked up the project, so it asks for none."""
    sender = build_sender(service)
    sender.project_ids[(CONNECTION, PROJECT_NAME)] = PROJECT_ID
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
    assert dict(lookup.url.params) == {"name": PROJECT_NAME, "limit": "1"}
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


def test_a_score_whose_project_is_not_found_waits_and_the_scores_after_it_are_sent() -> None:
    # Arrange
    service = FakeLangSmith()
    unknown = build_score(project="a-project-not-made-yet")
    known = build_score()

    # Act
    report = build_sender(service).send([unknown, known])

    # Assert
    assert report.waiting == [unknown]
    assert report.written == [known]


def test_a_project_lookup_that_fails_does_not_hold_up_the_other_projects() -> None:
    # Arrange: the first lookup fails on both attempts, the second project is found
    service = FakeLangSmith(
        projects={PROJECT_NAME: PROJECT_ID, "b-project": "b-project-id"},
        queued_answers=[httpx.Response(503), httpx.Response(503)],
    )
    stalled = build_score(project="a-project")
    found = build_score(project="b-project")

    # Act
    report = build_sender(service).send([stalled, found])

    # Assert
    assert report.waiting == [stalled]
    assert report.written == [found]
    [post] = service.find_requests("POST", "/feedback")
    assert read_request_json(post)["session_id"] == "b-project-id"


def test_a_known_project_is_not_looked_up_again_and_holds_up_no_other() -> None:
    # Arrange: the sender knows the first project, and not the second
    service = FakeLangSmith(projects={"a-project": "a-id", "b-project": "b-id"})
    sender = build_sender(service)
    sender.project_ids[(CONNECTION, "a-project")] = "a-id"
    known, unknown = build_score(project="a-project"), build_score(project="b-project")

    # Act
    report = sender.send([known, unknown])

    # Assert
    assert report.written == [known, unknown]
    [lookup] = service.find_requests("GET", "/sessions")
    assert lookup.url.params["name"] == "b-project"


def test_langsmith_workspace_id_comes_before_langchain_workspace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGSMITH_API_KEY", "key")
    monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "langsmith-workspace")
    monkeypatch.setenv("LANGCHAIN_WORKSPACE_ID", "langchain-workspace")

    # Act
    credentials = read_langsmith_credentials()

    # Assert
    assert credentials is not None
    assert credentials.workspace_id == "langsmith-workspace"


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


@pytest.mark.parametrize(
    ("answer", "logged"),
    [
        (
            httpx.Response(200, json={"unexpected": True}),
            "score export: LangSmith's project lookup answered in an unknown shape",
        ),
        (
            httpx.Response(401, json={"detail": "unauthorised"}),
            "score export: LangSmith answered the project lookup with HTTP 401",
        ),
    ],
    ids=["unknown-shape", "unauthorised"],
)
def test_a_project_lookup_that_fails_leaves_the_scores_waiting_and_is_logged(
    answer: httpx.Response,
    logged: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    service = FakeLangSmith(queued_answers=[answer])
    score = build_score()

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.langsmith_scores"):
        report = build_sender(service).send([score])

    # Assert
    assert report.waiting == [score]
    assert [record.getMessage() for record in caplog.records] == [logged]


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
def test_the_client_authenticates_with_the_key_and_any_workspace(
    workspace_id: str | None,
    expected: str | None,
) -> None:
    # Arrange
    connection = LangSmithCredentials(
        api_key=SecretStr("key-for-test"),
        endpoint="https://smith.example.test/api/v1",
        workspace_id=workspace_id,
    )

    # Act
    client = build_langsmith_client(connection)

    # Assert: the feedback path is relative to the endpoint, as the SDK posts it
    request = client.build_request("POST", "/feedback")
    assert str(request.url) == "https://smith.example.test/api/v1/feedback"
    assert request.headers["x-api-key"] == "key-for-test"
    assert request.headers.get("X-Tenant-Id") == expected
    assert client.timeout == httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    client.close()


def test_the_feedback_goes_through_the_connection_of_the_tracer_that_traced_the_run() -> None:
    # Arrange: two tracers on two connections, each with a project of the same name
    service = FakeLangSmith()
    through_environment = build_score()
    through_own_client = build_score(connection=OWN_CONNECTION)

    # Act
    report = build_sender(service).send([through_environment, through_own_client])

    # Assert: one client per connection, and each request carries its own key and workspace
    assert report.written == [through_environment, through_own_client]
    assert service.connections == [CONNECTION, OWN_CONNECTION]
    posts = service.find_requests("POST", "/feedback")
    assert [(post.url.host, post.headers["x-api-key"]) for post in posts] == [
        ("langsmith.test", LANGSMITH_KEY),
        ("own-smith.test", "own-tracer-key"),
    ]
    assert [post.headers.get("X-Tenant-Id") for post in posts] == [None, "own-workspace"]
    assert [read_request_json(post)["run_id"] for post in posts] == [
        str(through_environment.step_id),
        str(through_own_client.step_id),
    ]
    lookups = service.find_requests("GET", "/sessions")
    assert [lookup.url.host for lookup in lookups] == ["langsmith.test", "own-smith.test"]


def test_a_pause_on_one_connection_holds_the_scores_of_the_next() -> None:
    # Arrange
    service = FakeLangSmith(
        queued_answers=[httpx.Response(429, headers={"Retry-After": "9"})],
    )
    first, second = build_score(), build_score(connection=OWN_CONNECTION)

    # Act
    report = build_sender(service).send([first, second])

    # Assert
    assert report.waiting == [first, second]
    assert report.pause_seconds == 9.0
    assert service.connections == [CONNECTION]


def test_a_score_without_a_connection_is_refused() -> None:
    # Arrange
    service = FakeLangSmith()
    score = build_score(connection=None)

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.refused == [score]
    assert report.refusal == "the score names no LangSmith connection"
    assert service.requests == []


def test_the_sender_closes_every_client_it_built() -> None:
    # Arrange
    service = FakeLangSmith()
    sender = build_sender(service)
    sender.send([build_score(), build_score(connection=OWN_CONNECTION)])

    # Act
    sender.close()

    # Assert
    assert [client.is_closed for client in sender.clients.values()] == [True, True]


def test_a_tracer_client_with_a_url_and_a_key_gives_its_own_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the environment points elsewhere
    set_score_credentials(monkeypatch)
    client = StubClient(
        api_url="https://own-smith.test/api/v1/",
        api_key="own-tracer-key",
        workspace_id="own-workspace",
    )

    # Act
    connection = read_client_connection(client)

    # Assert
    assert connection == OWN_CONNECTION
    assert "own-tracer-key" not in repr(connection)


@pytest.mark.parametrize(
    "client",
    [
        MagicMock(),
        StubClient(),
        StubClient(api_url="https://own-smith.test/api/v1", api_key=None),
        StubClient(api_url="  ", api_key="own-tracer-key"),
        None,
    ],
    ids=["mock", "nothing-set", "no-key", "blank-url", "no-client"],
)
def test_a_tracer_client_without_both_a_url_and_a_key_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    client: object,
) -> None:
    # Arrange
    set_score_credentials(monkeypatch)

    # Act
    connection = read_client_connection(client)

    # Assert: the environment's connection whole, never a mix of the two
    assert connection == CONNECTION


def test_a_tracer_key_no_header_may_carry_is_refused_without_its_value() -> None:
    # Arrange
    client = StubClient(api_url="https://own-smith.test", api_key="own\nkey")

    # Act
    with pytest.raises(ConfigurationError) as raised:
        read_client_connection(client)

    # Assert
    assert str(raised.value).startswith("the LangSmith tracer's API key holds a control")
    assert "own\nkey" not in str(raised.value)


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


def test_the_posts_go_out_side_by_side_but_never_more_than_six_at_once() -> None:
    # Arrange
    service = FakeLangSmith(post_seconds=0.02)
    sender = LangSmithFeedbackSender(build_client=service.build_client, posts_in_flight=6)
    sender.project_ids[(CONNECTION, PROJECT_NAME)] = PROJECT_ID
    scores = [build_score() for _ in range(20)]

    # Act
    report = sender.send(scores)
    sender.close()

    # Assert
    assert report.written == scores
    assert 2 <= service.most_in_flight <= 6
    assert len(service.feedback) == 20


def test_a_sender_built_when_no_thread_can_start_posts_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: as during the exit drain, when the interpreter refuses new threads
    def refuse(thread: object) -> None:
        message = "can't create new thread at interpreter shutdown"
        raise RuntimeError(message)

    monkeypatch.setattr(request_pool.threading.Thread, "start", refuse)
    service = FakeLangSmith()
    sender = LangSmithFeedbackSender(build_client=service.build_client, posts_in_flight=6)
    scores = [build_score() for _ in range(4)]

    # Act
    report = sender.send(scores)

    # Assert
    assert sender.pool.threads == []
    assert report.written == scores
    assert service.most_in_flight == 1


def test_a_send_starts_no_batch_once_its_time_budget_is_spent() -> None:
    # Arrange: each post takes two seconds on the sender's clock, against a budget of five
    now = [0.0]
    service = FakeLangSmith()
    service.on_post = lambda: now.__setitem__(0, now[0] + 2.0)
    sender = LangSmithFeedbackSender(
        build_client=service.build_client,
        posts_in_flight=1,
        send_budget_seconds=5.0,
        clock=lambda: now[0],
    )
    sender.project_ids[(CONNECTION, PROJECT_NAME)] = PROJECT_ID
    scores = [build_score() for _ in range(5)]

    # Act
    report = sender.send(scores)

    # Assert: batches started at 0, 2 and 4 seconds; the rest wait
    assert report.written == scores[:3]
    assert report.waiting == scores[3:]
    assert report.pause_seconds is None


def test_the_sender_keeps_clients_for_the_latest_connections_only() -> None:
    # Arrange
    service = FakeLangSmith()
    sender = build_sender(service)
    connections = [
        LangSmithCredentials(
            api_key=SecretStr(f"key-{index}"),
            endpoint=f"https://smith-{index}.test/api/v1",
            workspace_id=None,
        )
        for index in range(MAX_CLIENTS + 1)
    ]

    # Act
    for connection in connections:
        sender.send([build_score(connection=connection)])

    # Assert: the oldest connection's client is closed, and its project forgotten
    assert list(sender.clients) == connections[1:]
    assert [client.is_closed for client in service.clients] == [True] + [False] * MAX_CLIENTS
    assert not any(key[0] == connections[0] for key in sender.project_ids)
    assert any(key[0] == connections[1] for key in sender.project_ids)


def test_langsmith_api_key_comes_before_langchain_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "langchain-key")

    # Act
    credentials = read_langsmith_credentials()

    # Assert
    assert credentials is not None
    assert credentials.api_key.get_secret_value() == "langsmith-key"
