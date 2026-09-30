"""Langfuse scores: one lookup for every waiting step, then one ingestion request for the scores.

The fake Langfuse holds the observations it has ingested, pages them by
cursor, and keeps each score by its id, so a score sent twice is stored once.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import (
    LANGFUSE_BASE_URL,
    MAX_OBSERVATION_PAGES,
    START_TIME_MARGIN,
    UNKNOWN_START_REACH,
    LangfuseScoreSender,
    build_langfuse_sender,
    read_langfuse_credentials,
    read_step_start,
)
from langchain_sync_monitors.scores import PendingScore, Tracer
from tests.support.score_services import FakeLangfuse, build_step_id, read_request_json


def build_score(*, value: float = 0.9, name: str = "monitor_suspicion") -> PendingScore:
    return PendingScore(
        step_id=build_step_id(),
        name=name,
        value=value,
        tracer=Tracer.LANGFUSE,
        project=None,
        queued_at=0.0,
    )


def build_sender(service: FakeLangfuse) -> LangfuseScoreSender:
    return LangfuseScoreSender(http_client=service.build_client())


def read_lookup_filter(request: httpx.Request) -> list[dict[str, str]]:
    return json.loads(request.url.params["filter"])


def test_one_lookup_finds_every_waiting_step_and_one_request_writes_their_scores() -> None:
    # Arrange
    service = FakeLangfuse()
    scores = [build_score(value=0.2), build_score(value=0.9), build_score(value=0.5)]
    for score in scores:
        service.add_step(str(score.step_id))
    service.add_step(str(build_step_id()))

    # Act
    report = build_sender(service).send(scores)

    # Assert
    assert report.written == scores
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 1
    assert len(service.find_requests("POST", "/api/public/ingestion")) == 1
    assert sorted(item["value"] for item in service.scores.values()) == [0.2, 0.5, 0.9]


def test_the_lookup_asks_for_monitor_steps_since_the_earliest_waiting_step_began() -> None:
    # Arrange
    service = FakeLangfuse()
    earlier = build_score()
    later = build_score()

    # Act
    build_sender(service).send([later, earlier])

    # Assert
    [lookup] = service.find_requests("GET", "/api/public/v2/observations")
    assert lookup.url.params["fields"] == "core,basic,metadata"
    assert lookup.url.params["limit"] == "1000"
    assert "cursor" not in lookup.url.params
    since = read_step_start(earlier.step_id) - START_TIME_MARGIN
    assert read_lookup_filter(lookup) == [
        {"type": "string", "column": "name", "operator": "=", "value": "monitor step"},
        {"type": "datetime", "column": "startTime", "operator": ">=", "value": since.isoformat()},
    ]


def test_the_score_is_a_numeric_score_on_the_step_observation_in_its_environment() -> None:
    # Arrange
    service = FakeLangfuse()
    score = build_score(value=0.75, name="outer_suspicion")
    observation = service.add_step(str(score.step_id), environment="production")

    # Act
    build_sender(service).send([score])

    # Assert
    [post] = service.find_requests("POST", "/api/public/ingestion")
    [event] = read_request_json(post)["batch"]
    assert event["type"] == "score-create"
    assert event["body"] == {
        "id": str(score.score_id),
        "traceId": observation["traceId"],
        "observationId": observation["id"],
        "name": "outer_suspicion",
        "value": 0.75,
        "dataType": "NUMERIC",
        "environment": "production",
    }


def test_an_observation_without_an_environment_gives_a_score_without_one() -> None:
    # Arrange
    service = FakeLangfuse()
    score = build_score()
    service.add_step(str(score.step_id))["environment"] = None

    # Act
    build_sender(service).send([score])

    # Assert
    [post] = service.find_requests("POST", "/api/public/ingestion")
    assert "environment" not in read_request_json(post)["batch"][0]["body"]


def test_a_step_not_ingested_yet_waits_and_is_scored_once_it_is() -> None:
    # Arrange
    service = FakeLangfuse()
    sender = build_sender(service)
    score = build_score()

    # Act
    first = sender.send([score])
    posts_before_ingestion = len(service.find_requests("POST", "/api/public/ingestion"))
    service.add_step(str(score.step_id))
    second = sender.send([score])

    # Assert
    assert first.waiting == [score]
    assert posts_before_ingestion == 0
    assert second.written == [score]


def test_nothing_is_posted_when_no_waiting_step_is_found() -> None:
    # Arrange
    service = FakeLangfuse()

    # Act
    report = build_sender(service).send([build_score(), build_score()])

    # Assert
    assert len(report.waiting) == 2
    assert service.find_requests("POST", "/api/public/ingestion") == []


def test_a_score_sent_again_is_stored_once_under_its_fixed_id() -> None:
    # Arrange
    score = build_score()
    service = FakeLangfuse()
    service.add_step(str(score.step_id))
    sender = build_sender(service)

    # Act
    sender.send([score])
    sender.send([score])

    # Assert: each request is a new event, and both carry the same score id
    events = [
        read_request_json(post)["batch"][0]
        for post in service.find_requests("POST", "/api/public/ingestion")
    ]
    assert len({event["id"] for event in events}) == 2
    assert {event["body"]["id"] for event in events} == {str(score.score_id)}
    assert len(service.scores) == 1


def test_the_lookup_follows_the_cursor_until_every_waiting_step_is_found() -> None:
    # Arrange
    service = FakeLangfuse(page_size=1)
    score = build_score()
    service.add_step(str(build_step_id()))
    service.add_step(str(score.step_id))
    service.add_step(str(build_step_id()))

    # Act
    report = build_sender(service).send([score])

    # Assert: the third page is never read
    lookups = service.find_requests("GET", "/api/public/v2/observations")
    assert [lookup.url.params.get("cursor") for lookup in lookups] == [None, "1"]
    assert report.written == [score]


def test_the_lookup_reads_at_most_the_page_limit_in_one_window() -> None:
    # Arrange
    service = FakeLangfuse(page_size=1)
    for _ in range(MAX_OBSERVATION_PAGES + 3):
        service.add_step(str(build_step_id()))
    score = build_score()

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == MAX_OBSERVATION_PAGES
    assert report.waiting == [score]


def test_a_rate_limited_lookup_pauses_the_tool_and_posts_nothing() -> None:
    # Arrange
    service = FakeLangfuse(queued_answers=[httpx.Response(429, headers={"Retry-After": "60"})])
    score = build_score()
    service.add_step(str(score.step_id))

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.waiting == [score]
    assert report.pause_seconds == 60.0
    assert service.find_requests("POST", "/api/public/ingestion") == []


def test_a_rate_limit_on_a_later_page_keeps_every_score_waiting() -> None:
    # Arrange
    service = FakeLangfuse(page_size=1)
    found_first = build_score()
    on_second_page = build_score()
    service.add_step(str(found_first.step_id))
    service.add_step(str(on_second_page.step_id))
    first_page = build_sender(service).build_lookup(
        since=datetime.now(UTC) - timedelta(hours=1), cursor=None
    )
    service.queued_answers = [
        httpx.Response(
            200,
            json={"data": [service.observations[0]], "meta": {"cursor": "1"}},
            request=first_page,
        ),
        httpx.Response(429, headers={"Retry-After": "5"}),
    ]

    # Act
    report = build_sender(service).send([found_first, on_second_page])

    # Assert
    assert sorted(map(str, (score.step_id for score in report.waiting))) == sorted(
        [str(found_first.step_id), str(on_second_page.step_id)]
    )
    assert report.pause_seconds == 5.0
    assert service.find_requests("POST", "/api/public/ingestion") == []


def test_each_event_is_recorded_by_its_own_status() -> None:
    # Arrange
    service = FakeLangfuse()
    written, refused, retried = build_score(), build_score(), build_score()
    service.add_step(str(written.step_id))
    refused_observation = service.add_step(str(refused.step_id))
    retried_observation = service.add_step(str(retried.step_id))
    service.ingestion_errors = {
        refused_observation["id"]: 400,
        retried_observation["id"]: 500,
    }

    # Act
    report = build_sender(service).send([written, refused, retried])

    # Assert
    assert report.written == [written]
    assert report.refused == [refused]
    assert report.waiting == [retried]
    assert report.refusal == "Langfuse answered HTTP 400 for a score"


INGESTION_ANSWERS = {
    "rate-limited": (httpx.Response(429, headers={"Retry-After": "20"}), "waiting", 20.0),
    "event-left-out": (
        httpx.Response(207, json={"successes": [{"id": "another-event", "status": 201}]}),
        "waiting",
        None,
    ),
    "unknown-shape": (httpx.Response(207, text="not json"), "waiting", None),
    "bad-request": (httpx.Response(400, json={"message": "no"}), "refused", None),
    "unauthorised": (httpx.Response(401, json={"message": "no"}), "refused", None),
    "forbidden": (httpx.Response(403, json={"message": "no"}), "refused", None),
}


@pytest.mark.parametrize(
    ("answer", "fate", "pause_seconds"),
    list(INGESTION_ANSWERS.values()),
    ids=list(INGESTION_ANSWERS),
)
def test_the_ingestion_answer_decides_whether_the_scores_wait_or_are_refused(
    answer: httpx.Response,
    fate: str,
    pause_seconds: float | None,
) -> None:
    # Arrange: the lookup finds the step, and the ingestion request gets the answer
    service = FakeLangfuse()
    score = build_score()
    lookup = httpx.Response(200, json={"data": [service.add_step(str(score.step_id))], "meta": {}})
    service.queued_answers = [lookup, answer]

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert {"waiting": report.waiting, "refused": report.refused}[fate] == [score]
    assert report.written == []
    assert report.pause_seconds == pause_seconds
    assert service.scores == {}
    if fate == "refused":
        assert report.refusal == f"Langfuse answered HTTP {answer.status_code}"


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(401, json={"message": "unauthorised"}),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_a_lookup_that_fails_leaves_the_scores_waiting(answer: httpx.Response) -> None:
    # Arrange
    service = FakeLangfuse(queued_answers=[answer])
    score = build_score()

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.waiting == [score]
    assert report.pause_seconds is None


def test_an_observation_with_odd_metadata_does_not_void_the_page() -> None:
    # Arrange
    service = FakeLangfuse()
    score = build_score()
    service.add_step(str(build_step_id()))["metadata"] = "a string, not a mapping"
    service.add_step(str(score.step_id))

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert report.written == [score]


def test_a_step_starts_when_its_version_7_id_was_made() -> None:
    # Arrange
    before = datetime.now(UTC)
    step_id = build_step_id()
    after = datetime.now(UTC)

    # Act
    started = read_step_start(step_id)

    # Assert: the id keeps milliseconds
    assert before - timedelta(milliseconds=1) <= started <= after


def test_a_step_id_of_another_version_reaches_back_an_hour() -> None:
    # Arrange
    before = datetime.now(UTC)

    # Act
    started = read_step_start(uuid4())

    # Assert
    assert before - UNKNOWN_START_REACH - timedelta(seconds=1) <= started
    assert started <= datetime.now(UTC) - UNKNOWN_START_REACH


def test_the_credentials_come_from_the_variables_langfuse_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.test/")

    # Act
    credentials = read_langfuse_credentials()

    # Assert
    assert credentials is not None
    assert credentials.public_key == "pk-lf-test"
    assert credentials.secret_key.get_secret_value() == "sk-lf-test"
    assert credentials.base_url == "https://langfuse.example.test"
    assert "sk-lf-test" not in repr(credentials)


@pytest.mark.parametrize("missing", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_without_both_keys_there_are_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv(missing)

    # Act
    credentials = read_langfuse_credentials()

    # Assert
    assert credentials is None


def test_the_base_url_defaults_to_langfuse_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    credentials = read_langfuse_credentials()
    assert credentials is not None

    # Act
    sender = build_langfuse_sender(credentials)

    # Assert
    request = sender.http_client.build_request("GET", "/api/public/v2/observations")
    assert str(request.url) == f"{LANGFUSE_BASE_URL}/api/public/v2/observations"
    assert credentials.base_url == LANGFUSE_BASE_URL
    sender.close()


def test_a_secret_key_no_header_may_carry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk\x1blf")

    # Act, Assert
    with pytest.raises(ConfigurationError, match="LANGFUSE_SECRET_KEY"):
        read_langfuse_credentials()
