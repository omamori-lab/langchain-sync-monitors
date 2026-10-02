"""Langfuse scores: one lookup for every waiting step, then one ingestion request for the scores.

The fake Langfuse holds the observations it has ingested, pages them by
cursor, and keeps each score by its id, so a score sent twice is stored once.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.utils.uuid import uuid7

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import (
    LANGFUSE_BASE_URL,
    MAX_OBSERVATION_PAGES,
    STALE_AFTER_SECONDS,
    STALE_LOOKUP_SECONDS,
    START_TIME_MARGIN,
    UNKNOWN_START_REACH,
    IngestionAnswer,
    IngestionEventStatus,
    LangfuseScoreSender,
    ObservationPage,
    build_langfuse_sender,
    match_observations,
    read_langfuse_credentials,
    read_step_start,
    record_ingestion_answer,
)
from langchain_sync_monitors.score_requests import REQUEST_TIMEOUT_SECONDS
from langchain_sync_monitors.scores import DeliveryReport, PendingScore, Tracer
from tests.support.score_services import FakeLangfuse, build_step_id, read_request_json

SENDER_LOGGER = "langchain_sync_monitors.langfuse_scores"


def build_score(
    *,
    value: float = 0.9,
    name: str = "monitor_suspicion",
    step_id: UUID | None = None,
    queued_at: float = 0.0,
) -> PendingScore:
    return PendingScore(
        step_id=step_id or build_step_id(),
        name=name,
        value=value,
        tracer=Tracer.LANGFUSE,
        project=None,
        queued_at=queued_at,
    )


def build_clocked_case() -> tuple[FakeLangfuse, list[float], LangfuseScoreSender]:
    """Return a fake Langfuse, a clock at 1000 s that a test moves, and a sender on both."""
    service = FakeLangfuse()
    now = [1000.0]
    sender = LangfuseScoreSender(http_client=service.build_client(), clock=lambda: now[0])
    return service, now, sender


def build_step_id_seconds_ago(seconds: float) -> UUID:
    """Return the id of a step made `seconds` ago, as the monitor makes ids."""
    return uuid7(nanoseconds=int((datetime.now(UTC).timestamp() - seconds) * 1e9))


def build_sender(service: FakeLangfuse) -> LangfuseScoreSender:
    return LangfuseScoreSender(http_client=service.build_client(), clock=lambda: 0.0)


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


def test_the_lookup_asks_for_monitor_steps_that_started_while_the_waiting_steps_did() -> None:
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
    latest = read_step_start(later.step_id) + START_TIME_MARGIN
    assert read_lookup_filter(lookup) == [
        {"type": "string", "column": "name", "operator": "=", "value": "monitor step"},
        {"type": "datetime", "column": "startTime", "operator": ">=", "value": since.isoformat()},
        {"type": "datetime", "column": "startTime", "operator": "<=", "value": latest.isoformat()},
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


def test_the_lookup_reads_at_most_three_pages_in_one_window() -> None:
    # Arrange: three pages a lookup is what the documented budget of 21 requests a minute rests on
    service = FakeLangfuse(page_size=1)
    for _ in range(10):
        service.add_step(str(build_step_id()))
    score = build_score()

    # Act
    report = build_sender(service).send([score])

    # Assert
    assert MAX_OBSERVATION_PAGES == 3
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 3
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
    service.queued_answers = [
        httpx.Response(200, json={"data": [service.observations[0]], "meta": {"cursor": "1"}}),
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
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the lookup finds the step, and the ingestion request gets the answer
    service = FakeLangfuse()
    score = build_score()
    lookup = httpx.Response(200, json={"data": [service.add_step(str(score.step_id))], "meta": {}})
    service.queued_answers = [lookup, answer]

    # Act
    with caplog.at_level(logging.WARNING, logger=SENDER_LOGGER):
        report = build_sender(service).send([score])

    # Assert
    assert [record.getMessage() for record in caplog.records] == (
        ["score export: Langfuse's ingestion API answered in an unknown shape"]
        if answer.content == b"not json"
        else []
    )
    assert {"waiting": report.waiting, "refused": report.refused}[fate] == [score]
    assert report.written == []
    assert report.pause_seconds == pause_seconds
    assert service.scores == {}
    if fate == "refused":
        assert report.refusal == f"Langfuse answered HTTP {answer.status_code}"


@pytest.mark.parametrize(
    ("answer", "logged"),
    [
        (
            httpx.Response(401, json={"message": "unauthorised"}),
            "score export: Langfuse answered the step lookup with HTTP 401",
        ),
        (
            httpx.Response(200, json={"unexpected": True}),
            "score export: Langfuse's step lookup answered in an unknown shape",
        ),
    ],
    ids=["unauthorised", "unknown-shape"],
)
def test_a_lookup_that_fails_leaves_the_scores_waiting_and_is_logged(
    answer: httpx.Response,
    logged: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    service = FakeLangfuse(queued_answers=[answer])
    score = build_score()

    # Act
    with caplog.at_level(logging.WARNING, logger=SENDER_LOGGER):
        report = build_sender(service).send([score])

    # Assert
    assert report.waiting == [score]
    assert report.pause_seconds is None
    assert [record.getMessage() for record in caplog.records] == [logged]


def test_an_unknown_event_in_the_answer_hides_none_of_the_events_after_it() -> None:
    # Arrange
    written, refused = build_score(), build_score()
    events = {"event-written": written, "event-refused": refused}
    answer = IngestionAnswer(
        successes=[
            IngestionEventStatus(id="another-event", status=201),
            IngestionEventStatus(id="event-written", status=201),
        ],
        errors=[
            IngestionEventStatus(id="another-event", status=400),
            IngestionEventStatus(id="event-refused", status=400),
        ],
    )
    report = DeliveryReport()

    # Act
    record_ingestion_answer(report, events=events, answer=answer)

    # Assert
    assert report.written == [written]
    assert report.refused == [refused]
    assert report.waiting == []


def test_only_the_wanted_steps_match_among_the_observations_found() -> None:
    # Arrange
    service = FakeLangfuse()
    wanted = str(build_step_id())
    service.add_step(wanted)
    service.add_step(str(build_step_id()))
    service.add_step(str(build_step_id()))["metadata"] = {"monitor_label": "monitor"}
    page = ObservationPage.model_validate({"data": service.observations})

    # Act
    matched = match_observations(page, wanted={wanted, str(build_step_id())})

    # Assert
    assert list(matched) == [wanted]
    assert matched[wanted].id == service.observations[0]["id"]


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
    assert isinstance(sender.http_client.auth, httpx.BasicAuth)
    assert sender.http_client.timeout == httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    sender.close()


def test_langfuse_base_url_comes_before_langfuse_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://eu.langfuse.example.test/")
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.langfuse.example.test")

    # Act
    credentials = read_langfuse_credentials()

    # Assert
    assert credentials is not None
    assert credentials.base_url == "https://eu.langfuse.example.test"


def test_a_secret_key_no_header_may_carry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk\x1blf")

    # Act, Assert
    with pytest.raises(ConfigurationError, match="LANGFUSE_SECRET_KEY"):
        read_langfuse_credentials()


def test_one_lookup_finds_steps_made_at_different_times_whose_spans_started_later() -> None:
    # Arrange: two steps half a minute apart; each span started half a second after its id
    service = FakeLangfuse()
    earlier = build_score(step_id=build_step_id_seconds_ago(30.0))
    later = build_score(step_id=build_step_id_seconds_ago(0.0))
    for score in (earlier, later):
        service.add_step(str(score.step_id))

    # Act
    report = build_sender(service).send([later, earlier])

    # Assert
    assert report.written == [later, earlier]
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 1


def test_a_step_waiting_a_minute_is_stale_and_looked_up_at_most_once_a_minute() -> None:
    # Arrange: a step never ingested, whose score has waited exactly the stale limit
    service, now, sender = build_clocked_case()
    stale = build_score(
        step_id=build_step_id_seconds_ago(290.0), queued_at=1000.0 - STALE_AFTER_SECONDS
    )

    # Act: windows at 0, 10 and 20 seconds, and one a minute after the first
    reports = []
    for offset in (0.0, 10.0, 20.0, STALE_LOOKUP_SECONDS):
        now[0] = 1000.0 + offset
        reports.append(sender.send([stale]))

    # Assert: asked for in the first window and a minute later only
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 2
    assert all(report.waiting == [stale] for report in reports)


def test_a_step_just_under_a_minute_is_looked_up_every_window() -> None:
    # Arrange
    service, _, sender = build_clocked_case()
    fresh = build_score(queued_at=1000.0 - STALE_AFTER_SECONDS + 0.001)

    # Act
    sender.send([fresh])
    sender.send([fresh])

    # Assert
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 2


def test_a_fresh_lookup_reaches_back_only_as_far_as_the_fresh_steps() -> None:
    # Arrange: a stale step made 290 s ago and looked up a second ago, and a fresh one made now
    service, now, sender = build_clocked_case()
    stale = build_score(step_id=build_step_id_seconds_ago(290.0), queued_at=0.0)
    now[0] = 999.0
    sender.send([stale])
    fresh = build_score(queued_at=1000.0)
    service.add_step(str(fresh.step_id))
    now[0] = 1000.0

    # Act
    report = sender.send([stale, fresh])

    # Assert: one more lookup, for the fresh step alone, so its window is seconds wide
    [_, lookup] = service.find_requests("GET", "/api/public/v2/observations")
    window = read_lookup_filter(lookup)
    earliest = datetime.fromisoformat(window[1]["value"])
    latest = datetime.fromisoformat(window[2]["value"])
    assert latest - earliest == 2 * START_TIME_MARGIN
    assert report.written == [fresh]
    assert report.waiting == [stale]


def test_a_step_that_turns_stale_between_stale_lookups_is_looked_up_with_the_fresh_ones() -> None:
    # Arrange: at 1000 s the stale lookup takes an old step; a newer one is 55 s old then
    service, now, sender = build_clocked_case()
    old = build_score(step_id=build_step_id_seconds_ago(290.0), queued_at=900.0)
    newer = build_score(step_id=build_step_id_seconds_ago(55.0), queued_at=945.0)
    sender.send([old, newer])
    service.add_step(str(newer.step_id))
    now[0] = 1010.0

    # Act: 10 s later, as in the exit drain, the newer step has turned stale
    report = sender.send([old, newer])

    # Assert: the newer step is sought and found at once; the old one waits for its minute
    lookups = service.find_requests("GET", "/api/public/v2/observations")
    assert len(lookups) == 3
    assert report.written == [newer]
    assert report.waiting == [old]


def test_a_rate_limited_fresh_lookup_skips_the_stale_one_and_leaves_it_due() -> None:
    # Arrange
    service, now, sender = build_clocked_case()
    service.queued_answers = [httpx.Response(429, headers={"Retry-After": "30"})]
    stale = build_score(queued_at=1000.0 - STALE_AFTER_SECONDS)
    fresh = build_score(queued_at=1000.0)

    # Act: a window answered 429, then one once the pause is over
    paused = sender.send([stale, fresh])
    now[0] = 1030.0
    sender.send([stale, fresh])

    # Assert: the 429 stopped the send with its pause, and both lookups ran after it
    assert paused.pause_seconds == 30.0
    assert paused.waiting == [stale, fresh]
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 3


def test_a_rate_limited_stale_lookup_is_due_again_once_the_pause_is_over() -> None:
    # Arrange
    service, now, sender = build_clocked_case()
    service.queued_answers = [httpx.Response(429, headers={"Retry-After": "5"})]
    stale = build_score(queued_at=1000.0 - STALE_AFTER_SECONDS)

    # Act
    paused = sender.send([stale])
    now[0] = 1005.0
    sender.send([stale])

    # Assert: the stale step is asked for again 5 s later, not a minute later
    assert paused.pause_seconds == 5.0
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 2


def test_the_next_stale_lookup_counts_its_minute_from_the_end_of_the_last() -> None:
    # Arrange: each lookup takes 5 s on the sender's clock
    service = FakeLangfuse()
    now = [1000.0]

    def answer_slowly(request: httpx.Request) -> httpx.Response:
        now[0] += 5.0
        return service.build_reply(request)

    http_client = httpx.Client(
        base_url=LANGFUSE_BASE_URL, transport=httpx.MockTransport(answer_slowly)
    )
    sender = LangfuseScoreSender(http_client=http_client, clock=lambda: now[0])
    stale = build_score(queued_at=0.0)
    sender.send([stale])

    # Act: a window a minute after the lookup's start, then one a minute after its end
    now[0] = 1000.0 + STALE_LOOKUP_SECONDS
    sender.send([stale])
    after_its_start = len(service.find_requests("GET", "/api/public/v2/observations"))
    now[0] = 1005.0 + STALE_LOOKUP_SECONDS
    sender.send([stale])

    # Assert
    assert after_its_start == 1
    assert len(service.find_requests("GET", "/api/public/v2/observations")) == 2
