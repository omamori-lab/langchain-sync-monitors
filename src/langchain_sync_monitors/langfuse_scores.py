"""Suspicion scores as Langfuse scores on each step's observation, through Langfuse's public API."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, Literal, TypedDict
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic import field_validator as validate_field

from langchain_sync_monitors.score_requests import (
    build_http_client,
    is_rate_limited,
    read_environment_value,
    read_pause_seconds,
    send_request,
)
from langchain_sync_monitors.scores import DeliveryReport, PendingScore
from langchain_sync_monitors.spans import STEP_SPAN_NAME

logger = logging.getLogger(__name__)

LANGFUSE_BASE_URL: Final = "https://cloud.langfuse.com"
"""Langfuse's default base URL, as its SDK has it when neither variable is set."""

OBSERVATION_PAGE_SIZE: Final = 1000
"""The most observations one page of Langfuse's observations API holds [@langfuse2026api]."""

MAX_OBSERVATION_PAGES: Final = 3
"""The most pages one lookup reads, each one request against the rate limit."""

STALE_AFTER_SECONDS: Final = 60.0
"""How long a score waits before its step counts as stale, and is looked up less often."""

STALE_LOOKUP_SECONDS: Final = 60.0
"""How often the stale steps are looked up, together."""

START_TIME_MARGIN: Final = timedelta(seconds=5)
"""How far before the earliest and after the latest waiting step's start the lookup reaches."""

UNKNOWN_START_REACH: Final = timedelta(hours=1)
"""How far back the lookup reaches for a step whose id does not say when it began."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LangfuseCredentials:
    """What reaches Langfuse's API: the project's key pair and the base URL."""

    public_key: str
    secret_key: SecretStr
    base_url: str


def read_langfuse_credentials() -> LangfuseCredentials | None:
    """Read Langfuse's credentials from the variables its SDK reads, or None without both keys.

    The keys are `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, and the
    base URL `LANGFUSE_BASE_URL`, or `LANGFUSE_HOST` [@langfuse2026].
    """
    public_key = read_environment_value("LANGFUSE_PUBLIC_KEY")
    secret_key = read_environment_value("LANGFUSE_SECRET_KEY")
    if public_key is None or secret_key is None:
        return None
    base_url = read_environment_value("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
    return LangfuseCredentials(
        public_key=public_key,
        secret_key=SecretStr(secret_key),
        base_url=(base_url or LANGFUSE_BASE_URL).rstrip("/"),
    )


def read_step_start(step_id: UUID) -> datetime:
    """Return when the step began, from its id, or an hour ago for an id that does not say.

    A version 7 UUID begins with the milliseconds since the Unix epoch at
    which it was made [@rfc9562]. The step's id is made before its span
    starts, so the time is no later than the start Langfuse records.
    """
    if step_id.version == 7:
        return datetime.fromtimestamp((step_id.int >> 80) / 1000, tz=UTC)
    return datetime.now(UTC) - UNKNOWN_START_REACH


class StepMetadata(BaseModel):
    """The one metadata key the sender reads from a `monitor step` observation."""

    monitor_step_id: str | None = None


class LangfuseObservation(BaseModel):
    """The part of a Langfuse observation the sender reads: ids, environment and step id."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    trace_id: str = Field(alias="traceId")
    environment: str | None = None
    metadata: StepMetadata | None = None

    @validate_field("metadata", mode="before")
    @classmethod
    def read_mapping_only(cls, value: object) -> object:
        """Read metadata only when it is a mapping, so one odd observation cannot void a page."""
        return value if isinstance(value, dict) else None


class PageMeta(BaseModel):
    """The cursor of the next page, absent on the last one."""

    cursor: str | None = None


class ObservationPage(BaseModel):
    """One page of Langfuse's v2 observations API [@langfuse2026api; @pydantic2026]."""

    data: list[LangfuseObservation]
    meta: PageMeta = PageMeta()


class IngestionEventStatus(BaseModel):
    """What Langfuse's ingestion API says of one event: its id, status and any message."""

    id: str
    status: int
    message: str | None = None


class IngestionAnswer(BaseModel):
    """The multi-status answer of Langfuse's ingestion API, per event [@langfuse2026api]."""

    successes: list[IngestionEventStatus] = []
    errors: list[IngestionEventStatus] = []


class ObservationFilter(TypedDict):
    """One condition of the observations API's `filter` parameter."""

    type: Literal["string", "datetime"]
    column: str
    operator: Literal["=", ">=", "<="]
    value: str


class LangfuseScoreBody(BaseModel):
    """A numeric score on one observation; its aliases are the field names of Langfuse's API."""

    id: str
    trace_id: str = Field(serialization_alias="traceId")
    observation_id: str = Field(serialization_alias="observationId")
    name: str
    value: float
    data_type: Literal["NUMERIC"] = Field(default="NUMERIC", serialization_alias="dataType")
    environment: str | None = None


class LangfuseScoreEvent(BaseModel):
    """One `score-create` event of an ingestion batch; its `id` is new for every request."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    type: Literal["score-create"] = "score-create"
    body: LangfuseScoreBody


@dataclass(frozen=True, slots=True, kw_only=True)
class StartWindow:
    """When the waiting steps started, widened by `START_TIME_MARGIN` on both sides."""

    earliest: datetime
    latest: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class StepLookup:
    """One lookup to make: the steps it seeks, and for the stale lookup, how far it reaches.

    `stale_reach` is, on the sender's clock, the latest queueing time of the
    scores the stale lookup takes; None for the fresh lookup.
    """

    scores: list[PendingScore]
    stale_reach: float | None = None


@dataclass(slots=True, kw_only=True)
class LookupResult:
    """What lookups found: each step's observation by step id, and the pause a `429` asks for."""

    found: dict[str, LangfuseObservation] = field(default_factory=dict)
    pause_seconds: float | None = None


def split_by_queueing_time(
    scores: Sequence[PendingScore],
    *,
    reach: float,
) -> tuple[list[PendingScore], list[PendingScore]]:
    """Return the scores queued after `reach`, the fresh ones, and those queued at or before it."""
    fresh = [score for score in scores if score.queued_at > reach]
    stale = [score for score in scores if score.queued_at <= reach]
    return fresh, stale


def read_start_window(scores: Sequence[PendingScore]) -> StartWindow:
    """Return the window in which every waiting step's span started."""
    starts = [read_step_start(score.step_id) for score in scores]
    return StartWindow(
        earliest=min(starts) - START_TIME_MARGIN,
        latest=max(starts) + START_TIME_MARGIN,
    )


def build_step_filter(window: StartWindow) -> list[ObservationFilter]:
    """Return the filter for the `monitor step` observations that started within the window.

    The upper bound keeps each lookup to the steps that wait, so a busy
    project's later steps do not fill its pages.
    """
    return [
        ObservationFilter(type="string", column="name", operator="=", value=STEP_SPAN_NAME),
        ObservationFilter(
            type="datetime",
            column="startTime",
            operator=">=",
            value=window.earliest.isoformat(),
        ),
        ObservationFilter(
            type="datetime",
            column="startTime",
            operator="<=",
            value=window.latest.isoformat(),
        ),
    ]


def build_score_event(
    score: PendingScore,
    *,
    observation: LangfuseObservation,
) -> LangfuseScoreEvent:
    """Return the event that writes the score on its step's observation, in its environment."""
    body = LangfuseScoreBody(
        id=str(score.score_id),
        trace_id=observation.trace_id,
        observation_id=observation.id,
        name=score.name,
        value=score.value,
        environment=observation.environment or None,
    )
    return LangfuseScoreEvent(body=body)


def read_observation_page(response: httpx.Response | None) -> ObservationPage | None:
    """Return one page of observations, or None for a failed request, an error or an unknown shape.

    A failed request was logged where it was sent; an error answer or an
    unknown shape is logged here.
    """
    if response is None:
        return None
    if not response.is_success:
        logger.warning(
            "score export: Langfuse answered the step lookup with HTTP %d",
            response.status_code,
        )
        return None
    try:
        return ObservationPage.model_validate_json(response.content)
    except ValidationError:
        logger.warning("score export: Langfuse's step lookup answered in an unknown shape")
        return None


def match_observations(
    page: ObservationPage,
    *,
    wanted: set[str],
) -> dict[str, LangfuseObservation]:
    """Return the page's observations whose `monitor_step_id` is one of the wanted steps."""
    matched: dict[str, LangfuseObservation] = {}
    for observation in page.data:
        step_id = observation.metadata.monitor_step_id if observation.metadata else None
        if step_id is not None and step_id in wanted:
            matched[step_id] = observation
    return matched


def read_ingestion_answer(response: httpx.Response) -> IngestionAnswer:
    """Return the ingestion API's answer, or an empty one, which leaves every score waiting."""
    try:
        return IngestionAnswer.model_validate_json(response.content)
    except ValidationError:
        logger.warning("score export: Langfuse's ingestion API answered in an unknown shape")
        return IngestionAnswer()


def is_worth_sending_again(status: int) -> bool:
    """Tell whether an event refused with this status may pass later: a `429` or a server error."""
    return status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR


def record_ingestion_answer(
    report: DeliveryReport,
    *,
    events: dict[str, PendingScore],
    answer: IngestionAnswer,
) -> None:
    """Record each event's score by its status: written, waiting to be sent again, or refused."""
    for success in answer.successes:
        if success.id in events:
            report.written.append(events.pop(success.id))
    for error in answer.errors:
        if error.id not in events:
            continue
        score = events.pop(error.id)
        if is_worth_sending_again(error.status):
            report.waiting.append(score)
        else:
            report.refused.append(score)
            report.refusal = f"Langfuse answered HTTP {error.status} for a score"
    # An event the answer leaves out is sent again; its fixed score id keeps one score.
    report.waiting.extend(events.values())


class LangfuseScoreSender:
    """Finds each step's observation by its `monitor_step_id`, then writes the scores at once.

    Langfuse gives observations random ids [@langfuse2026traceids], so each
    step is found by the `monitor_step_id` its `monitor step` observation
    carries, through `GET /api/public/v2/observations`, the only real-time
    read path [@langfuse2026api]. One query serves many waiting steps: the
    observations named `monitor step` that started between the earliest and
    the latest of their starts, matched here by id, since the API filters
    metadata on one value only. Pages come newest first, and follow the
    cursor up to `MAX_OBSERVATION_PAGES`.

    A step Langfuse never ingests, such as one traced to another project,
    would keep that window wide for the whole wait, and its pages many. So the
    steps whose scores have waited `STALE_AFTER_SECONDS` on `clock`, the
    worker's clock, are looked up apart, together, at most once every
    `STALE_LOOKUP_SECONDS`, counted from the end of the last one that no
    `429` stopped; the fresh ones every window. A step that turns stale
    between two stale lookups is looked up with the fresh ones until the
    next takes it, so one that turns stale during the exit drain is still
    sought there, and the fresh lookup reaches back about two minutes at
    most. One process so asks
    the general rate limit, which every project and key of the organisation
    shares, for at most `MAX_OBSERVATION_PAGES` pages per lookup: 6 fresh
    lookups and 1 stale one a minute, 21 requests at most and 7 when each
    lookup fits a page, and twice the fresh ones during the exit drain
    [@langfuse2026apilimits]. The writes go to the ingestion bucket.

    The scores found go in one `POST /api/public/ingestion`, as
    `score-create` events, which is how Langfuse's own SDK sends scores
    [@langfuse2026]; the endpoint keeps taking score events when it stops
    taking any other on 16 November 2026 [@langfuse2026api]. The one-score
    endpoint, `POST /api/public/scores`, would spend one request per score of
    the general rate limit, 30 a minute on the Hobby plan, which the lookups
    spend too [@langfuse2026apilimits]. Each score is `NUMERIC`, on the
    step's trace and observation, in the observation's environment, with
    the score's fixed id, so a score written twice is stored once. No text,
    the monitor's reason included, leaves the process.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.http_client = http_client
        self.clock = clock
        self.last_stale_lookup: float | None = None
        self.stale_reach = -math.inf

    def send(self, scores: Sequence[PendingScore]) -> DeliveryReport:
        """Look up the steps due, fresh and stale apart, and write the scores found at once."""
        report = DeliveryReport()
        lookup = self.run_lookups(scores)
        report.pause_seconds = lookup.pause_seconds
        ready = [score for score in scores if str(score.step_id) in lookup.found]
        report.waiting.extend(score for score in scores if str(score.step_id) not in lookup.found)
        if report.pause_seconds is not None:
            report.waiting.extend(ready)
        elif ready:
            self.write_scores(ready, observations=lookup.found, report=report)
        return report

    def run_lookups(self, scores: Sequence[PendingScore]) -> LookupResult:
        """Make the lookups due, until one meets a `429`; note when each stale lookup ends."""
        result = LookupResult()
        for lookup in self.choose_lookups(scores):
            found = self.find_observations(lookup.scores)
            result.found.update(found.found)
            if found.pause_seconds is not None:
                result.pause_seconds = found.pause_seconds
                break
            if lookup.stale_reach is not None:
                self.last_stale_lookup = self.clock()
                self.stale_reach = lookup.stale_reach
        return result

    def is_stale_lookup_due(self, now: float) -> bool:
        """Tell whether the stale steps are due a lookup: the first, or a minute after the last."""
        last = self.last_stale_lookup
        return last is None or now - last >= STALE_LOOKUP_SECONDS

    def choose_lookups(self, scores: Sequence[PendingScore]) -> list[StepLookup]:
        """Return the lookups to make now: the fresh steps, and the stale ones when due.

        Until the stale lookup is due, only the steps the last one took wait;
        the steps that turned stale since are looked up with the fresh ones.
        """
        now = self.clock()
        due = self.is_stale_lookup_due(now)
        reach = now - STALE_AFTER_SECONDS if due else self.stale_reach
        fresh, stale = split_by_queueing_time(scores, reach=reach)
        lookups = [StepLookup(scores=fresh)] if fresh else []
        if stale and due:
            lookups.append(StepLookup(scores=stale, stale_reach=reach))
        return lookups

    def find_observations(self, scores: Sequence[PendingScore]) -> LookupResult:
        """Return the waiting steps' observations found, and the pause a `429` asks for, if any."""
        wanted = {str(score.step_id) for score in scores}
        window = read_start_window(scores)
        result = LookupResult()
        cursor: str | None = None
        for _ in range(MAX_OBSERVATION_PAGES):
            request = self.build_lookup(window=window, cursor=cursor)
            response = send_request(self.http_client, request=request)
            if is_rate_limited(response):
                result.pause_seconds = read_pause_seconds(response)
                break
            page = read_observation_page(response)
            if page is None:
                break
            result.found.update(match_observations(page, wanted=wanted))
            cursor = page.meta.cursor
            if cursor is None or wanted <= result.found.keys():
                break
        return result

    def build_lookup(self, *, window: StartWindow, cursor: str | None) -> httpx.Request:
        """Return the request for one page of the `monitor step` observations in the window."""
        parameters = {
            "fields": "core,basic,metadata",
            "limit": str(OBSERVATION_PAGE_SIZE),
            "filter": json.dumps(build_step_filter(window)),
        }
        if cursor is not None:
            parameters["cursor"] = cursor
        return self.http_client.build_request(
            "GET", "/api/public/v2/observations", params=parameters
        )

    def write_scores(
        self,
        ready: Sequence[PendingScore],
        *,
        observations: dict[str, LangfuseObservation],
        report: DeliveryReport,
    ) -> None:
        """Write the scores in one ingestion request, and record each by the answer to its event."""
        batch = [
            build_score_event(score, observation=observations[str(score.step_id)])
            for score in ready
        ]
        events = {event.id: score for event, score in zip(batch, ready, strict=True)}
        body = {
            "batch": [
                event.model_dump(mode="json", by_alias=True, exclude_none=True) for event in batch
            ]
        }
        request = self.http_client.build_request("POST", "/api/public/ingestion", json=body)
        response = send_request(self.http_client, request=request)
        if response is None or is_rate_limited(response):
            report.waiting.extend(ready)
            report.pause_seconds = None if response is None else read_pause_seconds(response)
        elif not response.is_success:
            report.refused.extend(ready)
            report.refusal = f"Langfuse answered HTTP {response.status_code}"
        else:
            record_ingestion_answer(report, events=events, answer=read_ingestion_answer(response))

    def close(self) -> None:
        """Close the sender's HTTP client."""
        self.http_client.close()


def build_langfuse_sender(credentials: LangfuseCredentials) -> LangfuseScoreSender:
    """Return a sender that authenticates with the project's key pair."""
    http_client = build_http_client(
        base_url=credentials.base_url,
        auth=(credentials.public_key, credentials.secret_key.get_secret_value()),
    )
    return LangfuseScoreSender(http_client=http_client)
