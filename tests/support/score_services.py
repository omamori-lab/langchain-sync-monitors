"""Fake LangSmith and Langfuse APIs on `httpx.MockTransport`, and a score worker built on them.

Each fake records every request it receives and keeps what was written by id,
so a score written twice is stored once, as the real services store it. A
test can queue answers, such as a `429` or a `500`, that the fake gives
before it answers as the service would.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.utils.uuid import uuid7
from pydantic import SecretStr

from langchain_sync_monitors import score_export
from langchain_sync_monitors.langfuse_scores import LangfuseScoreSender, read_step_start
from langchain_sync_monitors.langsmith_scores import (
    LangSmithFeedbackSender,
    build_langsmith_headers,
)
from langchain_sync_monitors.score_export import PROCESS_NOTICES, PROCESS_SCORE_WORKER
from langchain_sync_monitors.score_worker import ScoreWorker, WorkerTimings
from langchain_sync_monitors.scores import LangSmithCredentials, ScoreSender, Tracer
from tests.support.tracing import RecordingTracer

LANGSMITH_ENDPOINT = "https://langsmith.test/api/v1"
LANGFUSE_BASE_URL = "https://langfuse.test"
PROJECT_NAME = "monitor-scores"
PROJECT_ID = "5b1f7c1e-3e59-4f05-9a3c-0d7b8a0f1c42"
STEP_SPAN_NAME = "monitor step"
SPAN_START_DELAY = timedelta(milliseconds=500)
"""How long after its id is made a step's span starts, in the fake Langfuse."""
LANGSMITH_KEY = "test-langsmith-key"
LANGFUSE_KEYS = ("pk-test", "sk-test")
"""The public and secret keys `set_score_credentials` sets, which the fake's clients send."""
CONNECTION = LangSmithCredentials(
    api_key=SecretStr(LANGSMITH_KEY), endpoint=LANGSMITH_ENDPOINT, workspace_id=None
)
"""The connection the environment `set_score_credentials` sets up names."""


def read_request_json(request: httpx.Request) -> Any:
    return json.loads(request.content)


@dataclass
class FakeLangSmith:
    """LangSmith's feedback and project endpoints, under any connection's endpoint.

    Feedback on a run in `unknown_runs` is answered `404`, as for a run not
    ingested yet, and a feedback id seen before is answered `409`. Each
    client it builds carries the library's own headers for its connection,
    and `connections` records the connections clients were built for.
    """

    projects: dict[str, str] = field(default_factory=lambda: {PROJECT_NAME: PROJECT_ID})
    unknown_runs: set[str] = field(default_factory=set)
    queued_answers: list[httpx.Response] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    feedback: dict[str, dict[str, Any]] = field(default_factory=dict)
    connections: list[LangSmithCredentials] = field(default_factory=list)
    clients: list[httpx.Client] = field(default_factory=list)
    post_seconds: float = 0.0
    on_post: Callable[[], None] | None = None
    in_flight: int = 0
    most_in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def build_reply(self, request: httpx.Request) -> httpx.Response:
        """Answer as LangSmith would, after `post_seconds`, counting the requests in flight."""
        with self.lock:
            self.requests.append(request)
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
            answer = self.queued_answers.pop(0) if self.queued_answers else None
        try:
            time.sleep(self.post_seconds)
            if self.on_post is not None and request.method == "POST":
                self.on_post()
            return answer or self.build_answer(request)
        finally:
            with self.lock:
                self.in_flight -= 1

    def build_answer(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v1")
        if request.method == "GET" and path == "/sessions":
            name = request.url.params["name"]
            found = [{"id": self.projects[name], "name": name}] if name in self.projects else []
            return httpx.Response(200, json=found)
        if request.method == "POST" and path == "/feedback":
            body = read_request_json(request)
            if body["run_id"] in self.unknown_runs:
                return httpx.Response(404, json={"detail": "Run not found"})
            if body["id"] in self.feedback:
                return httpx.Response(409, json={"detail": "Feedback already exists"})
            self.feedback[body["id"]] = body
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"detail": "Not Found"})

    def build_client(self, connection: LangSmithCredentials = CONNECTION) -> httpx.Client:
        self.connections.append(connection)
        client = httpx.Client(
            base_url=connection.endpoint,
            headers=build_langsmith_headers(connection),
            transport=httpx.MockTransport(self.build_reply),
        )
        self.clients.append(client)
        return client

    def find_requests(self, method: str, path: str) -> list[httpx.Request]:
        return [
            request
            for request in self.requests
            if request.method == method and request.url.path == f"/api/v1{path}"
        ]


def is_matching(observation: dict[str, Any], conditions: list[dict[str, str]]) -> bool:
    """Tell whether an observation meets every name and start-time condition of a lookup."""
    for condition in conditions:
        if condition["column"] == "name" and observation["name"] != condition["value"]:
            return False
        if condition["column"] == "startTime":
            started = datetime.fromisoformat(observation["startTime"])
            bound = datetime.fromisoformat(condition["value"])
            if condition["operator"] == ">=" and started < bound:
                return False
            if condition["operator"] == "<=" and started > bound:
                return False
    return True


@dataclass
class FakeLangfuse:
    """Langfuse's v2 observations and ingestion endpoints.

    `observations` hold what the service has ingested; `ingest_steps` adds
    the `monitor step` runs a tracer recorded. Scores are kept by their body
    id, so a score sent twice is stored once.
    """

    observations: list[dict[str, Any]] = field(default_factory=list)
    queued_answers: list[httpx.Response] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    page_size: int | None = None
    ingestion_errors: dict[str, int] = field(default_factory=dict)

    def build_reply(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.queued_answers:
            return self.queued_answers.pop(0)
        if request.method == "GET" and request.url.path == "/api/public/v2/observations":
            return self.build_lookup_reply(request)
        if request.method == "POST" and request.url.path == "/api/public/ingestion":
            return self.build_ingestion_reply(request)
        return httpx.Response(404, json={"message": "Not Found"})

    def build_lookup_reply(self, request: httpx.Request) -> httpx.Response:
        conditions = json.loads(request.url.params["filter"])
        matching = [item for item in self.observations if is_matching(item, conditions)]
        start = int(request.url.params.get("cursor", "0"))
        size = self.page_size or len(matching) or 1
        page = matching[start : start + size]
        meta = {"cursor": str(start + size)} if start + size < len(matching) else {}
        return httpx.Response(200, json={"data": page, "meta": meta})

    def build_ingestion_reply(self, request: httpx.Request) -> httpx.Response:
        successes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for event in read_request_json(request)["batch"]:
            status = self.ingestion_errors.pop(event["body"]["observationId"], None)
            if status is not None:
                errors.append({"id": event["id"], "status": status, "message": "refused"})
                continue
            self.scores[event["body"]["id"]] = event["body"]
            successes.append({"id": event["id"], "status": 201})
        return httpx.Response(207, json={"successes": successes, "errors": errors})

    def add_step(
        self,
        step_id: str,
        *,
        environment: str = "default",
        started: datetime | None = None,
    ) -> dict[str, Any]:
        """Ingest a step whose span started at `started`, or, as spans do, just after its id."""
        if started is None:
            started = read_step_start(UUID(step_id)) + SPAN_START_DELAY
        observation = {
            "id": uuid4().hex[:16],
            "traceId": uuid4().hex,
            "name": STEP_SPAN_NAME,
            "startTime": started.isoformat(),
            "environment": environment,
            "metadata": {"monitor_step_id": step_id, "monitor_label": "monitor"},
        }
        self.observations.append(observation)
        return observation

    def ingest_steps(self, tracer: RecordingTracer) -> None:
        """Add an observation for every `monitor step` run the tracer recorded."""
        known = {item["metadata"]["monitor_step_id"] for item in self.observations}
        for run in tracer.find_runs(STEP_SPAN_NAME):
            if str(run.run_id) not in known:
                self.add_step(str(run.run_id))

    def build_client(self, *, auth: tuple[str, str] = LANGFUSE_KEYS) -> httpx.Client:
        """Return a client that reaches this fake with a key pair, as the library's does."""
        return httpx.Client(
            base_url=LANGFUSE_BASE_URL,
            auth=auth,
            transport=httpx.MockTransport(self.build_reply),
        )

    def find_requests(self, method: str, path: str) -> list[httpx.Request]:
        return [
            request
            for request in self.requests
            if request.method == method and request.url.path == path
        ]


class LangfuseHandler(RecordingTracer):
    """Stands in for Langfuse's LangChain handler: it lives in a `langfuse.` module."""


LangfuseHandler.__module__ = "langfuse.langchain.CallbackHandler"


def build_langsmith_tracer(client: Any, *, project_name: str = PROJECT_NAME) -> LangChainTracer:
    """Return LangSmith's tracer on a mock client, which records runs and sends nothing."""
    return LangChainTracer(client=client, project_name=project_name)


@dataclass
class FakeClock:
    """A monotonic clock a test moves by hand; `sleep` moves it too."""

    now: float = 1000.0
    sleeps: list[float] = field(default_factory=list)

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class ScoreServices:
    """Both fake services, the worker that writes to them, and its clock."""

    langsmith: FakeLangSmith
    langfuse: FakeLangfuse
    clock: FakeClock
    worker: ScoreWorker

    def send_window(self, *, after_seconds: float = 10.0) -> None:
        """Move the clock by a window, and let the worker send once."""
        self.clock.now += after_seconds
        self.worker.send_window()


def build_fake_sender(
    tracer: Tracer,
    *,
    langsmith: FakeLangSmith,
    langfuse: FakeLangfuse,
    clock: FakeClock,
) -> ScoreSender:
    """Return a sender on the fake service, posting one at a time, on the worker's clock."""
    if tracer is Tracer.LANGSMITH:
        return LangSmithFeedbackSender(
            build_client=langsmith.build_client, posts_in_flight=1, clock=clock.read
        )
    return LangfuseScoreSender(http_client=langfuse.build_client(), clock=clock.read)


def set_score_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test credentials for both tools, turn tracing by variable off, and fake langfuse."""
    monkeypatch.setenv("LANGSMITH_API_KEY", LANGSMITH_KEY)
    monkeypatch.setenv("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", LANGFUSE_KEYS[0])
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", LANGFUSE_KEYS[1])
    monkeypatch.setenv("LANGFUSE_BASE_URL", LANGFUSE_BASE_URL)
    for name in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGSMITH_WORKSPACE_ID",
        "LANGCHAIN_WORKSPACE_ID",
        "LANGFUSE_TRACING_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    installed = score_export.is_package_installed
    monkeypatch.setattr(
        score_export,
        "is_package_installed",
        lambda name: name == "langfuse" or installed(name),
    )


def install_score_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    timings: WorkerTimings | None = None,
    build_sender: Callable[[Tracer], ScoreSender | None] | None = None,
) -> ScoreServices:
    """Make the process's score worker one on the fake services, never started, on a fake clock."""
    langsmith = FakeLangSmith()
    langfuse = FakeLangfuse()
    clock = FakeClock()
    worker = ScoreWorker(
        build_sender=build_sender
        or (
            lambda tracer: build_fake_sender(
                tracer, langsmith=langsmith, langfuse=langfuse, clock=clock
            )
        ),
        timings=timings,
        clock=clock.read,
        sleep=clock.sleep,
    )
    monkeypatch.setattr(PROCESS_SCORE_WORKER, "worker", worker)
    monkeypatch.setattr(PROCESS_SCORE_WORKER, "process_id", os.getpid())
    return ScoreServices(langsmith=langsmith, langfuse=langfuse, clock=clock, worker=worker)


@pytest.fixture
def score_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[ScoreServices]:
    """Credentials for both tools, a process score worker on the fake services, no notices said."""
    set_score_credentials(monkeypatch)
    monkeypatch.setattr(PROCESS_NOTICES, "said", set())
    services = install_score_worker(monkeypatch)
    yield services
    services.worker.close_senders()


def build_step_id() -> UUID:
    """Return a step id as the monitor makes one: a version 7 UUID."""
    return uuid7()
