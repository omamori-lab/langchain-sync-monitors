"""LangSmith's and Langfuse's ends of the wire, and the tracers that reach them.

LangSmith's feedback goes through a `langsmith.Client`, which takes no
transport, so its stand-in is a server on 127.0.0.1 that records each request
and answers as LangSmith does. Langfuse's client takes an `httpx` client, so
its stand-in is a mock transport, under Langfuse's own handler, tracer
provider and span exporter: the ids a score names can be read from the step
span the handler exported.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from uuid import UUID

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from langchain_sync_monitors.request_pool import PROCESS_POOL, RequestPool
from langchain_sync_monitors.score_export import PROCESS_NOTICES, Tracer
from tests.support.tracing import RecordedRun, RecordingTracer

PROJECT_NAME = "monitor-scores"
PROJECT_ID = "5b1f7c1e-3e59-4f05-9a3c-0d7b8a0f1c42"
TENANT_ID = "0b6c8f2e-7d41-4c1a-9d57-3a2e5f8b9c10"
STEP_SPAN_NAME = "monitor step"
FLAG_NAME = "flagged"
NOT_A_STEP = 0
"""The step number `read_sent` gives a score or flag that sits on a run that is no step span."""
ANSWER_HOLD_SECONDS = 5.0
"""How long the stand-in holds a feedback answer that is not released."""
ARRIVAL_SECONDS = 5.0
"""How long a test waits for a request to reach a stand-in."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LangSmithRequest:
    """One request the LangSmith stand-in received."""

    method: str
    path: str
    query: str
    body: bytes


class LangSmithHandler(BaseHTTPRequestHandler):
    """Records a request on the server and answers it as LangSmith does."""

    def do_GET(self) -> None:
        self.write_reply()

    def do_POST(self) -> None:
        self.write_reply()

    def do_PATCH(self) -> None:
        self.write_reply()

    def write_reply(self) -> None:
        server = cast("LangSmithServer", self.server)
        path, _, query = self.path.partition("?")
        length = int(self.headers.get("Content-Length") or 0)
        request = LangSmithRequest(
            method=self.command, path=path, query=query, body=self.rfile.read(length)
        )
        server.remember(request)
        status = 200
        reply: object = {}
        if path == "/sessions":
            server.release.wait(ANSWER_HOLD_SECONDS)
            reply = [
                {
                    "id": PROJECT_ID,
                    "name": PROJECT_NAME,
                    "tenant_id": TENANT_ID,
                    "reference_dataset_id": None,
                }
            ]
            status = server.sessions_status
            server.record_sessions_answer(status)
        elif path == "/feedback":
            server.release.wait(ANSWER_HOLD_SECONDS)
            refused = json.loads(request.body)["key"] in server.fail_keys
            status = 400 if refused else server.feedback_status
            server.record_answer(status)
        body = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Log nothing: the base class writes each request to standard error."""


class LangSmithServer(ThreadingHTTPServer):
    """LangSmith's end of the wire: the project and feedback endpoints, and a 200 for the rest.

    A feedback or project answer waits until `release` is set. `feedback_status`
    and `sessions_status` are the statuses it answers those two endpoints with,
    and a feedback whose key is in `fail_keys` is answered 400, which the SDK
    does not send again.
    """

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), LangSmithHandler)
        self.requests: list[LangSmithRequest] = []
        self.feedback_status = 200
        self.sessions_status = 200
        self.release = threading.Event()
        self.fail_keys: set[str] = set()
        self.answered: list[int] = []
        self.sessions_answered: list[int] = []
        self.arrived = threading.Condition()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def remember(self, request: LangSmithRequest) -> None:
        with self.arrived:
            self.requests.append(request)
            self.arrived.notify_all()

    def record_answer(self, status: int) -> None:
        """Note that a feedback request was answered, with its status."""
        with self.arrived:
            self.answered.append(status)
            self.arrived.notify_all()

    def record_sessions_answer(self, status: int) -> None:
        """Note that a project request was answered, with its status."""
        with self.arrived:
            self.sessions_answered.append(status)
            self.arrived.notify_all()

    def wait_for_answers(self, count: int) -> list[int]:
        """Return the statuses answered once `count` are, or after the wait."""
        with self.arrived:
            self.arrived.wait_for(lambda: len(self.answered) >= count, timeout=ARRIVAL_SECONDS)
            return list(self.answered)

    def read_feedback(self) -> list[dict[str, Any]]:
        """Return the body of each feedback received, in the order received."""
        with self.arrived:
            return [
                json.loads(request.body)
                for request in self.requests
                if request.method == "POST" and request.path == "/feedback"
            ]

    def read_requests(self, method: str, path: str) -> list[LangSmithRequest]:
        return [
            request
            for request in self.requests
            if request.method == method and request.path == path
        ]


@pytest.fixture(scope="session")
def langsmith_server() -> Iterator[LangSmithServer]:
    server = LangSmithServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def langsmith(langsmith_server: LangSmithServer) -> LangSmithServer:
    """The stand-in server with nothing received, answering 200 to everything at once."""
    with langsmith_server.arrived:
        langsmith_server.requests.clear()
        langsmith_server.answered.clear()
        langsmith_server.sessions_answered.clear()
    langsmith_server.fail_keys.clear()
    langsmith_server.feedback_status = 200
    langsmith_server.sessions_status = 200
    langsmith_server.release.set()
    return langsmith_server


class OtelModeClient(Client):
    """A client that says it sends through OpenTelemetry, as one built in `otel` mode does."""

    @property
    def tracing_mode(self) -> Any:
        return "otel"


def build_langsmith_tracer(
    server: LangSmithServer,
    *,
    client_class: type[Client] = Client,
) -> LangChainTracer:
    """Return LangSmith's tracer on a client that reaches the stand-in server."""
    client = client_class(
        api_url=server.url,
        api_key="lsv2-test",
        # The server's details, given so the client does not ask for them.
        info={},
        auto_batch_tracing=False,
    )
    return LangChainTracer(client=client, project_name=PROJECT_NAME)


@pytest.fixture
def inline_pool(monkeypatch: pytest.MonkeyPatch) -> RequestPool:
    """Make the process's request pool run every call in turn, so a send ends with its step."""
    pool = RequestPool(size=1)
    monkeypatch.setattr(PROCESS_POOL, "pool", pool)
    monkeypatch.setattr(PROCESS_POOL, "process_id", os.getpid())
    return pool


@pytest.fixture
def fresh_notices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget the warnings this process has said once."""
    monkeypatch.setattr(PROCESS_NOTICES, "said", set())


class FakeLangfuseHandler(RecordingTracer):
    """A handler from a `langfuse` module that keeps neither the client nor the runs a handler of
    Langfuse's SDK keeps.
    """


FakeLangfuseHandler.__module__ = "langfuse.langchain.CallbackHandler"


class MisbuiltLangfuseHandler(FakeLangfuseHandler):
    """A handler from a `langfuse` module that keeps an observation for each run, as Langfuse's
    does, but a client that is not Langfuse's.
    """

    def __init__(self) -> None:
        super().__init__()
        self._langfuse_client = object()
        self._runs: dict[UUID, SimpleNamespace] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._runs[run_id] = SimpleNamespace(trace_id="1" * 32, id="2" * 16)
        super().on_chain_start(
            serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )


class LangfuseWire:
    """Langfuse's end of the wire: records each request's path and JSON body, accepts them all."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content or b"null")))
        # The API accepts a batch with 202.
        return httpx.Response(202, json={"message": "Accepted"})


@dataclass(slots=True, kw_only=True)
class LangfuseStandIn:
    """A Langfuse client on a stand-in wire, with the span exporter its handler writes to."""

    public_key: str
    client: Any
    wire: LangfuseWire
    exporter: Any

    def build_handler(self) -> Any:
        """Return Langfuse's own LangChain handler, tracing through this client."""
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(public_key=self.public_key)

    def read_scores(self) -> list[dict[str, Any]]:
        """Return the scores the wire received, once the client has sent every queued score."""
        self.client.flush()
        return [
            score
            for _, body in self.wire.requests
            for score in (body if isinstance(body, list) else [body])
            if isinstance(score, dict) and "name" in score
        ]

    def read_step_ids(self) -> list[tuple[str, str]]:
        """Return the trace id and observation id of each `monitor step` span, as Langfuse writes
        them: lower-case hex of 32 and 16 digits, in the order the spans started.
        """
        self.client.flush()
        steps = sorted(
            (span for span in self.exporter.get_finished_spans() if span.name == STEP_SPAN_NAME),
            key=lambda span: span.start_time,
        )
        return [
            (format(span.context.trace_id, "032x"), format(span.context.span_id, "016x"))
            for span in steps
        ]

    def read_flag_places(self) -> list[tuple[str, str]]:
        """Return the name of the span of each flag received, and of that span's parent."""
        spans = {span.context.span_id: span for span in self.exporter.get_finished_spans()}
        places = []
        for score in self.read_scores():
            if score["name"] == FLAG_NAME:
                span = spans[int(score["observationId"], 16)]
                places.append((span.name, spans[span.parent.span_id].name))
        return places


@contextmanager
def build_langfuse(**options: Any) -> Generator[LangfuseStandIn]:
    """Build a Langfuse client on a stand-in wire; `options` are the client's own, such as
    `tracing_enabled`. The tests that use it need the `langfuse` extra.
    """
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    wire = LangfuseWire()
    exporter = InMemorySpanExporter()
    # Langfuse keeps one client's resources per public key in a process.
    public_key = f"pk-lf-{uuid.uuid4()}"
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url="https://langfuse.test",
        httpx_client=httpx.Client(transport=httpx.MockTransport(wire.respond)),
        # A score is sent as soon as it is queued, not in batches of 15.
        flush_at=1,
        # Its own provider and exporter, so nothing joins the global provider or leaves the
        # process.
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        **options,
    )
    try:
        yield LangfuseStandIn(public_key=public_key, client=client, wire=wire, exporter=exporter)
    finally:
        client.shutdown()


@pytest.fixture
def langfuse() -> Iterator[LangfuseStandIn]:
    """A Langfuse client on a stand-in wire."""
    with build_langfuse() as stand_in:
        yield stand_in


type Sent = dict[tuple[int, str], list[tuple[float | bool, str | None]]]
"""What a tool received: for each step number and score name, each value with its comment."""

NUMERIC_METADATA = {"failed_closed": False, "cause": None, "model": "keyword"}
"""The metadata of a score the keyword monitor gave: it did not fail closed, and its model."""


@dataclass(slots=True, kw_only=True)
class LangSmithBackend:
    """LangSmith as a run reaches it: its tracer on the stand-in server, and a recorder of runs."""

    failure_type: ClassVar[str] = "LangSmithError"
    """The class of the error the SDK raises for a feedback the server refuses."""

    tracer_member: Tracer
    server: LangSmithServer
    tracer: LangChainTracer
    recorder: RecordingTracer

    def build_handlers(self) -> list[BaseCallbackHandler]:
        return [self.tracer, self.recorder]

    def fail_score(self, name: str) -> None:
        """Make the server refuse every feedback named `name`."""
        self.server.fail_keys.add(name)

    def read_sent(self) -> Sent:
        """Return the feedback accepted, by the number of the step run it names."""
        steps = {str(run.run_id): number for number, run in enumerate(self.read_step_runs(), 1)}
        sent: Sent = {}
        for item in self.read_accepted():
            key = (steps.get(item["run_id"], NOT_A_STEP), item["key"])
            sent.setdefault(key, []).append((item["score"], item.get("comment")))
        return sent

    def read_accepted(self) -> list[dict[str, Any]]:
        return [
            item for item in self.server.read_feedback() if item["key"] not in self.server.fail_keys
        ]

    def read_step_runs(self) -> list[RecordedRun]:
        return self.recorder.find_runs(STEP_SPAN_NAME)

    def read_flag_places(self) -> list[tuple[str, str]]:
        """Return the name of the run of each flag accepted, and of that run's parent."""
        runs = [
            self.recorder.runs[uuid.UUID(item["run_id"])]
            for item in self.read_accepted()
            if item["key"] == FLAG_NAME
        ]
        return [(run.name, self.recorder.find_parent(run).name) for run in runs]

    def read_metadata(self) -> list[dict[str, Any]]:
        """Return the metadata of each score accepted that is not a flag."""
        return [
            item["feedback_source"]["metadata"]
            for item in self.read_accepted()
            if item["key"] != FLAG_NAME
        ]


@dataclass(slots=True, kw_only=True)
class LangfuseBackend:
    """Langfuse as a run reaches it: the SDK's handler on a client with a stand-in wire."""

    failure_type: ClassVar[str] = "RuntimeError"
    """The class of the error the failing client raises."""

    tracer_member: Tracer
    stand_in: LangfuseStandIn
    handler: Any
    monkeypatch: pytest.MonkeyPatch

    def build_handlers(self) -> list[BaseCallbackHandler]:
        return [self.handler]

    def fail_score(self, name: str) -> None:
        """Make the Langfuse client raise for every score named `name`."""
        from langfuse import Langfuse

        create_score = Langfuse.create_score

        def refuse(client: Langfuse, **keywords: Any) -> Any:
            if keywords.get("name") == name:
                message = f"the client refused {name}"
                raise RuntimeError(message)
            return create_score(client, **keywords)

        self.monkeypatch.setattr(Langfuse, "create_score", refuse)

    def read_sent(self) -> Sent:
        """Return the scores received, by the number of the step span they name."""
        steps = {ids: number for number, ids in enumerate(self.stand_in.read_step_ids(), 1)}
        sent: Sent = {}
        for score in self.stand_in.read_scores():
            ids = (score["traceId"], score["observationId"])
            key = (steps.get(ids, NOT_A_STEP), score["name"])
            sent.setdefault(key, []).append((score["value"], score.get("comment")))
        return sent

    def read_flag_places(self) -> list[tuple[str, str]]:
        return self.stand_in.read_flag_places()

    def read_metadata(self) -> list[dict[str, Any]]:
        return [
            score["metadata"] for score in self.stand_in.read_scores() if score["name"] != FLAG_NAME
        ]


type Backend = LangSmithBackend | LangfuseBackend


@pytest.fixture(params=["langsmith", "langfuse"])
def backend(
    request: pytest.FixtureRequest,
    langsmith: LangSmithServer,
    inline_pool: RequestPool,
    fresh_notices: None,
    monkeypatch: pytest.MonkeyPatch,
) -> Backend:
    """Each tool in turn, as a run reaches it, with every send finished when its step is."""
    if request.param == "langsmith":
        return LangSmithBackend(
            tracer_member=Tracer.LANGSMITH,
            server=langsmith,
            tracer=build_langsmith_tracer(langsmith),
            recorder=RecordingTracer(),
        )
    stand_in = request.getfixturevalue("langfuse")
    return LangfuseBackend(
        tracer_member=Tracer.LANGFUSE,
        stand_in=stand_in,
        handler=stand_in.build_handler(),
        monkeypatch=monkeypatch,
    )
