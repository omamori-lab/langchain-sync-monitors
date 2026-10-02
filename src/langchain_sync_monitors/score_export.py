"""The `export_scores` option: each monitored step's suspicion, as a score in the tracing tools.

`MonitorMiddleware(export_scores={Tracer.LANGSMITH, Tracer.LANGFUSE})` sends
each committed step's highest suspicion to the tools named, off by default.
The score is named `<label>_suspicion`, `monitor_suspicion` under the default
label, and sits on the step's `monitor step` span: feedback on its run in
LangSmith, a numeric score on its observation in Langfuse. Both tools filter,
sort and chart these, which they do not with metadata [@langsmith2026dashboards;
@langfuse2026scores].

- **Checked when the monitor is built.** A tool whose credentials are
  missing, or Langfuse without the `langfuse` package, raises
  `ConfigurationError`. The credentials are the variables each tool's SDK
  reads: `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY`, and
  `LANGFUSE_PUBLIC_KEY` with `LANGFUSE_SECRET_KEY`. The LangSmith key must
  be in the environment even when a tracer's own client holds one, since
  that client is known only during a run.
- **Only a traced run sends, and only where it is traced.** The handlers of
  the step span name the tools the run is traced to. For LangSmith's
  `LangChainTracer`, the feedback goes to the tracer's project, through its
  client's endpoint, key and workspace. Langfuse's LangChain handler is
  recognised by its package without importing it. The Langfuse writer uses
  the environment's keys, and scores only steps it finds in that project by
  their `monitor_step_id`; a handler built with other keys traces to a
  project whose steps it never finds, so it writes nothing there, and warns
  once. A run traced to neither tool sends nothing, and says nothing, nor
  does a run with LangSmith tracing turned off, nor one whose Langfuse
  tracing is off, by its variable or by a client built with
  `tracing_enabled=False` or `sample_rate=0`; a LangSmith client in
  OpenTelemetry mode sends nothing, with one warning.
- **A step with no sample writes no score**, such as a step that halts
  because an earlier step was halted.
- **The agent never waits.** The score goes on the queue of one worker
  thread per process, which `score_worker` describes: it writes to
  LangSmith within a window of the step, and to Langfuse once Langfuse has
  ingested the step, usually 10 to 25 seconds after it in our checks and at
  times more than 30. At exit it drains what waits, for up to 30 seconds,
  before Langfuse's own exit flush. The last steps' Langfuse scores are
  written only if the program flushed Langfuse, or its flush interval sent
  the spans, and Langfuse ingests them within the drain; otherwise they are
  dropped. Nothing it does raises into a run; its failures are logged by
  `langchain_sync_monitors.score_worker`.
- **Only numbers and ids leave the process**: the monitor's reason is never
  sent, since neither tool masks feedback comments or score comments.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.run_helpers import get_tracing_context

from langchain_sync_monitors._langchain import TracedRun
from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import build_langfuse_sender, read_langfuse_credentials
from langchain_sync_monitors.langsmith_scores import (
    LangSmithFeedbackSender,
    read_client_connection,
    read_langsmith_credentials,
)
from langchain_sync_monitors.model_calls import is_package_installed
from langchain_sync_monitors.options import check_enum_option, describe_option_value
from langchain_sync_monitors.score_requests import PROCESS_ORIGIN
from langchain_sync_monitors.score_worker import ScoreWorker
from langchain_sync_monitors.scores import (
    LangSmithCredentials,
    PendingScore,
    ScoreSender,
    Tracer,
)
from langchain_sync_monitors.spans import find_max_suspicion

logger = logging.getLogger(__name__)

SCORE_NAME_SUFFIX = "_suspicion"
"""What follows a monitor's label in the name of its score."""

LANGFUSE_PACKAGE = "langfuse"
"""The top-level package of Langfuse's SDK, whose LangChain handler traces a run to Langfuse."""

OTEL_MODE_NOTICE = (
    "score export: LangSmith's tracer sends this run through OpenTelemetry, whose run ids are "
    "not the steps' ids, so no LangSmith score is sent in this process while it does"
)
"""Said once per process when a LangSmith client runs in OpenTelemetry mode."""


@dataclass(slots=True, kw_only=True)
class ProcessNotices:
    """The warnings this process says once, such as that LangSmith's OTel mode gets no score."""

    said: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def warn_once(self, message: str) -> None:
        """Log the warning the first time it comes up in this process."""
        with self.lock:
            if message in self.said:
                return
            self.said.add(message)
        logger.warning(message)


PROCESS_NOTICES = ProcessNotices()
"""The warnings this process has said."""


def build_score_name(label: str) -> str:
    """Return the name of a monitor's score: its label and `_suspicion`."""
    return f"{label}{SCORE_NAME_SUFFIX}"


def check_score_export(tracers: object) -> frozenset[Tracer]:
    """Return the tools as a frozenset, or raise `ConfigurationError` for any that cannot work.

    `tracers` must be a set of `Tracer` members; a plain string is refused.
    Each tool needs its credentials in the environment, and Langfuse also
    needs the `langfuse` package, without which no run is traced to it.
    """
    if not isinstance(tracers, AbstractSet):
        message = (
            "export_scores must be a set of Tracer members, such as "
            f"{{Tracer.LANGSMITH, Tracer.LANGFUSE}}, got {describe_option_value(tracers)}"
        )
        raise ConfigurationError(message)
    for tracer in tracers:
        check_enum_option(tracer, option_type=Tracer, parameter_name="export_scores")
    for tracer in sorted(tracers):
        check_tool_requirements(tracer)
    return frozenset(tracers)


def check_tool_requirements(tracer: Tracer) -> None:
    """Raise `ConfigurationError` when the tool's package or credentials are missing."""
    if tracer is Tracer.LANGSMITH:
        if read_langsmith_credentials() is None:
            message = (
                "export_scores asks for Tracer.LANGSMITH, but neither LANGSMITH_API_KEY nor "
                "LANGCHAIN_API_KEY is set: set the key LangSmith's tracer uses in the "
                "environment, even when the tracer's own client holds it, or leave "
                "Tracer.LANGSMITH out"
            )
            raise ConfigurationError(message)
        return
    if not is_package_installed(LANGFUSE_PACKAGE):
        message = (
            "export_scores asks for Tracer.LANGFUSE, but the langfuse package, whose "
            "LangChain handler traces a run to Langfuse, is not installed. "
            "Install it with: uv add langfuse (or pip install langfuse)"
        )
        raise ConfigurationError(message)
    if read_langfuse_credentials() is None:
        message = (
            "export_scores asks for Tracer.LANGFUSE, but LANGFUSE_PUBLIC_KEY or "
            "LANGFUSE_SECRET_KEY is not set: set the keys Langfuse's handler uses"
        )
        raise ConfigurationError(message)


def is_langfuse_handler(handler: BaseCallbackHandler) -> bool:
    """Tell whether a handler is Langfuse's LangChain handler, or built on it, without an import."""
    return any(
        cls.__module__.partition(".")[0] == LANGFUSE_PACKAGE for cls in type(handler).__mro__
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreDestination:
    """Where one score goes: a tool, and for LangSmith the project and the connection."""

    tracer: Tracer
    project: str | None = None
    connection: LangSmithCredentials | None = None


def is_langfuse_tracing_off() -> bool:
    """Tell whether Langfuse's SDK traces nothing, as `LANGFUSE_TRACING_ENABLED=false` asks.

    The test is the SDK's own: the variable, lowercased, is `false` [@langfuse2026].
    """
    return os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false"


def read_langsmith_destination(tracer: LangChainTracer) -> ScoreDestination | None:
    """Return where a LangSmith tracer sends the run, or None when no score can go there.

    None, quietly, when tracing is off for the run, as `tracing_context(enabled=False)`
    turns it off, which the tracer itself checks [@langchaincore2026]. None,
    warned once, when the client runs in OpenTelemetry mode, whose run ids are
    derived from span ids, so that no feedback could find its step
    [@langsmithsdk2026]. None, with a warning, when neither the client nor the
    environment has a key.
    """
    if get_tracing_context().get("enabled") is False:
        return None
    if getattr(tracer.client, "tracing_mode", None) == "otel":
        PROCESS_NOTICES.warn_once(OTEL_MODE_NOTICE)
        return None
    connection = read_client_connection(tracer.client)
    if connection is None:
        logger.warning("score export: the LangSmith tracer has no API key, so no score is sent")
        return None
    return ScoreDestination(
        tracer=Tracer.LANGSMITH, project=tracer.project_name, connection=connection
    )


def is_langfuse_handler_silent(handler: BaseCallbackHandler) -> bool:
    """Tell whether a Langfuse handler's client sends no trace, so that no step of it is found.

    Without this test, each score of such a run waits for a step that never
    arrives, and every exit spends the whole drain on it. Langfuse exposes
    neither switch publicly [@langfuse2026], so three private reads make the
    test, all of them on Langfuse's objects and none on LangChain's:

    - `handler._langfuse_client`, the client the handler traces through,
      which the handler keeps only there. Without one, it counts as tracing.
    - `client._tracing_enabled`, which `Langfuse(tracing_enabled=False)`
      sets false. Unless it is exactly `False`, the client counts as tracing.
    - `client._resources.tracer_provider.sampler.rate`, which `sample_rate=0`
      sets to 0 on the sampler of the client's tracer provider. Unless it is
      a number of 0 or below, the client counts as tracing.

    Each read is a `getattr` with that fallback, so a handler or client of
    another shape, such as a later SDK's, never raises, and its scores wait
    as they would have. The reads stay in this one function, since
    `_langchain.py` holds LangChain's untyped surfaces, not Langfuse's.
    """
    client = getattr(handler, "_langfuse_client", None)
    if getattr(client, "_tracing_enabled", True) is False:
        return True
    provider = getattr(getattr(client, "_resources", None), "tracer_provider", None)
    rate = getattr(getattr(provider, "sampler", None), "rate", None)
    return isinstance(rate, int | float) and not isinstance(rate, bool) and rate <= 0


def read_langfuse_destination(handler: BaseCallbackHandler) -> ScoreDestination | None:
    """Return Langfuse as the destination, or None when its tracing is off or samples nothing.

    Tracing is off by its variable or by the handler's client, and the client
    samples nothing at a sample rate of 0.
    """
    if is_langfuse_tracing_off() or is_langfuse_handler_silent(handler):
        return None
    return ScoreDestination(tracer=Tracer.LANGFUSE)


def read_destination(
    handler: BaseCallbackHandler,
    *,
    tracers: AbstractSet[Tracer],
) -> ScoreDestination | None:
    """Return where the handler traces to, if it is the tracer of a tool among `tracers`."""
    if Tracer.LANGSMITH in tracers and isinstance(handler, LangChainTracer):
        return read_langsmith_destination(handler)
    if Tracer.LANGFUSE in tracers and is_langfuse_handler(handler):
        return read_langfuse_destination(handler)
    return None


def find_score_destinations(
    handlers: Sequence[BaseCallbackHandler],
    *,
    tracers: AbstractSet[Tracer],
) -> list[ScoreDestination]:
    """Return the tools among `tracers` that these handlers trace to, each project once."""
    destinations = (read_destination(handler, tracers=tracers) for handler in handlers)
    return list(dict.fromkeys(item for item in destinations if item is not None))


def build_score_sender(tracer: Tracer) -> ScoreSender | None:
    """Return the tool's sender: LangSmith's, or Langfuse's on the environment's keys, if any."""
    if tracer is Tracer.LANGSMITH:
        # Each LangSmith score carries its tracer's connection.
        return LangSmithFeedbackSender()
    langfuse = read_langfuse_credentials()
    return None if langfuse is None else build_langfuse_sender(langfuse)


def build_score_worker() -> ScoreWorker:
    """Return a worker whose senders use the credentials in the environment."""
    return ScoreWorker(build_sender=build_score_sender)


@dataclass(slots=True, kw_only=True)
class ProcessScoreWorker:
    """The one score worker of this process, started with the first score queued.

    Its thread drains at exit through `atexit`. The lock is taken only to
    start the worker, never to queue a score. A child forked from this
    process forgets the parent's worker and lock, whose thread and holder did
    not come with it, and starts a worker of its own; the parent drains what
    it queued. The child's worker sends as the parent's does, except that
    on macOS it reads proxies only from the environment, never from System
    Settings, since that lookup kills a forked child (see
    `score_requests.is_system_proxy_lookup_safe`). A `multiprocessing` child
    started by fork leaves through `os._exit`, so the scores still waiting
    then are dropped.
    """

    build_worker: Callable[[], ScoreWorker] = build_score_worker
    worker: ScoreWorker | None = None
    process_id: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def read_worker(self) -> ScoreWorker:
        """Return the process's worker, starting it and its exit drain the first time."""
        worker = self.worker
        if worker is not None and self.process_id == os.getpid():
            return worker
        with self.lock:
            # Another thread may have started the worker while this one waited for the lock.
            if self.worker is None or self.process_id != os.getpid():
                self.worker = self.start_worker()
            return self.worker

    def start_worker(self) -> ScoreWorker:
        """Start a worker for this process, with its exit drain in place of any older one's.

        The caller holds the lock.
        """
        if self.worker is not None:
            atexit.unregister(self.worker.stop)
        worker = self.build_worker()
        worker.start()
        atexit.register(worker.stop)
        self.process_id = os.getpid()
        return worker

    def forget_after_fork(self) -> None:
        """In a forked child, drop the parent's worker, its exit drain and its lock."""
        if self.worker is not None:
            atexit.unregister(self.worker.stop)
        self.lock = threading.Lock()
        self.worker = None
        self.process_id = None


PROCESS_SCORE_WORKER = ProcessScoreWorker()
"""The score worker of this process."""


def forget_process_state_after_fork() -> None:
    """In a forked child, start afresh: no inherited worker, and no lock another thread held.

    It also marks the process as forked, so that its HTTP clients skip the
    macOS system proxy lookup that would kill it.
    """
    PROCESS_SCORE_WORKER.forget_after_fork()
    PROCESS_NOTICES.lock = threading.Lock()
    PROCESS_ORIGIN.forked = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=forget_process_state_after_fork)


class ScoreSettings(Protocol):
    """What `queue_step_score` reads from a monitor: its label and the tools it exports to."""

    @property
    def label(self) -> str:
        """The monitor's label, which names its score."""
        ...

    @property
    def export_scores(self) -> AbstractSet[Tracer]:
        """The tools the monitor sends its scores to."""
        ...


def queue_step_score(
    traced_step: TracedRun,
    *,
    record: StepRecord,
    monitor: ScoreSettings,
) -> None:
    """Queue the committed step's highest suspicion for each tool its run is traced to.

    Nothing is queued when the monitor exports to no tool, when the step span
    had no handler, or when the step judged no sample. It never raises: a
    failure is logged by the error's type alone, and the run goes on.
    """
    value = find_max_suspicion(record["samples"])
    if not monitor.export_scores or traced_step.run_id is None or value is None:
        return
    try:
        destinations = find_score_destinations(traced_step.handlers, tracers=monitor.export_scores)
        if destinations:
            put_step_scores(
                PROCESS_SCORE_WORKER.read_worker(),
                destinations=destinations,
                step_id=traced_step.run_id,
                name=build_score_name(monitor.label),
                value=value,
            )
    except Exception as error:
        logger.warning(
            "score export: the step's score could not be queued: %s", type(error).__name__
        )


def put_step_scores(
    worker: ScoreWorker,
    *,
    destinations: Sequence[ScoreDestination],
    step_id: UUID,
    name: str,
    value: float,
) -> None:
    """Put the step's score on the worker's queue once for each destination."""
    for destination in destinations:
        worker.put(
            PendingScore(
                step_id=step_id,
                name=name,
                value=value,
                tracer=destination.tracer,
                project=destination.project,
                queued_at=worker.clock(),
                connection=destination.connection,
            ),
        )
