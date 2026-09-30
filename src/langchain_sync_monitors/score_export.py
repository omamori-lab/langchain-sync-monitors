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
  reads: `LANGSMITH_API_KEY`, and `LANGFUSE_PUBLIC_KEY` with
  `LANGFUSE_SECRET_KEY`.
- **Only a traced run sends.** The handlers of the step span name the tools
  the run is traced to: LangSmith's `LangChainTracer`, whose project the
  feedback goes to, and Langfuse's LangChain handler, recognised by its
  package without importing it. A run traced to neither sends nothing, and
  says nothing.
- **A step with no sample writes no score**, such as a step that halts
  because an earlier step was halted.
- **The agent never waits.** The score goes on the queue of one worker
  thread per process, which `score_worker` describes: it writes to
  LangSmith within a window of the step, and to Langfuse once Langfuse has
  ingested the step, about 15 seconds after it in our checks. At exit it
  drains what waits, for up to 30 seconds. Nothing it does raises into a
  run; its failures are logged by `langchain_sync_monitors.score_worker`.
- **Only numbers and ids leave the process**: the judge's reason is never
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
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer

from langchain_sync_monitors._langchain import TracedRun
from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import build_langfuse_sender, read_langfuse_credentials
from langchain_sync_monitors.langsmith_scores import (
    build_langsmith_sender,
    read_langsmith_credentials,
)
from langchain_sync_monitors.model_calls import is_package_installed
from langchain_sync_monitors.options import check_enum_option, describe_option_value
from langchain_sync_monitors.score_worker import ScoreWorker
from langchain_sync_monitors.scores import PendingScore, ScoreSender, Tracer
from langchain_sync_monitors.spans import find_max_suspicion

logger = logging.getLogger(__name__)

SCORE_NAME_SUFFIX = "_suspicion"
"""What follows a monitor's label in the name of its score."""

LANGFUSE_PACKAGE = "langfuse"
"""The top-level package of Langfuse's SDK, whose LangChain handler traces a run to Langfuse."""


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
                "export_scores asks for Tracer.LANGSMITH, but LANGSMITH_API_KEY is not set: "
                "set the key LangSmith's tracer uses, or leave Tracer.LANGSMITH out"
            )
            raise ConfigurationError(message)
        return
    if not is_package_installed(LANGFUSE_PACKAGE):
        message = (
            "export_scores asks for Tracer.LANGFUSE, but the langfuse package, whose "
            "LangChain handler traces a run to Langfuse, is not installed: pip install langfuse"
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
    """Where one score goes: a tool, and for LangSmith the project the run is traced to."""

    tracer: Tracer
    project: str | None = None


def read_destination(
    handler: BaseCallbackHandler,
    *,
    tracers: AbstractSet[Tracer],
) -> ScoreDestination | None:
    """Return where the handler traces to, if it is the tracer of a tool among `tracers`."""
    if Tracer.LANGSMITH in tracers and isinstance(handler, LangChainTracer):
        return ScoreDestination(tracer=Tracer.LANGSMITH, project=handler.project_name)
    if Tracer.LANGFUSE in tracers and is_langfuse_handler(handler):
        return ScoreDestination(tracer=Tracer.LANGFUSE)
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
    """Return the tool's sender on the credentials in the environment, or None without them."""
    if tracer is Tracer.LANGSMITH:
        langsmith = read_langsmith_credentials()
        return None if langsmith is None else build_langsmith_sender(langsmith)
    langfuse = read_langfuse_credentials()
    return None if langfuse is None else build_langfuse_sender(langfuse)


def build_score_worker() -> ScoreWorker:
    """Return a worker whose senders use the credentials in the environment."""
    return ScoreWorker(build_sender=build_score_sender)


@dataclass(slots=True, kw_only=True)
class ProcessScoreWorker:
    """The one score worker of this process, started with the first score queued.

    Its thread drains at exit through `atexit`. A process forked after the
    worker started starts a worker of its own, since threads do not survive
    a fork.
    """

    build_worker: Callable[[], ScoreWorker] = build_score_worker
    worker: ScoreWorker | None = None
    process_id: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def read_worker(self) -> ScoreWorker:
        """Return the process's worker, starting it and its exit drain the first time."""
        with self.lock:
            if self.worker is None or self.process_id != os.getpid():
                if self.worker is not None:
                    atexit.unregister(self.worker.stop)
                worker = self.build_worker()
                worker.start()
                atexit.register(worker.stop)
                self.worker = worker
                self.process_id = os.getpid()
            return self.worker


PROCESS_SCORE_WORKER = ProcessScoreWorker()
"""The score worker of this process."""


def queue_step_score(
    traced_step: TracedRun,
    *,
    record: StepRecord,
    label: str,
    tracers: AbstractSet[Tracer],
) -> None:
    """Queue the committed step's highest suspicion for each tool its run is traced to.

    Nothing is queued when `tracers` is empty, when the step span had no
    handler, or when the step judged no sample. It never raises: a failure is
    logged, and the run goes on.
    """
    value = find_max_suspicion(record["samples"])
    if not tracers or traced_step.run_id is None or value is None:
        return
    try:
        destinations = find_score_destinations(traced_step.handlers, tracers=tracers)
        if destinations:
            put_step_scores(
                PROCESS_SCORE_WORKER.read_worker(),
                destinations=destinations,
                step_id=traced_step.run_id,
                name=build_score_name(label),
                value=value,
            )
    except Exception:
        logger.warning("score export: the step's score could not be queued", exc_info=True)


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
            ),
        )
