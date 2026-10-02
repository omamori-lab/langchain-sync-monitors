"""The `export_scores` option: checked when the monitor is built, and queued only for traced runs.

The option is off by default. A tool whose credentials, or for Langfuse whose
package, are missing is refused with `ConfigurationError` when the monitor is
built. A step's score is queued once per tool its span's handlers trace to,
never for a step with no sample, and queuing never raises into the run.
"""

from __future__ import annotations

import copy
import logging
import pickle
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import tracing_context
from pydantic import SecretStr

from langchain_sync_monitors import score_export
from langchain_sync_monitors._langchain import TracedRun
from langchain_sync_monitors.contracts import SampleRecord, StepRecord, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import LangfuseScoreSender
from langchain_sync_monitors.langsmith_scores import LangSmithFeedbackSender
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import TrustedMonitoring
from langchain_sync_monitors.score_export import (
    OTEL_MODE_NOTICE,
    PROCESS_SCORE_WORKER,
    ProcessScoreWorker,
    ScoreDestination,
    build_score_name,
    build_score_sender,
    find_score_destinations,
    is_langfuse_handler,
    queue_step_score,
)
from langchain_sync_monitors.score_worker import ScoreWorker
from langchain_sync_monitors.scores import LangSmithCredentials, Tracer
from tests.support.agents import build_keyword_monitor
from tests.support.score_services import (
    CONNECTION,
    PROJECT_NAME,
    LangfuseHandler,
    ScoreServices,
    build_langsmith_tracer,
    build_step_id,
    set_score_credentials,
)
from tests.support.tracing import RecordingTracer

CREDENTIAL_VARIABLES = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


BOTH_TOOLS = frozenset({Tracer.LANGSMITH, Tracer.LANGFUSE})
OWN_CONNECTION = LangSmithCredentials(
    api_key=SecretStr("own-tracer-key"),
    endpoint="https://own-smith.test/api/v1",
    workspace_id=None,
)


@dataclass(frozen=True, kw_only=True)
class MonitorSettings:
    """What `queue_step_score` reads from a monitor: its label and the tools it exports to."""

    label: str = "monitor"
    export_scores: frozenset[Tracer] = frozenset()


def build_own_client() -> MagicMock:
    """Return a LangSmith client double that names its own endpoint and key."""
    client = MagicMock()
    client.api_url = OWN_CONNECTION.endpoint
    client.api_key = OWN_CONNECTION.api_key.get_secret_value()
    client.workspace_id = None
    return client


def build_monitor(**options: Any) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=TrustedMonitoring(audit_threshold=0.8),
        **options,
    )


def clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def build_record(*suspicions: float) -> StepRecord:
    """Return a committed step's record with one judged sample per suspicion."""
    return StepRecord(
        agent="main",
        monitor="monitor",
        step_number=1,
        outcome="allowed",
        flagged=False,
        blocked_count=0,
        samples=[
            SampleRecord(suspicion=suspicion, reason="judged", proposal="step", executed=False)
            for suspicion in suspicions
        ],
    )


def build_traced_step(*handlers: BaseCallbackHandler) -> TracedRun:
    return TracedRun(run_id=build_step_id(), handlers=list(handlers))


def test_export_scores_is_off_by_default_and_needs_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    clear_credentials(monkeypatch)

    # Act
    monitor = build_monitor()

    # Assert
    assert monitor.export_scores == frozenset()


def test_the_tools_asked_for_are_kept_as_a_frozenset(score_services: ScoreServices) -> None:
    # Act
    monitor = build_monitor(export_scores={Tracer.LANGSMITH, Tracer.LANGFUSE})

    # Assert
    assert monitor.export_scores == frozenset({Tracer.LANGSMITH, Tracer.LANGFUSE})
    assert isinstance(monitor.export_scores, frozenset)


@pytest.mark.parametrize(
    "value",
    [Tracer.LANGSMITH, "langsmith", [Tracer.LANGSMITH], {"langsmith"}, None],
    ids=["a-member", "a-string", "a-list", "a-set-of-strings", "none"],
)
def test_anything_but_a_set_of_tracers_is_refused(
    score_services: ScoreServices,
    value: object,
) -> None:
    # Act, Assert
    with pytest.raises(ConfigurationError, match="export_scores"):
        build_monitor(export_scores=value)


def test_langsmith_without_its_key_is_refused_when_the_monitor_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    clear_credentials(monkeypatch)

    # Act, Assert
    with pytest.raises(ConfigurationError, match="LANGSMITH_API_KEY is not set"):
        build_monitor(export_scores={Tracer.LANGSMITH})


def test_langfuse_without_its_package_is_refused_when_the_monitor_is_built(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(score_export, "is_package_installed", lambda name: False)

    # Act, Assert
    with pytest.raises(ConfigurationError, match=r"uv add langfuse \(or pip install langfuse\)"):
        build_monitor(export_scores={Tracer.LANGFUSE})


@pytest.mark.parametrize("missing", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_langfuse_without_its_keys_is_refused_when_the_monitor_is_built(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    # Arrange
    monkeypatch.delenv(missing)

    # Act, Assert
    with pytest.raises(ConfigurationError, match="LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY"):
        build_monitor(export_scores={Tracer.LANGFUSE})


def test_langsmith_alone_needs_neither_langfuse_nor_its_keys(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(score_export, "is_package_installed", lambda name: False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY")

    # Act
    monitor = build_monitor(export_scores={Tracer.LANGSMITH})

    # Assert
    assert monitor.export_scores == {Tracer.LANGSMITH}


def test_a_subagent_monitor_and_a_copied_one_keep_the_option_and_it_pickles(
    score_services: ScoreServices,
) -> None:
    # Arrange
    monitor = build_monitor(export_scores={Tracer.LANGFUSE})

    # Act
    subagent_monitor = monitor.copy_for_subagent(subagent_name="researcher")
    # A deep copy rebuilds the monitor through the same reduce protocol pickle uses.
    copied = copy.deepcopy(monitor)
    pickled = pickle.dumps(monitor)

    # Assert
    assert subagent_monitor.export_scores == {Tracer.LANGFUSE}
    assert subagent_monitor.task_author is TaskAuthor.PARENT_AGENT
    assert copied is not monitor
    assert copied.export_scores == {Tracer.LANGFUSE}
    assert pickled.startswith(pickle.PROTO)


def test_the_score_is_named_after_the_monitor_label() -> None:
    # Act
    names = [build_score_name("monitor"), build_score_name("outer")]

    # Assert
    assert names == ["monitor_suspicion", "outer_suspicion"]


def test_langfuse_handler_is_recognised_by_its_package_and_so_is_a_subclass() -> None:
    # Arrange
    class OwnLangfuseHandler(LangfuseHandler):
        """A user's handler built on Langfuse's."""

    # Act
    recognised = [
        is_langfuse_handler(LangfuseHandler()),
        is_langfuse_handler(OwnLangfuseHandler()),
        is_langfuse_handler(RecordingTracer()),
    ]

    # Assert
    assert recognised == [True, True, False]


def test_the_destinations_are_the_tools_asked_for_that_the_run_is_traced_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    set_score_credentials(monkeypatch)
    client = MagicMock()
    handlers = [
        RecordingTracer(),
        build_langsmith_tracer(client),
        build_langsmith_tracer(client),
        build_langsmith_tracer(client, project_name="second-project"),
        build_langsmith_tracer(build_own_client()),
        LangfuseHandler(),
        LangfuseHandler(),
    ]

    # Act
    both = find_score_destinations(handlers, tracers={Tracer.LANGSMITH, Tracer.LANGFUSE})
    langfuse_only = find_score_destinations(handlers, tracers={Tracer.LANGFUSE})
    untraced = find_score_destinations([RecordingTracer()], tracers={Tracer.LANGSMITH})

    # Assert
    assert both == [
        ScoreDestination(tracer=Tracer.LANGSMITH, project=PROJECT_NAME, connection=CONNECTION),
        ScoreDestination(tracer=Tracer.LANGSMITH, project="second-project", connection=CONNECTION),
        ScoreDestination(tracer=Tracer.LANGSMITH, project=PROJECT_NAME, connection=OWN_CONNECTION),
        ScoreDestination(tracer=Tracer.LANGFUSE),
    ]
    assert langfuse_only == [ScoreDestination(tracer=Tracer.LANGFUSE)]
    assert untraced == []


def test_a_traced_step_queues_its_highest_suspicion_once_per_tool(
    score_services: ScoreServices,
) -> None:
    # Arrange
    traced_step = build_traced_step(build_langsmith_tracer(MagicMock()), LangfuseHandler())

    # Act
    queue_step_score(
        traced_step,
        record=build_record(0.2, 0.7, 0.4),
        monitor=MonitorSettings(label="outer", export_scores=BOTH_TOOLS),
    )
    score_services.worker.waiting.take_incoming()

    # Assert
    queued = score_services.worker.waiting.by_tracer
    assert sorted(queued) == [Tracer.LANGFUSE, Tracer.LANGSMITH]
    for tracer, [score] in queued.items():
        assert score.step_id == traced_step.run_id
        assert score.name == "outer_suspicion"
        assert score.value == 0.7
        assert score.tracer is tracer
        assert score.queued_at == score_services.clock.now
    assert queued[Tracer.LANGSMITH][0].project == PROJECT_NAME
    assert queued[Tracer.LANGSMITH][0].connection == CONNECTION
    assert queued[Tracer.LANGFUSE][0].project is None
    assert queued[Tracer.LANGFUSE][0].connection is None


def test_a_step_traced_through_its_own_client_queues_that_client_connection(
    score_services: ScoreServices,
) -> None:
    # Arrange: the environment names another endpoint and key
    traced_step = build_traced_step(build_langsmith_tracer(build_own_client()))

    # Act
    queue_step_score(
        traced_step,
        record=build_record(0.6),
        monitor=MonitorSettings(export_scores=frozenset({Tracer.LANGSMITH})),
    )
    score_services.worker.waiting.take_incoming()

    # Assert
    [score] = score_services.worker.waiting.by_tracer[Tracer.LANGSMITH]
    assert score.connection == OWN_CONNECTION


def test_a_langsmith_tracer_with_no_key_anywhere_sends_nothing_and_says_so(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    clear_credentials(monkeypatch)
    traced_step = build_traced_step(build_langsmith_tracer(MagicMock()))

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_export"):
        queue_step_score(
            traced_step,
            record=build_record(0.6),
            monitor=MonitorSettings(export_scores=frozenset({Tracer.LANGSMITH})),
        )

    # Assert
    assert score_services.worker.waiting.count() == 0
    assert [record.getMessage() for record in caplog.records] == [
        "score export: the LangSmith tracer has no API key, so no score is sent"
    ]


@pytest.mark.parametrize(
    ("traced_step", "record", "tracers"),
    [
        (TracedRun(), build_record(0.9), {Tracer.LANGSMITH}),
        (
            build_traced_step(build_langsmith_tracer(MagicMock())),
            build_record(),
            {Tracer.LANGSMITH},
        ),
        (build_traced_step(build_langsmith_tracer(MagicMock())), build_record(0.9), set()),
        (build_traced_step(RecordingTracer()), build_record(0.9), {Tracer.LANGSMITH}),
    ],
    ids=["not-traced", "no-sample", "option-off", "traced-elsewhere"],
)
def test_nothing_is_queued_and_no_worker_starts_when_there_is_nothing_to_send(
    monkeypatch: pytest.MonkeyPatch,
    traced_step: TracedRun,
    record: StepRecord,
    tracers: set[Tracer],
) -> None:
    # Arrange
    started: list[str] = []

    def refuse_to_start() -> ScoreWorker:
        started.append("worker")
        message = "no worker should start"
        raise AssertionError(message)

    monkeypatch.setattr(PROCESS_SCORE_WORKER, "worker", None)
    monkeypatch.setattr(PROCESS_SCORE_WORKER, "build_worker", refuse_to_start)

    # Act
    queue_step_score(
        traced_step, record=record, monitor=MonitorSettings(export_scores=frozenset(tracers))
    )

    # Assert
    assert started == []


def test_a_failure_to_queue_is_logged_and_never_raised(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    def break_the_queue(score: object) -> None:
        message = "the queue broke"
        raise RuntimeError(message)

    monkeypatch.setattr(score_services.worker, "put", break_the_queue)
    traced_step = build_traced_step(LangfuseHandler())

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_export"):
        queue_step_score(
            traced_step,
            record=build_record(0.9),
            monitor=MonitorSettings(export_scores=frozenset({Tracer.LANGFUSE})),
        )

    # Assert
    assert "the step's score could not be queued" in caplog.text


def test_the_process_worker_starts_once_with_its_exit_drain_and_again_after_a_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    built: list[ScoreWorker] = []
    registered: list[object] = []
    unregistered: list[object] = []

    def build_worker() -> ScoreWorker:
        worker = ScoreWorker(build_sender=lambda tracer: None)
        monkeypatch.setattr(worker, "start", lambda: None)
        built.append(worker)
        return worker

    monkeypatch.setattr(score_export.atexit, "register", registered.append)
    monkeypatch.setattr(score_export.atexit, "unregister", unregistered.append)
    process = ProcessScoreWorker(build_worker=build_worker)

    # Act
    first = process.read_worker()
    again = process.read_worker()
    monkeypatch.setattr(score_export.os, "getpid", lambda: -1)
    after_fork = process.read_worker()

    # Assert
    assert first is again
    assert after_fork is not first
    assert built == [first, after_fork]
    assert registered == [first.stop, after_fork.stop]
    assert unregistered == [first.stop]


def test_the_langfuse_sender_needs_the_environment_and_the_langsmith_one_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    set_score_credentials(monkeypatch)

    # Act
    langfuse = build_score_sender(Tracer.LANGFUSE)
    clear_credentials(monkeypatch)
    without_keys = [build_score_sender(Tracer.LANGSMITH), build_score_sender(Tracer.LANGFUSE)]

    # Assert: each LangSmith score carries its own tracer's connection
    assert isinstance(langfuse, LangfuseScoreSender)
    assert isinstance(without_keys[0], LangSmithFeedbackSender)
    assert without_keys[1] is None
    langfuse.close()
    without_keys[0].close()


def queue_one_score(traced_step: TracedRun, *, tracers: frozenset[Tracer]) -> None:
    """Queue the score of a step judged 0.6, from a monitor that exports to `tracers`."""
    queue_step_score(
        traced_step, record=build_record(0.6), monitor=MonitorSettings(export_scores=tracers)
    )


def test_a_langsmith_client_in_otel_mode_gets_no_score_and_one_warning(
    score_services: ScoreServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: OTel mode derives run ids from span ids, so no feedback could find its step
    client = build_own_client()
    client.tracing_mode = "otel"
    traced_step = build_traced_step(build_langsmith_tracer(client))

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_export"):
        queue_one_score(traced_step, tracers=frozenset({Tracer.LANGSMITH}))
        queue_one_score(traced_step, tracers=frozenset({Tracer.LANGSMITH}))

    # Assert
    assert score_services.worker.waiting.count() == 0
    assert [record.getMessage() for record in caplog.records] == [OTEL_MODE_NOTICE]


@pytest.mark.parametrize("mode", ["langsmith", "hybrid"])
def test_a_langsmith_client_that_sends_run_ids_still_gets_its_score(
    score_services: ScoreServices,
    mode: str,
) -> None:
    # Arrange: hybrid mode also sends the runs to LangSmith with their own ids
    client = build_own_client()
    client.tracing_mode = mode
    traced_step = build_traced_step(build_langsmith_tracer(client))

    # Act
    queue_one_score(traced_step, tracers=frozenset({Tracer.LANGSMITH}))

    # Assert
    assert score_services.worker.waiting.count() == 1


def test_a_run_with_langsmith_tracing_turned_off_gets_no_langsmith_score(
    score_services: ScoreServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    traced_step = build_traced_step(build_langsmith_tracer(MagicMock()), LangfuseHandler())

    # Act
    with (
        caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_export"),
        tracing_context(enabled=False),
    ):
        queue_one_score(traced_step, tracers=BOTH_TOOLS)
    score_services.worker.waiting.take_incoming()

    # Assert: quietly, and Langfuse still gets its score
    assert list(score_services.worker.waiting.by_tracer) == [Tracer.LANGFUSE]
    assert caplog.records == []


@pytest.mark.parametrize(("value", "queued"), [("false", 0), ("FALSE", 0), ("true", 1), ("0", 1)])
def test_langfuse_tracing_turned_off_by_its_variable_gets_no_langfuse_score(
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    queued: int,
) -> None:
    # Arrange: the SDK turns tracing off only when the variable, lowercased, is false
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", value)
    traced_step = build_traced_step(LangfuseHandler())

    # Act
    queue_one_score(traced_step, tracers=frozenset({Tracer.LANGFUSE}))

    # Assert
    assert score_services.worker.waiting.count() == queued


@dataclass(frozen=True)
class StubSampler:
    """Stands in for an OpenTelemetry sampler: `TraceIdRatioBased` has a `rate`."""

    rate: object


@dataclass(frozen=True)
class StubLangfuseClient:
    """Stands in for a Langfuse client, with the private attributes its handler holds."""

    _tracing_enabled: object = True
    _resources: object = None


@dataclass(frozen=True)
class StubResources:
    """Stands in for Langfuse's resource manager and its tracer provider."""

    tracer_provider: object = None


def build_langfuse_handler(client: object) -> LangfuseHandler:
    """Return a Langfuse handler whose client is `client`, as `get_client` would set it."""
    handler = LangfuseHandler()
    handler._langfuse_client = client  # ty: ignore[unresolved-attribute]
    return handler


def build_sampled_client(rate: object) -> StubLangfuseClient:
    """Return a Langfuse client whose tracer provider samples at `rate`."""
    provider = MagicMock(sampler=StubSampler(rate=rate))
    return StubLangfuseClient(_resources=StubResources(tracer_provider=provider))


@pytest.mark.parametrize(
    "client",
    [
        StubLangfuseClient(_tracing_enabled=False),
        build_sampled_client(0.0),
        build_sampled_client(0),
    ],
    ids=["tracing_enabled=False", "sample_rate=0.0", "sample_rate=0"],
)
def test_a_langfuse_client_that_sends_no_trace_gets_no_langfuse_score(
    score_services: ScoreServices,
    client: StubLangfuseClient,
) -> None:
    # Arrange
    traced_step = build_traced_step(build_langfuse_handler(client))

    # Act
    queue_one_score(traced_step, tracers=frozenset({Tracer.LANGFUSE}))

    # Assert
    assert score_services.worker.waiting.count() == 0


@pytest.mark.parametrize(
    "client",
    [
        StubLangfuseClient(),
        build_sampled_client(0.5),
        build_sampled_client(True),
        build_sampled_client("0"),
        StubLangfuseClient(_resources=StubResources(tracer_provider=object())),
        None,
    ],
    ids=["tracing", "sample_rate=0.5", "rate=True", "rate='0'", "no sampler", "no client"],
)
def test_a_langfuse_client_that_traces_or_cannot_be_read_still_gets_its_score(
    score_services: ScoreServices,
    client: StubLangfuseClient | None,
) -> None:
    # Arrange
    traced_step = build_traced_step(build_langfuse_handler(client))

    # Act
    queue_one_score(traced_step, tracers=frozenset({Tracer.LANGFUSE}))

    # Assert
    assert score_services.worker.waiting.count() == 1


def test_a_forked_child_forgets_a_lock_another_thread_held_and_the_parent_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: another thread holds the lock, as at the moment of a fork
    built: list[ScoreWorker] = []
    unregistered: list[object] = []

    def build_worker() -> ScoreWorker:
        worker = ScoreWorker(build_sender=lambda tracer: None)
        monkeypatch.setattr(worker, "start", lambda: None)
        built.append(worker)
        return worker

    monkeypatch.setattr(score_export.atexit, "register", lambda hook: None)
    monkeypatch.setattr(score_export.atexit, "unregister", unregistered.append)
    process = ProcessScoreWorker(build_worker=build_worker)
    parent = process.read_worker()
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with process.lock:
            held.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    held.wait(timeout=5.0)

    # Act
    process.forget_after_fork()
    child = process.read_worker()
    release.set()
    holder.join(timeout=5.0)

    # Assert: the child started its own worker without waiting, and dropped the parent's drain
    assert child is not parent
    assert built == [parent, child]
    assert unregistered == [parent.stop]


def test_queueing_a_score_never_takes_the_lock_once_the_worker_runs(
    score_services: ScoreServices,
) -> None:
    # Arrange: the lock is held, as if a fork had copied it held
    traced_step = build_traced_step(LangfuseHandler())

    # Act: on a thread, so that a queue that waited for the lock fails the test, not hangs it
    queuer = threading.Thread(
        target=queue_one_score,
        args=(traced_step,),
        kwargs={"tracers": frozenset({Tracer.LANGFUSE})},
        daemon=True,
    )
    with PROCESS_SCORE_WORKER.lock:
        queuer.start()
        queuer.join(timeout=5.0)
        still_waiting = queuer.is_alive()

    # Assert
    assert not still_waiting
    assert score_services.worker.waiting.count() == 1
