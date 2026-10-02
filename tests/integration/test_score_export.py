"""A monitored agent's suspicion scores reach LangSmith and Langfuse, under `invoke` and `ainvoke`.

The run is traced to LangSmith through its tracer on a mock client, and to
Langfuse through a handler that stands in for Langfuse's, as the step span's
handlers show them to the monitor. The score worker is the process's, on
fake services and a clock the test moves, so each window is one call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import stamina
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors import middleware
from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun, TrustedMonitoring
from langchain_sync_monitors.score_requests import RETRY_ATTEMPTS
from langchain_sync_monitors.score_worker import UNFOUND_STEP_HINTS
from langchain_sync_monitors.scores import PendingScore, Tracer
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.log_records import find_logged_leaks
from tests.support.monitors import BENIGN_SUSPICION
from tests.support.score_services import (
    LANGSMITH_KEY,
    PROJECT_ID,
    LangfuseHandler,
    ScoreServices,
    build_langsmith_tracer,
    read_request_json,
)
from tests.support.tracing import RecordingTracer
from tests.unit.middleware.test_standing_halts import ReturningToModelMiddleware

BOTH_TOOLS = frozenset({Tracer.LANGSMITH, Tracer.LANGFUSE})
EXFILTRATION_SUSPICION = 0.9


def build_monitor(
    *, label: str = "monitor", tracers: frozenset[Tracer] = BOTH_TOOLS
) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=TrustedMonitoring(audit_threshold=0.8),
        label=label,
        export_scores=tracers,
    )


def build_agent(*middleware: AgentMiddleware[Any, Any, Any]) -> Runnable[Any, Any]:
    return create_agent(
        model=ScriptedChatModel(responses=[build_exfiltration_step(), AIMessage("Done.")]),
        tools=Workspace().build_tools(),
        middleware=list(middleware),
    )


def run_traced(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
    handlers: list[BaseCallbackHandler],
) -> dict[str, Any]:
    return run_agent(agent, mode=mode, config=RunnableConfig(callbacks=handlers))


def read_step_ids(tracer: RecordingTracer) -> list[str]:
    return [str(run.run_id) for run in tracer.find_runs("monitor step")]


def read_feedback(services: ScoreServices) -> dict[str, tuple[str, float]]:
    """Return each written feedback's run id, key and score."""
    return {
        item["run_id"]: (item["key"], item["score"])
        for item in services.langsmith.feedback.values()
    }


def read_scores(services: ScoreServices) -> dict[str, tuple[str, float]]:
    """Return each written score's step id, by its observation, and its name and value."""
    step_by_observation = {
        item["id"]: item["metadata"]["monitor_step_id"] for item in services.langfuse.observations
    }
    return {
        step_by_observation[item["observationId"]]: (item["name"], item["value"])
        for item in services.langfuse.scores.values()
    }


def test_each_step_score_reaches_both_tools_on_the_step_span(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]

    # Act
    result = run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)
    score_services.langfuse.ingest_steps(langfuse)
    score_services.send_window()

    # Assert
    first, second = read_step_ids(langfuse)
    expected = {
        first: ("monitor_suspicion", EXFILTRATION_SUSPICION),
        second: ("monitor_suspicion", BENIGN_SUSPICION),
    }
    assert read_feedback(score_services) == expected
    assert read_scores(score_services) == expected
    assert {item["session_id"] for item in score_services.langsmith.feedback.values()} == {
        PROJECT_ID
    }
    assert [record["samples"][0]["suspicion"] for record in result["monitor_log"]] == [
        EXFILTRATION_SUSPICION,
        BENIGN_SUSPICION,
    ]
    assert score_services.worker.waiting.count() == 0


def test_a_step_langfuse_has_not_ingested_waits_for_a_later_window(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]
    run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)

    # Act
    score_services.send_window()
    scores_before_ingestion = dict(score_services.langfuse.scores)
    score_services.langfuse.ingest_steps(langfuse)
    score_services.send_window()

    # Assert: LangSmith needed no lookup, and Langfuse was asked once a window
    assert len(read_feedback(score_services)) == 2
    assert scores_before_ingestion == {}
    assert len(read_scores(score_services)) == 2
    lookups = score_services.langfuse.find_requests("GET", "/api/public/v2/observations")
    assert len(lookups) == 2


def test_a_lookup_sent_again_after_a_server_error_logs_no_key_or_request(
    run_mode: RunMode,
    score_services: ScoreServices,
    retry_details: list[RetryDetails],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: each tool's first request, its lookup, meets a server error
    caplog.set_level(logging.DEBUG)
    score_services.langsmith.queued_answers.append(httpx.Response(503))
    score_services.langfuse.queued_answers.append(httpx.Response(503))
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]
    run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)
    score_services.langfuse.ingest_steps(langfuse)

    # Act
    with stamina.set_testing(True, attempts=RETRY_ATTEMPTS):
        score_services.send_window()

    # Assert: each lookup was sent again and every score written, no hook was handed the
    # request, and no record, on any logger, holds a key, a header or a live request
    assert len(read_feedback(score_services)) == len(read_scores(score_services)) == 2
    assert [(details.args, details.kwargs) for details in retry_details] == [((), {}), ((), {})]
    headers = [
        score_services.langsmith.requests[0].headers["x-api-key"],
        score_services.langfuse.requests[0].headers["authorization"],
    ]
    assert find_logged_leaks(caplog.records, secrets=[LANGSMITH_KEY, *headers]) == []


def test_only_the_tools_asked_for_receive_scores(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]
    monitor = build_monitor(tracers=frozenset({Tracer.LANGFUSE}))

    # Act
    run_traced(build_agent(monitor), mode=run_mode, handlers=handlers)
    score_services.langfuse.ingest_steps(langfuse)
    score_services.send_window()

    # Assert
    assert score_services.langsmith.requests == []
    assert len(read_scores(score_services)) == 2


@pytest.mark.parametrize("traced", [True, False], ids=["traced-elsewhere", "not-traced"])
def test_a_run_traced_to_neither_tool_queues_nothing(
    run_mode: RunMode,
    score_services: ScoreServices,
    traced: bool,
) -> None:
    # Arrange
    handlers: list[BaseCallbackHandler] = [RecordingTracer()] if traced else []

    # Act
    result = run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)
    score_services.send_window()

    # Assert
    assert len(result["monitor_log"]) == 2
    assert score_services.worker.waiting.count() == 0
    assert score_services.langsmith.requests == []
    assert score_services.langfuse.requests == []


def test_several_monitors_write_their_scores_under_their_own_labels(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]
    agent = build_agent(build_monitor(label="outer"), build_monitor(label="inner"))

    # Act
    run_traced(agent, mode=run_mode, handlers=handlers)
    score_services.langfuse.ingest_steps(langfuse)
    score_services.send_window()

    # Assert: each monitor's step span gets its own score, sent once, as the record's float
    names = sorted(name for name, _ in read_feedback(score_services).values())
    assert names == ["inner_suspicion", "inner_suspicion", "outer_suspicion", "outer_suspicion"]
    assert read_scores(score_services) == read_feedback(score_services)
    assert len(score_services.langsmith.find_requests("POST", "/feedback")) == 4
    [ingestion] = score_services.langfuse.find_requests("POST", "/api/public/ingestion")
    assert len(read_request_json(ingestion)["batch"]) == 4
    assert all(isinstance(value, float) for _, value in read_feedback(score_services).values())


@dataclass(kw_only=True)
class CertainMonitor(Monitor):
    """Judges every step certainly an attack, with the integer 1 as its suspicion."""

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return Verdict(suspicion=1, reason="certain")

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return Verdict(suspicion=1, reason="certain")


def test_the_score_is_the_record_float_even_for_an_integer_verdict(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange
    monitor = MonitorMiddleware(
        monitor=CertainMonitor(),
        protocol=TrustedMonitoring(audit_threshold=0.8),
        export_scores=frozenset({Tracer.LANGSMITH}),
    )
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock())]

    # Act
    result = run_traced(build_agent(monitor), mode=run_mode, handlers=handlers)
    score_services.send_window()

    # Assert: the feedback carries 1.0, a float, as the record does
    recorded = [record["samples"][0]["suspicion"] for record in result["monitor_log"]]
    sent = [
        read_request_json(post)["score"]
        for post in score_services.langsmith.find_requests("POST", "/feedback")
    ]
    assert recorded == sent == [1.0, 1.0]
    assert all(type(value) is float for value in [*recorded, *sent])


def test_a_step_with_no_sample_writes_no_score(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange: the first step is halted, and the steps a returning hook asks for halt unsampled
    langfuse = LangfuseHandler()
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock()), langfuse]
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=DeferToResample(
            fallback=HaltRun(), defer_threshold=0.5, audit_threshold=None, max_resamples=0
        ),
        export_scores=BOTH_TOOLS,
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [ReturningToModelMiddleware(returns=2), monitor]
    agent = create_agent(
        model=ScriptedChatModel(responses=[build_exfiltration_step(), build_read_step()]),
        tools=Workspace().build_tools(),
        middleware=stack,
    )

    # Act
    result = run_traced(agent, mode=run_mode, handlers=handlers)
    score_services.langfuse.ingest_steps(langfuse)
    score_services.send_window()

    # Assert
    log = result["monitor_log"]
    assert [len(record["samples"]) for record in log] == [1, 0, 0]
    first_step = read_step_ids(langfuse)[0]
    assert read_feedback(score_services) == {
        first_step: ("monitor_suspicion", EXFILTRATION_SUSPICION)
    }
    assert read_scores(score_services) == read_feedback(score_services)


def test_langsmith_feedback_goes_through_the_tracer_client_not_the_environment(
    run_mode: RunMode,
    score_services: ScoreServices,
) -> None:
    # Arrange: the tracer's client names its own endpoint and key; the environment another
    client = MagicMock()
    client.otel_exporter = None
    client.api_url = "https://own-smith.test/api/v1"
    client.api_key = "own-tracer-key"
    client.workspace_id = "own-workspace"
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(client)]

    # Act
    run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)
    score_services.send_window()

    # Assert: every request went to the tracer's endpoint, with its key and workspace
    requests = score_services.langsmith.requests
    assert {request.url.host for request in requests} == {"own-smith.test"}
    assert {request.headers["x-api-key"] for request in requests} == {"own-tracer-key"}
    assert {request.headers["X-Tenant-Id"] for request in requests} == {"own-workspace"}
    assert len(read_feedback(score_services)) == 2


def test_a_langfuse_run_whose_steps_the_environment_project_never_holds_gets_no_score(
    run_mode: RunMode,
    score_services: ScoreServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the handler traces elsewhere, as one built with other keys would
    handlers: list[BaseCallbackHandler] = [LangfuseHandler()]
    run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)

    # Act: the environment's project is asked for the steps until the scores are given up
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        for _ in range(31):
            score_services.send_window()

    # Assert: looked up, never written, and the likely cause said once
    assert score_services.langfuse.find_requests("GET", "/api/public/v2/observations") != []
    assert score_services.langfuse.find_requests("POST", "/api/public/ingestion") == []
    assert score_services.worker.waiting.count() == 0
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count(UNFOUND_STEP_HINTS[Tracer.LANGFUSE]) == 1
    assert any("gave up on 2 langfuse score(s)" in message for message in messages)


def test_a_worker_that_fails_never_reaches_the_run(
    run_mode: RunMode,
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    error_text = "text from the transcript"

    def break_the_queue(score: object) -> None:
        raise RuntimeError(error_text)

    monkeypatch.setattr(score_services.worker, "put", break_the_queue)
    handlers: list[BaseCallbackHandler] = [LangfuseHandler()]

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_export"):
        result = run_traced(build_agent(build_monitor()), mode=run_mode, handlers=handlers)

    # Assert
    assert result["messages"][-1].text == "Done."
    assert caplog.text.count("the step's score could not be queued: RuntimeError") == 2
    assert error_text not in caplog.text


def build_one_step_agent(monitor: MonitorMiddleware) -> Runnable[Any, Any]:
    return create_agent(
        model=ScriptedChatModel(responses=[AIMessage("Done.")]),
        tools=Workspace().build_tools(),
        middleware=[monitor],
    )


def test_a_step_whose_commit_fails_queues_no_score(
    run_mode: RunMode,
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the commit raises, as a state the update cannot read would make it
    def fail_to_commit(*args: object, **kwargs: object) -> object:
        message = "the commit failed"
        raise MonitorError(message)

    monkeypatch.setattr(middleware, "commit_step", fail_to_commit)
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock())]
    monitor = build_monitor(tracers=frozenset({Tracer.LANGSMITH}))

    # Act
    with pytest.raises(MonitorError, match="the commit failed"):
        run_traced(build_one_step_agent(monitor), mode=run_mode, handlers=handlers)
    waiting = score_services.worker.waiting.count()
    score_services.send_window()

    # Assert: a step that is not in monitor_log gets no score in the tracing tool
    assert waiting == 0
    assert score_services.langsmith.requests == []


def test_a_committed_step_queues_one_score_after_its_commit(
    run_mode: RunMode,
    score_services: ScoreServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: record when the commit returns and when the score is queued
    events: list[str] = []
    commit = middleware.commit_step
    put = score_services.worker.put

    def commit_and_note(*args: Any, **kwargs: Any) -> Any:
        response = commit(*args, **kwargs)
        events.append("committed")
        return response

    def put_and_note(score: PendingScore) -> None:
        events.append("queued")
        put(score)

    monkeypatch.setattr(middleware, "commit_step", commit_and_note)
    monkeypatch.setattr(score_services.worker, "put", put_and_note)
    handlers: list[BaseCallbackHandler] = [build_langsmith_tracer(MagicMock())]
    monitor = build_monitor(tracers=frozenset({Tracer.LANGSMITH}))

    # Act
    result = run_traced(build_one_step_agent(monitor), mode=run_mode, handlers=handlers)
    waiting = score_services.worker.waiting.count()
    score_services.send_window()

    # Assert: the one step is committed, then its one score is queued and written
    assert len(result["monitor_log"]) == 1
    assert events == ["committed", "queued"]
    assert waiting == 1
    assert list(read_feedback(score_services).values()) == [("monitor_suspicion", BENIGN_SUSPICION)]
