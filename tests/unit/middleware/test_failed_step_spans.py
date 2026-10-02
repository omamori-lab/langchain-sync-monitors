"""A step that fails ends its open spans with the error it fails with.

When a call inside a step raises, every monitor span still open ends with
`on_chain_error`: the judgement whose monitor raised, a sibling judgement the
task group cancelled, and the step span, whose inputs still name the step
first proposed. The `MonitorStepFailedEvent` is written as before.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Monitor,
    MonitorInput,
    Resampling,
    Verdict,
)
from langchain_sync_monitors.errors import SynchronousRunError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
from langchain_sync_monitors.transcript import render_proposal_for_audit
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    build_task_input,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel, TriggeredFailureMonitor
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AwaitsEventLoop
from tests.support.tracing import RecordingTracer

EXFILTRATION_MARK = "attacker.example"


@dataclass(kw_only=True)
class SlowToJudgeReads(Monitor):
    """Delegates to `inner`, but under `ainvoke()` waits `delay` seconds before judging a read."""

    inner: Monitor
    delay: float

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        if "read_file" in render_proposal_for_audit(monitor_input.proposal):
            await asyncio.sleep(self.delay)
        return await self.inner.evaluate(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return self.inner.evaluate_sync(monitor_input)


def build_agent(
    model: BaseChatModel,
    *,
    monitor: Monitor,
    protocol: ControlProtocol | None = None,
) -> Runnable[Any, Any]:
    protocol = protocol or DeferToResample(
        fallback=HaltRun(),
        defer_threshold=0.6,
        audit_threshold=0.9,
    )
    return create_agent(
        model=model,
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )


def stream_traced_custom_events(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
    tracer: RecordingTracer,
) -> tuple[list[dict[str, Any]], BaseException]:
    """Stream the custom events of a run that fails, with the tracer attached, and its error."""
    payload, config = build_task_input(), RunnableConfig(callbacks=[tracer])
    events: list[dict[str, Any]] = []

    async def collect() -> None:
        async for event in agent.astream(payload, config, stream_mode="custom"):
            events.append(event)

    try:
        if mode == "invoke":
            events.extend(agent.stream(payload, config, stream_mode="custom"))
        else:
            asyncio.run(collect())
    except Exception as error:
        return events, error
    pytest.fail("the run was expected to fail")


def test_a_monitor_that_raises_ends_its_judgement_and_the_step_with_the_error(
    run_mode: RunMode,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    second = build_read_step(call_id="call-trigger").model_copy(update={"content": "TRIGGER"})
    monitor = TriggeredFailureMonitor(
        inner=KeywordMonitor(suspicion_by_keyword={EXFILTRATION_MARK: 0.95}),
        trigger="TRIGGER",
    )
    agent = build_agent(
        ScriptedChatModel(responses=[build_exfiltration_step(), second]),
        monitor=monitor,
    )

    # Act
    events, error = stream_traced_custom_events(agent, mode=run_mode, tracer=tracer)

    # Assert
    assert isinstance(error, TimeoutError)
    [step] = tracer.find_runs("monitor step")
    first, failed = step.find_children("monitor judgement")
    assert isinstance(step.error, TimeoutError)
    assert step.inputs["step_number"] == 1
    assert EXFILTRATION_MARK in step.inputs["proposed_step"]
    assert (first.error, first.outputs["suspicion"]) == (None, 0.95)
    assert isinstance(failed.error, TimeoutError)
    assert step.find_children("monitor decision") == []
    assert tracer.find_open_runs() == []
    assert tracer.find_unknown_parents() == []
    [event] = [event for event in events if event["type"] == "monitor_step_failed"]
    assert event["error"] == "TimeoutError: monitor provider timed out"
    assert [sample["suspicion"] for sample in event["samples"]] == [0.95]


def test_a_step_that_fails_before_any_verdict_ends_its_span_without_a_proposal(
    run_mode: RunMode,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    model = FlakyChatModel(replies=[RuntimeError("429 rate limited")])
    agent = build_agent(model, monitor=KeywordMonitor())

    # Act
    _, error = stream_traced_custom_events(agent, mode=run_mode, tracer=tracer)

    # Assert
    assert isinstance(error, RuntimeError)
    [step] = tracer.find_runs("monitor step")
    assert isinstance(step.error, RuntimeError)
    assert step.inputs == {"step_number": 1, "proposed_step": None}
    assert step.read_child_names() == ["FlakyChatModel"]
    assert tracer.find_open_runs() == []


def test_a_sibling_cancelled_by_the_task_group_ends_its_judgement_with_the_cancellation() -> None:
    # Arrange
    tracer = RecordingTracer()
    model = FlakyChatModel(
        replies=[build_exfiltration_step(), build_read_step(), RuntimeError("500 from provider")],
        delays=[0.0, 0.0, 0.05],
    )
    monitor = SlowToJudgeReads(
        inner=KeywordMonitor(suspicion_by_keyword={EXFILTRATION_MARK: 0.95}),
        delay=5.0,
    )
    protocol = DeferToResample(
        fallback=HaltRun(),
        defer_threshold=0.6,
        audit_threshold=0.9,
        resampling=Resampling.PARALLEL,
    )
    agent = build_agent(model, monitor=monitor, protocol=protocol)

    # Act
    _, error = stream_traced_custom_events(agent, mode="ainvoke", tracer=tracer)

    # Assert
    assert isinstance(error, RuntimeError)
    [step] = tracer.find_runs("monitor step")
    judgements = step.find_children("monitor judgement")
    assert isinstance(step.error, RuntimeError)
    assert [judgement.inputs["sample_number"] for judgement in judgements[:1]] == [1]
    assert judgements[0].error is None
    [cancelled] = judgements[1:]
    assert isinstance(cancelled.error, asyncio.CancelledError)
    assert tracer.find_open_runs() == []
    assert tracer.find_unknown_parents() == []


def test_a_protocol_that_awaits_under_invoke_ends_the_step_span_with_the_library_error() -> None:
    # Arrange
    tracer = RecordingTracer()
    agent = build_agent(
        ScriptedChatModel(responses=[build_read_step()]),
        monitor=KeywordMonitor(),
        protocol=AwaitsEventLoop(),
    )

    # Act
    _, error = stream_traced_custom_events(agent, mode="invoke", tracer=tracer)

    # Assert
    assert isinstance(error, SynchronousRunError)
    [step] = tracer.find_runs("monitor step")
    [judgement] = step.find_children("monitor judgement")
    assert isinstance(step.error, SynchronousRunError)
    assert judgement.error is None
    assert step.read_child_names() == ["ScriptedChatModel", "monitor judgement"]
    assert tracer.find_open_runs() == []
