"""A step that fails ends its open spans with the error it fails with.

When a call inside a step raises, every monitor span still open ends with
`on_chain_error`: the judgement whose monitor raised, a sibling judgement of
the same resample that the task group cancelled, and the step span, whose
inputs still name the step first proposed. The `MonitorStepFailedEvent` is
written as before.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import hyoshigi
import pytest
from hyoshigi import Action, SuspicionScore, Trajectory
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_sync_monitors import DeferToResample, MonitorTwins, Resampling
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    build_task_input,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel, TriggeredFailureMonitor
from tests.support.monitors import KeywordMonitor, read_action_text
from tests.support.tracing import RecordingTracer

EXFILTRATION_MARK = "attacker.example"


@dataclass(frozen=True, kw_only=True)
class SlowToRateReadsRater(hyoshigi.AsyncMonitor):
    """Rates with `inner`, after waiting `delay` seconds when the action reads a file."""

    inner: hyoshigi.AsyncMonitor
    delay: float

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        if "read_file" in read_action_text(action):
            await asyncio.sleep(self.delay)
        return await self.inner.rate(action, trajectory=trajectory)


@dataclass(frozen=True, kw_only=True)
class SlowToJudgeReads(MonitorTwins):
    """The twins of `inner`, whose async twin waits `delay` seconds before rating a read."""

    inner: MonitorTwins
    delay: float
    monitor: hyoshigi.Monitor = field(init=False)
    async_monitor: hyoshigi.AsyncMonitor = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "monitor", self.inner.monitor)
        object.__setattr__(
            self,
            "async_monitor",
            SlowToRateReadsRater(inner=self.inner.async_monitor, delay=self.delay),
        )
        super().__post_init__()


def build_agent(
    model: BaseChatModel,
    *,
    monitor: MonitorTwins,
    protocol: DeferToResample | None = None,
) -> Runnable[Any, Any]:
    protocol = protocol or DeferToResample(
        defer_threshold=0.6,
        audit_threshold=0.9,
        max_resamples=2,
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
    [call] = step.inputs["proposed_step"]["tool_calls"]
    assert EXFILTRATION_MARK in call["arguments"]
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
    # Arrange: two resamples rated at once; the read's rating is slow, the other's raises
    tracer = RecordingTracer()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("TRIGGER")],
    )
    monitor = SlowToJudgeReads(
        inner=TriggeredFailureMonitor(
            inner=KeywordMonitor(suspicion_by_keyword={EXFILTRATION_MARK: 0.95}),
            trigger="TRIGGER",
        ),
        delay=5.0,
    )
    protocol = DeferToResample(
        defer_threshold=0.6,
        audit_threshold=0.9,
        max_resamples=2,
        resampling=Resampling.PARALLEL,
    )
    agent = build_agent(model, monitor=monitor, protocol=protocol)

    # Act
    _, error = stream_traced_custom_events(agent, mode="ainvoke", tracer=tracer)

    # Assert
    assert isinstance(error, TimeoutError)
    [step] = tracer.find_runs("monitor step")
    judgements = step.find_children("monitor judgement")
    assert isinstance(step.error, TimeoutError)
    assert [
        (judgement.inputs["sample_number"], type(judgement.error)) for judgement in judgements
    ] == [(1, type(None)), (2, asyncio.CancelledError), (3, TimeoutError)]
    assert tracer.find_open_runs() == []
    assert tracer.find_unknown_parents() == []
