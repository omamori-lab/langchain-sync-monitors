"""A pending step refuses the model and the monitor once its step is over, in both modes.

A protocol that starts a task on the event loop and returns its decision
without awaiting it leaves the task running after the step was committed.
Nothing can use a sample drawn then, so the step refuses it, under
`invoke()` as from a notebook's running loop, and under `ainvoke()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Outcome,
    PendingStep,
    StepDecision,
)
from langchain_sync_monitors.errors import MonitorError, SynchronousRunError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor

type LateCall = Callable[[PendingStep], Awaitable[object]]

CLOSED_STEP_REFUSALS: dict[RunMode, tuple[type[MonitorError], str]] = {
    "invoke": (SynchronousRunError, "after its synchronous invoke() step was over"),
    "ainvoke": (MonitorError, "after its step was over, from a task a control protocol started"),
}
"""The error a closed step raises in each run mode, and the reason its message gives."""


async def draw_another_sample(step: PendingStep) -> object:
    return await step.sample(count=1)


async def draw_no_sample(step: PendingStep) -> object:
    return await step.sample(count=0)


async def request_a_trusted_step(step: PendingStep) -> object:
    return await step.request_trusted_step(trusted_model=ScriptedChatModel(responses=[]))


@dataclass
class LeaveATaskBehind(ControlProtocol):
    """Allows its one sample, and leaves a task that calls the step after it is decided."""

    late_call: LateCall
    late_tasks: list[asyncio.Task[object]] = field(default_factory=list)

    async def decide(self, step: PendingStep) -> StepDecision:
        [sample] = await step.sample(count=1)

        async def call_later() -> object:
            await asyncio.sleep(0.05)
            return await self.late_call(step)

        self.late_tasks.append(asyncio.get_running_loop().create_task(call_later()))
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=False,
        )


@pytest.mark.parametrize("late_call", [draw_another_sample, draw_no_sample, request_a_trusted_step])
def test_a_task_left_running_after_the_step_is_refused_the_model(
    run_mode: RunMode,
    late_call: LateCall,
) -> None:
    # Arrange: the run happens inside a running loop, as in a notebook
    model = ScriptedChatModel(responses=[AIMessage("answer A"), AIMessage("answer B")])
    monitor = KeywordMonitor()
    protocol = LeaveATaskBehind(late_call=late_call)
    agent = create_agent(
        model,
        tools=[],
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )
    payload = {"messages": [HumanMessage("Summarise q3.md.")]}

    async def run_and_wait() -> tuple[dict[str, Any], int, BaseException | None]:
        if run_mode == "invoke":
            result = agent.invoke(payload)
        else:
            result = await agent.ainvoke(payload)
        calls_at_return = len(model.calls)
        (late_task,) = protocol.late_tasks
        await asyncio.wait([late_task])
        return result, calls_at_return, late_task.exception()

    # Act
    result, calls_at_return, late_error = asyncio.run(run_and_wait())

    # Assert
    assert len(result["monitor_log"]) == 1
    assert len(model.calls) == calls_at_return == 1
    assert len(monitor.inputs) == 1
    expected_error, expected_reason = CLOSED_STEP_REFUSALS[run_mode]
    assert type(late_error) is expected_error
    assert expected_reason in str(late_error)
