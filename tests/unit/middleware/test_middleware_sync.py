"""One protocol serves invoke() and ainvoke(), and fails loudly when invoke() cannot drive it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Outcome,
    PendingStep,
    Sample,
    StepDecision,
)
from langchain_sync_monitors.errors import SynchronousRunError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import build_keyword_monitor, build_task_input, read_texts, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.protocols import AcceptFirst, AwaitsEventLoop

ANSWER = "Q3 revenue grew 12%."


@pytest.fixture
def answering_model() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[AIMessage(ANSWER)])


@pytest.mark.parametrize("delay", [0.0, 0.01])
def test_invoke_rejects_a_protocol_that_awaits_real_async_work(
    answering_model: ScriptedChatModel,
    delay: float,
) -> None:
    # Arrange
    protocol = AwaitsEventLoop(delay=delay)
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    agent = create_agent(answering_model, middleware=[middleware])

    # Act
    with pytest.raises(SynchronousRunError):
        run_agent(agent, mode="invoke")

    # Assert
    assert protocol.finished_cleanly == [True]


def test_ainvoke_runs_the_same_protocol(answering_model: ScriptedChatModel) -> None:
    # Arrange
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AwaitsEventLoop(delay=0.01),
    )
    agent = create_agent(answering_model, middleware=[middleware])

    # Act
    result = run_agent(agent, mode="ainvoke")

    # Assert
    assert read_texts(result["messages"])[-1] == ANSWER
    assert result["monitor_log"][0]["outcome"] == "allowed"


async def test_invoke_works_inside_a_running_event_loop(answering_model: ScriptedChatModel) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    agent = create_agent(answering_model, middleware=[middleware])

    # Act
    result = agent.invoke(build_task_input())

    # Assert
    assert read_texts(result["messages"])[-1] == ANSWER
    assert len(result["monitor_log"]) == 1


async def test_invoke_inside_a_running_loop_still_rejects_real_async_work(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AwaitsEventLoop(delay=0.01),
    )
    agent = create_agent(answering_model, middleware=[middleware])

    # Act / Assert
    with pytest.raises(SynchronousRunError):
        agent.invoke(build_task_input())


def test_a_request_outside_a_graph_commits_without_a_stream_writer(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    request: ModelRequest[Any] = ModelRequest(
        model=answering_model,
        messages=[HumanMessage("Summarise.")],
    )

    # Act
    result = middleware.wrap_model_call(
        request,
        lambda inner: ModelResponse(result=[inner.model.invoke(inner.messages)]),
    )

    # Assert
    assert isinstance(result, ExtendedModelResponse)
    assert read_texts(result.model_response.result) == [ANSWER]
    assert result.command is not None
    assert isinstance(result.command.update, dict)
    [record] = result.command.update["monitor_log"]
    assert record["step_number"] == 1


@dataclass(kw_only=True)
class LeavesADrawBehind(ControlProtocol):
    """Runs its first sample and schedules a second draw it never awaits, as a buggy one might."""

    left_behind: list[asyncio.Future[tuple[Sample, ...]]] = field(default_factory=list)

    async def decide(self, step: PendingStep) -> StepDecision:
        [sample] = await step.sample(count=1)
        self.left_behind.append(asyncio.ensure_future(step.sample(count=1)))
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=False,
        )


async def test_a_draw_left_behind_by_invoke_never_reaches_the_model(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    protocol = LeavesADrawBehind()
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    agent = create_agent(answering_model, middleware=[middleware])

    # Act: the step is committed, then the loop runs the draw it left behind
    agent.invoke(build_task_input())
    await asyncio.sleep(0.05)

    # Assert
    assert len(answering_model.calls) == 1
    [left_behind] = protocol.left_behind
    assert isinstance(left_behind.exception(), SynchronousRunError)
