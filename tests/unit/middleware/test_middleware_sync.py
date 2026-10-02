"""One protocol serves invoke() and ainvoke(), and fails loudly when invoke() cannot drive it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Monitor,
    MonitorInput,
    Outcome,
    PendingStep,
    StepDecision,
    Verdict,
)
from langchain_sync_monitors.errors import SynchronousRunError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    build_keyword_monitor,
    build_task_input,
    read_texts,
    run_agent,
)
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


def decide_outside_a_graph(
    middleware: MonitorMiddleware,
    request: ModelRequest[Any],
    *,
    mode: RunMode,
) -> ExtendedModelResponse[Any]:
    """Run the hook `mode` calls on a request built by hand, as no graph is running."""
    if mode == "invoke":
        return middleware.wrap_model_call(
            request,
            lambda inner: ModelResponse(result=[inner.model.invoke(inner.messages)]),
        )

    async def handle(inner: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[await inner.model.ainvoke(inner.messages)])

    return asyncio.run(middleware.awrap_model_call(request, handle))


def test_a_request_outside_a_graph_commits_without_a_stream_writer(
    run_mode: RunMode,
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    request: ModelRequest[Any] = ModelRequest(
        model=answering_model,
        messages=[HumanMessage("Summarise.")],
    )

    # Act
    result = decide_outside_a_graph(middleware, request, mode=run_mode)

    # Assert
    assert isinstance(result, ExtendedModelResponse)
    assert read_texts(result.model_response.result) == [ANSWER]
    assert result.command is not None
    assert isinstance(result.command.update, dict)
    [record] = result.command.update["monitor_log"]
    assert record["step_number"] == 1


type LeftBehindCall = Literal["sample", "trusted_step"]


@dataclass(kw_only=True)
class LeavesAModelCallBehind(ControlProtocol):
    """Runs its first sample and schedules a model call it never awaits, as a buggy one might.

    The call left behind is a second draw or a trusted step, as `call` says.
    """

    call: LeftBehindCall
    trusted_model: ScriptedChatModel
    left_behind: list[asyncio.Future[None]] = field(default_factory=list)

    async def call_a_model(self, step: PendingStep) -> None:
        if self.call == "sample":
            await step.sample(count=1)
        else:
            await step.request_trusted_step(trusted_model=self.trusted_model)

    async def decide(self, step: PendingStep) -> StepDecision:
        [sample] = await step.sample(count=1)
        self.left_behind.append(asyncio.ensure_future(self.call_a_model(step)))
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=False,
        )


@pytest.mark.parametrize("call", ["sample", "trusted_step"])
async def test_a_model_call_left_behind_by_invoke_never_reaches_a_model(
    answering_model: ScriptedChatModel,
    call: LeftBehindCall,
) -> None:
    # Arrange
    trusted_model = ScriptedChatModel(responses=[AIMessage("I will summarise.")])
    protocol = LeavesAModelCallBehind(call=call, trusted_model=trusted_model)
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    agent = create_agent(answering_model, middleware=[middleware])

    # Act: the step is committed, then the loop runs the call it left behind
    agent.invoke(build_task_input())
    await asyncio.sleep(0.05)

    # Assert
    assert len(answering_model.calls) == 1
    assert trusted_model.calls == []
    [left_behind] = protocol.left_behind
    assert isinstance(left_behind.exception(), SynchronousRunError)


@dataclass(kw_only=True)
class EventLoopMonitor(Monitor):
    """A monitor whose `evaluate_sync` reaches for an event loop, which invoke() does not run."""

    error_text: str | None = None

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return Verdict(suspicion=0.0, reason="Looks fine.")

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        if self.error_text is not None:
            raise RuntimeError(self.error_text)
        asyncio.get_running_loop()
        return Verdict(suspicion=0.0, reason="unreachable")


def test_invoke_names_a_monitor_that_needs_an_event_loop(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=EventLoopMonitor(), protocol=AcceptFirst())
    agent = create_agent(answering_model, middleware=[middleware])

    # Act
    with pytest.raises(SynchronousRunError) as raised:
        run_agent(agent, mode="invoke")

    # Assert
    message = str(raised.value)
    assert "The monitor EventLoopMonitor" in message
    assert "evaluate_sync" in message
    assert "control protocol" not in message


def test_ainvoke_runs_the_monitor_that_needs_an_event_loop(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=EventLoopMonitor(), protocol=AcceptFirst())
    agent = create_agent(answering_model, middleware=[middleware])

    # Act
    result = run_agent(agent, mode="ainvoke")

    # Assert
    assert [record["outcome"] for record in result["monitor_log"]] == ["allowed"]


def test_invoke_passes_on_a_monitor_error_that_is_not_about_an_event_loop(
    answering_model: ScriptedChatModel,
) -> None:
    # Arrange
    monitor = EventLoopMonitor(error_text="the monitor is misconfigured")
    middleware = MonitorMiddleware(monitor=monitor, protocol=AcceptFirst())
    agent = create_agent(answering_model, middleware=[middleware])

    # Act / Assert
    with pytest.raises(RuntimeError, match="the monitor is misconfigured"):
        run_agent(agent, mode="invoke")
