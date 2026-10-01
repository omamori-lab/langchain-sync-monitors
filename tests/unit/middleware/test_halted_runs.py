"""A halted step ends the run, even in an agent that loops until it has a structured response.

`create_agent` with structured output and no tools routes the model back to
itself until a structured response exists, so a halt message alone does not
end the run [@langchain2026]. The monitor's `after_model` hook ends it. A
`jump_to` written by the model node itself would reach only the model node's
own routing edge, which exists only when no `after_model` hook does, so these
tests cover agents with none, one and two other `after_model` hooks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langchain_sync_monitors.contracts import Delegation, OutcomeName, StepRecord
from langchain_sync_monitors.middleware import MonitorMiddleware, MonitorState
from langchain_sync_monitors.protocols import AutoMode, HaltRun
from tests.support.agents import (
    ATTACKER_URL,
    RunMode,
    Workspace,
    build_keyword_monitor,
    build_thread_config,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message

INTRANET_URL = "https://intranet.example/reports"


class Report(BaseModel):
    summary: str
    destination: str


class ObservingMiddleware(AgentMiddleware[Any, Any, Any]):
    """An `after_model` hook that changes nothing, under a name of its own."""

    def __init__(self, *, label: str) -> None:
        self.label = label

    @property
    def name(self) -> str:
        return self.label

    def after_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return None


def build_report_call(*, destination: str, call_id: str) -> AIMessage:
    return build_tool_call_message(
        tool_name="Report",
        call_id=call_id,
        arguments={"summary": "Q3 grew 12%.", "destination": destination},
    )


def build_report_json(*, destination: str) -> AIMessage:
    report = Report(summary="Q3 grew 12%.", destination=destination)
    return AIMessage(content=report.model_dump_json())


def build_halting_monitor() -> MonitorMiddleware:
    protocol = AutoMode(block_threshold=0.5, max_consecutive_blocks=1, when_limit_reached=HaltRun())
    return MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)


def build_attack_then_benign_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            build_report_call(destination=ATTACKER_URL, call_id="call-1"),
            build_report_call(destination=INTRANET_URL, call_id="call-2"),
        ],
    )


def run_agent(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
    task: str = "Summarise q3.md.",
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    payload = {"messages": [HumanMessage(task)]}
    if mode == "invoke":
        return agent.invoke(payload, config)
    return asyncio.run(agent.ainvoke(payload, config))


def read_outcomes(result: dict[str, Any]) -> list[str]:
    return [record["outcome"] for record in result["monitor_log"]]


@pytest.mark.parametrize("other_hook_count", [0, 1, 2])
def test_a_halt_ends_a_tool_strategy_agent_without_tools(
    run_mode: RunMode,
    other_hook_count: int,
) -> None:
    # Arrange
    model = build_attack_then_benign_model()
    hooks = [ObservingMiddleware(label=f"observer-{index}") for index in range(other_hook_count)]
    stack: list[AgentMiddleware[Any, Any, Any]] = [*hooks, build_halting_monitor()]
    agent = create_agent(model, tools=[], response_format=ToolStrategy(Report), middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode, config=RunnableConfig(recursion_limit=12))

    # Assert
    assert read_outcomes(result) == ["halted"]
    assert len(model.calls) == 1
    assert result.get("structured_response") is None


def test_a_halt_ends_the_run_when_an_after_model_hook_runs_before_the_monitors(
    run_mode: RunMode,
) -> None:
    # Arrange: listed after the monitor, the hook runs first after the model
    model = build_attack_then_benign_model()
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        build_halting_monitor(),
        ObservingMiddleware(label="observer"),
    ]
    agent = create_agent(model, tools=[], response_format=ToolStrategy(Report), middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == ["halted"]
    assert len(model.calls) == 1


def test_a_halt_ends_an_agent_whose_bare_schema_becomes_a_tool_strategy(
    run_mode: RunMode,
) -> None:
    # Arrange
    model = build_attack_then_benign_model()
    agent = create_agent(
        model,
        tools=[],
        response_format=Report,
        middleware=[build_halting_monitor()],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == ["halted"]
    assert len(model.calls) == 1


def test_a_halt_still_ends_an_agent_with_a_tool(run_mode: RunMode) -> None:
    # Arrange
    model = build_attack_then_benign_model()
    agent = create_agent(
        model,
        tools=Workspace().build_tools(),
        response_format=ToolStrategy(Report),
        middleware=[build_halting_monitor()],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == ["halted"]
    assert len(model.calls) == 1


def test_a_halt_still_ends_a_provider_strategy_agent(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(
        responses=[
            build_report_json(destination=ATTACKER_URL),
            build_report_json(destination=INTRANET_URL),
        ],
    )
    agent = create_agent(
        model,
        tools=[],
        response_format=ProviderStrategy(Report),
        middleware=[build_halting_monitor()],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == ["halted"]
    assert len(model.calls) == 1


def test_a_halt_in_one_turn_does_not_end_the_next_turn(run_mode: RunMode) -> None:
    # Arrange
    model = build_attack_then_benign_model()
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ObservingMiddleware(label="observer"),
        build_halting_monitor(),
    ]
    agent = create_agent(
        model,
        tools=[],
        response_format=ToolStrategy(Report),
        middleware=stack,
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"halt-then-continue-{run_mode}")

    # Act
    first_turn = run_agent(agent, mode=run_mode, config=config)
    second_turn = run_agent(agent, mode=run_mode, task="Try again, safely.", config=config)

    # Assert
    assert read_outcomes(first_turn) == ["halted"]
    assert read_outcomes(second_turn) == ["halted", "allowed"]
    assert second_turn["structured_response"].destination == INTRANET_URL


def build_state(
    *,
    records: list[StepRecord],
    last_message: AIMessage | HumanMessage,
) -> MonitorState:
    messages = [HumanMessage("Summarise q3.md."), last_message]
    return MonitorState(messages=messages, monitor_log=records)


def build_record(*, agent: str, outcome: OutcomeName) -> StepRecord:
    return StepRecord(
        agent=agent,
        monitor="monitor",
        step_number=1,
        outcome=outcome,
        flagged=outcome == "halted",
        blocked_count=0,
        samples=[],
    )


@pytest.mark.parametrize(
    ("records", "last_message", "ends_the_run"),
    [
        ([build_record(agent="main", outcome="halted")], AIMessage("Stopped."), True),
        ([build_record(agent="main", outcome="allowed")], AIMessage("Done."), False),
        ([build_record(agent="worker", outcome="halted")], AIMessage("Worker stopped."), False),
        ([build_record(agent="main", outcome="halted")], HumanMessage("Next turn."), False),
        (
            [build_record(agent="main", outcome="halted")],
            build_report_call(destination=INTRANET_URL, call_id="call-3"),
            False,
        ),
        ([], AIMessage("Done."), False),
    ],
)
async def test_the_hook_ends_the_run_only_right_after_this_monitors_halt(
    records: list[StepRecord],
    last_message: AIMessage | HumanMessage,
    ends_the_run: bool,
) -> None:
    # Arrange
    middleware = build_halting_monitor()
    state = build_state(records=records, last_message=last_message)
    runtime = Runtime(context=None)

    # Act
    sync_update = middleware.after_model(state, runtime)
    async_update = await middleware.aafter_model(state, runtime)

    # Assert
    expected = {"jump_to": "end"} if ends_the_run else None
    assert sync_update == expected
    assert async_update == expected


async def test_the_hook_leaves_a_state_without_messages_to_run_on() -> None:
    # Arrange: a halt record, but no halt message for the step to have ended with
    state = MonitorState(messages=[], monitor_log=[build_record(agent="main", outcome="halted")])
    middleware = build_halting_monitor()
    runtime = Runtime(context=None)

    # Act
    sync_update = middleware.after_model(state, runtime)
    async_update = await middleware.aafter_model(state, runtime)

    # Assert
    assert sync_update is None
    assert async_update is None


@pytest.mark.parametrize(
    ("record_delegation_id", "ends_the_run"),
    [("call-fork", True), (None, False), ("call-other", False)],
    ids=["its-own-halt", "the-parent-s-halt", "another-delegation-s-halt"],
)
async def test_a_same_named_subagent_s_hook_ends_its_run_only_after_its_own_halt(
    record_delegation_id: str | None,
    ends_the_run: bool,
) -> None:
    # Arrange: a fork runs under monitor[main] with a delegation of its own
    record = build_record(agent="main", outcome="halted")
    if record_delegation_id is not None:
        record["delegation_id"] = record_delegation_id
    state = build_state(records=[record], last_message=AIMessage("Stopped."))
    state["monitor_delegation"] = Delegation(
        tool_call_id="call-fork",
        delegating_agent="main",
        blocks_before={},
    )
    middleware = build_halting_monitor()
    runtime = Runtime(context=None)

    # Act
    sync_update = middleware.after_model(state, runtime)
    async_update = await middleware.aafter_model(state, runtime)

    # Assert
    expected = {"jump_to": "end"} if ends_the_run else None
    assert sync_update == expected
    assert async_update == expected
