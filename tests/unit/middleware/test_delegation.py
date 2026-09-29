"""Each tool call hands a subagent its delegation; an agent without subagents is unchanged."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors.contracts import Delegation, StepRecord
from langchain_sync_monitors.delegation import add_delegation
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel

RECORD_KEYS = {
    "agent",
    "monitor",
    "step_number",
    "outcome",
    "flagged",
    "blocked_count",
    "samples",
}
PARENT_DELEGATION = Delegation(
    tool_call_id="call-parent",
    delegating_agent="main",
    blocks_before={"monitor": 5},
)


def build_record(*, monitor: str, blocked_count: int) -> StepRecord:
    return StepRecord(
        agent="main",
        monitor=monitor,
        step_number=1,
        outcome="steered" if blocked_count else "allowed",
        flagged=bool(blocked_count),
        blocked_count=blocked_count,
        samples=[],
    )


def build_tool_request(*, state: dict[str, Any], call_id: str | None) -> ToolCallRequest:
    runtime = ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={"name": "task", "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state=state,
        runtime=runtime,
    )


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=AutoMode(block_threshold=0.6)
    )


@pytest.fixture
def subagent_state() -> dict[str, Any]:
    return {
        "messages": [],
        "monitor_log": [
            build_record(monitor="monitor", blocked_count=2),
            build_record(monitor="other", blocked_count=1),
            build_record(monitor="monitor", blocked_count=0),
        ],
        "monitor_delegation": PARENT_DELEGATION,
    }


def test_a_tool_call_hands_on_the_thread_s_blocks_by_monitor(
    run_mode: RunMode,
    middleware: MonitorMiddleware,
    subagent_state: dict[str, Any],
) -> None:
    # Arrange
    request = build_tool_request(state=subagent_state, call_id="call-child")
    seen: list[ToolCallRequest] = []

    def handle(inner: ToolCallRequest) -> ToolMessage:
        seen.append(inner)
        return ToolMessage("done", tool_call_id="call-child")

    async def handle_async(inner: ToolCallRequest) -> ToolMessage:
        return handle(inner)

    # Act
    if run_mode == "invoke":
        middleware.wrap_tool_call(request, handle)
    else:
        asyncio.run(middleware.awrap_tool_call(request, handle_async))

    # Assert
    expected = Delegation(
        tool_call_id="call-child",
        delegating_agent="main",
        blocks_before={"monitor": 7, "other": 1},
    )
    [handled] = seen
    assert handled.state["monitor_delegation"] == expected
    assert handled.runtime.state["monitor_delegation"] == expected
    assert subagent_state["monitor_delegation"] == PARENT_DELEGATION


def test_a_request_with_this_agent_s_delegation_for_the_call_is_passed_on_as_it_is(
    subagent_state: dict[str, Any],
) -> None:
    # Arrange
    request = build_tool_request(state=subagent_state, call_id="call-child")
    request = add_delegation(request, agent="worker")

    # Act
    again = add_delegation(request, agent="worker")

    # Assert
    assert again is request


def test_an_inherited_delegation_is_replaced_when_the_call_reuses_its_id(
    subagent_state: dict[str, Any],
) -> None:
    # Arrange
    request = build_tool_request(state=subagent_state, call_id="call-parent")

    # Act
    handed_on = add_delegation(request, agent="worker")

    # Assert
    assert handed_on.state["monitor_delegation"] == Delegation(
        tool_call_id="call-parent",
        delegating_agent="worker",
        blocks_before={"monitor": 7, "other": 1},
    )


def test_a_tool_call_without_an_id_is_passed_on_as_it_is(
    subagent_state: dict[str, Any],
) -> None:
    # Arrange
    request = build_tool_request(state=subagent_state, call_id=None)

    # Act
    handed_on = add_delegation(request, agent="worker")

    # Assert
    assert handed_on is request


def test_an_agent_without_subagents_records_and_stores_what_it_did_before(
    run_mode: RunMode,
    middleware: MonitorMiddleware,
) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Summary.")],
    )
    agent = create_agent(
        model,
        tools=workspace.build_tools(),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"plain-{run_mode}")

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    log = result["monitor_log"]
    assert [set(record) for record in log] == [RECORD_KEYS, RECORD_KEYS]
    assert [(record["outcome"], record["blocked_count"]) for record in log] == [
        ("steered", 1),
        ("allowed", 0),
    ]
    assert workspace.executed == ["read_file:q3.md"]
    assert "monitor_delegation" not in result
    assert "monitor_delegation" not in agent.get_state(config).values
