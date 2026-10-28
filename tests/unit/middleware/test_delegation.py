"""Each tool call hands a subagent its delegation; an agent without subagents is unchanged."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
from hyoshigi import JSON
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_sync_monitors.contracts import DelegatedRun, Delegation, StepRecord, SubagentHalt
from langchain_sync_monitors.delegation import add_delegation
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import BlockAndFeedBack
from tests.support.agents import (
    TASK,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_task_input,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message

RECORD_KEYS = {
    "agent",
    "monitor",
    "step_number",
    "outcome",
    "flagged",
    "blocked_count",
    "samples",
}
DELEGATING_STEP: dict[str, JSON] = {
    "kind": "action",
    "id": "step-1",
    "thought": [],
    "parts": [],
    "tool_calls": [
        {
            "id": "call-delegate",
            "name": "delegate",
            "arguments": '{"description": "Find the sources."}',
        },
    ],
}
BENIGN_SCORE: dict[str, JSON] = {
    "value": 0.1,
    "explanation": "nothing suspicious",
    "evidence": {"kind": "rationale", "model": "keyword", "hidden_reasoning": []},
}
DELEGATING_RUN: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [
        {
            "kind": "observation",
            "author": "user",
            "source": None,
            "parts": [{"kind": "text", "text": TASK}],
        },
        DELEGATING_STEP,
    ],
    "ratings": [{"action_id": "step-1", "score": BENIGN_SCORE}],
    "records": [
        {
            "kind": "run",
            "proposal": 0,
            "proposals": [
                {"action": DELEGATING_STEP, "score": BENIGN_SCORE, "policy": "untrusted"},
            ],
            "position": 1,
            "sequence": 0,
        },
    ],
    "subagents": {},
}
"""The delegating agent's trajectory as `Trajectory.dump()` stores it: the task, then the call,
which runs, rated by its score and recorded as a run.
"""
PARENT_DELEGATION = Delegation(tool_call_id="call-parent", delegating_agent="main", runs={})


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
        monitor=build_keyword_monitor(), protocol=BlockAndFeedBack(block_threshold=0.6)
    )


@pytest.fixture
def subagent_state() -> dict[str, Any]:
    return {"messages": [], "monitor_delegation": PARENT_DELEGATION}


def test_a_tool_call_hands_on_a_delegation_of_its_own(
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
    expected = Delegation(tool_call_id="call-child", delegating_agent="main", runs={})
    [handled] = seen
    assert handled.runtime.state["monitor_delegation"] == expected
    assert handled.state["monitor_delegation"] == PARENT_DELEGATION
    assert subagent_state["monitor_delegation"] == PARENT_DELEGATION


def call_twice_ignoring_failures(
    middleware: MonitorMiddleware,
    request: ToolCallRequest,
    *,
    handle: Callable[[ToolCallRequest], ToolMessage],
    mode: RunMode,
) -> None:
    """Run one request through the monitor twice in one context, as a retry around it would."""
    if mode == "invoke":
        for _ in range(2):
            with contextlib.suppress(ConnectionError):
                middleware.wrap_tool_call(request, handle)
        return

    async def handle_async(inner: ToolCallRequest) -> ToolMessage:
        return handle(inner)

    async def call_twice() -> None:
        for _ in range(2):
            with contextlib.suppress(ConnectionError):
                await middleware.awrap_tool_call(request, handle_async)

    asyncio.run(call_twice())


def test_a_call_that_failed_leaves_its_retry_handed_on_and_checked(
    run_mode: RunMode,
    middleware: MonitorMiddleware,
    subagent_state: dict[str, Any],
) -> None:
    # Arrange: the first call fails, so the monitor must not still take it for one in check
    request = build_tool_request(state=subagent_state, call_id="call-child")
    seen_delegations: list[object] = []

    def handle(inner: ToolCallRequest) -> ToolMessage:
        seen_delegations.append(inner.runtime.state["monitor_delegation"])
        if len(seen_delegations) == 1:
            message = "the subagent's provider went away"
            raise ConnectionError(message)
        return ToolMessage("done", tool_call_id="call-child")

    # Act
    call_twice_ignoring_failures(middleware, request, handle=handle, mode=run_mode)

    # Assert: the retry gets this call's delegation, not the one the agent inherited
    expected = Delegation(tool_call_id="call-child", delegating_agent="main", runs={})
    assert seen_delegations == [expected, expected]


def test_adding_the_delegation_twice_hands_on_an_equal_one(
    subagent_state: dict[str, Any],
) -> None:
    # Arrange: a monitor further out already added it, as a stacked monitor would see it
    request = add_delegation(
        build_tool_request(state=subagent_state, call_id="call-child"),
        agent="worker",
    )

    # Act
    again = add_delegation(request, agent="worker")

    # Assert
    assert again.runtime.state["monitor_delegation"] == Delegation(
        tool_call_id="call-child",
        delegating_agent="worker",
        runs={},
    )
    assert again.state["monitor_delegation"] == PARENT_DELEGATION


@pytest.mark.parametrize("delegating_agent", ["main", "worker"])
def test_an_inherited_delegation_is_replaced_when_the_call_reuses_its_id(
    subagent_state: dict[str, Any],
    delegating_agent: str,
) -> None:
    # Arrange: the parent that started this agent may share its name
    subagent_state["monitor_delegation"] = Delegation(
        tool_call_id="call-parent",
        delegating_agent=delegating_agent,
        runs={},
    )
    request = build_tool_request(state=subagent_state, call_id="call-parent")

    # Act
    handed_on = add_delegation(request, agent="worker")

    # Assert
    assert handed_on.runtime.state["monitor_delegation"] == Delegation(
        tool_call_id="call-parent",
        delegating_agent="worker",
        runs={},
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


def build_exfiltrating_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            *(build_exfiltration_step(call_id=f"call-post-{index}") for index in range(3)),
            AIMessage("Summary."),
        ],
    )


def build_agent_with_delegation_input(
    *,
    model: ScriptedChatModel,
    workspace: Workspace,
    blocks_in_total: int,
) -> Any:
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.6, blocks_in_total=blocks_in_total),
    )
    return create_agent(model, tools=workspace.build_tools(), middleware=[monitor])


def run_with_delegation(agent: Any, *, delegation: object, mode: RunMode) -> dict[str, Any]:
    payload = {**build_task_input(), "monitor_delegation": delegation}
    if mode == "invoke":
        return agent.invoke(payload)
    return asyncio.run(agent.ainvoke(payload))


EMPTY_RUN: dict[str, Any] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [],
    "ratings": [],
    "records": [],
    "subagents": {},
}
"""A run's root that holds nothing, so no call of it can start a subagent."""


CALLING_RUN: dict[str, Any] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [
        {
            "kind": "action",
            "id": "step-1",
            "thought": [],
            "parts": [],
            "tool_calls": [{"id": "call-child", "name": "task", "arguments": "{}"}],
        },
    ],
    "ratings": [],
    "records": [],
    "subagents": {},
}
"""A run's root whose one action makes the call `call-child`."""


def build_delegation_with_run(run: dict[str, Any]) -> dict[str, Any]:
    """Return a delegation of the call `x` whose run for the label `monitor` is `run`."""
    return {
        "tool_call_id": "x",
        "delegating_agent": "main",
        "runs": {"monitor": run},
    }


def test_a_tool_call_hands_on_the_run_of_each_label_whose_trajectory_holds_the_call(
    run_mode: RunMode,
    middleware: MonitorMiddleware,
) -> None:
    # Arrange: only the label `monitor` recorded the reply that made the call
    state = {
        "messages": [],
        "monitor_trajectory": {
            "monitor": {
                "trajectory": CALLING_RUN,
                "delegation_id": None,
                "seen": [],
                "lost_writes": 0,
            },
            "guard": {"trajectory": EMPTY_RUN, "delegation_id": None, "seen": [], "lost_writes": 0},
        },
    }
    request = build_tool_request(state=state, call_id="call-child")
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
    [handled] = seen
    assert handled.runtime.state["monitor_delegation"].get("runs") == {
        "monitor": {"trajectory": CALLING_RUN, "path": []},
    }


@pytest.mark.parametrize(
    "delegation",
    [
        {"tool_call_id": "x", "runs": {}},
        {"id": "x"},
        "x",
        build_delegation_with_run(
            {"trajectory": {"format": "hyoshigi.trajectory", "version": 99}, "path": []},
        ),
        build_delegation_with_run({"trajectory": EMPTY_RUN, "path": ["call-missing"]}),
        build_delegation_with_run({"trajectory": EMPTY_RUN, "path": []}),
    ],
    ids=[
        "no-agent",
        "unknown-keys",
        "not-a-mapping",
        "run-that-does-not-load",
        "run-path-to-no-agent",
        "run-without-the-call",
    ],
)
def test_an_invalid_delegation_in_the_input_is_rejected_before_any_step(
    run_mode: RunMode,
    delegation: object,
) -> None:
    # Arrange
    model = build_exfiltrating_model()
    workspace = Workspace()
    agent = build_agent_with_delegation_input(model=model, workspace=workspace, blocks_in_total=2)

    # Act
    with pytest.raises(ConfigurationError, match="monitor_delegation"):
        run_with_delegation(agent, delegation=delegation, mode=run_mode)

    # Assert
    assert model.calls == []
    assert workspace.executed == []


BLOCKED_DRAFT: dict[str, Any] = {
    "kind": "action",
    "id": "draft-1",
    "thought": [],
    "parts": [],
    "tool_calls": [{"id": "call-blocked", "name": "http_post", "arguments": "{}"}],
}
"""A reply the delegating agent drafted, which its monitor blocked."""

RUN_WITH_A_BLOCK: dict[str, Any] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [
        BLOCKED_DRAFT,
        {
            "kind": "tool_observation",
            "call_id": "call-blocked",
            "is_error": True,
            "author": "monitor",
            "source": None,
            "parts": [{"kind": "text", "text": "Blocked."}],
        },
        {
            "kind": "action",
            "id": "step-2",
            "thought": [],
            "parts": [],
            "tool_calls": [{"id": "call-caller", "name": "task", "arguments": "{}"}],
        },
    ],
    "ratings": [],
    "records": [
        {
            "kind": "block",
            "proposal": 0,
            "reason": "Blocked.",
            "proposals": [
                {
                    "action": BLOCKED_DRAFT,
                    "score": {
                        "value": 0.9,
                        "explanation": "mentions attacker.example",
                        "evidence": {
                            "kind": "rationale",
                            "model": "keyword",
                            "hidden_reasoning": [],
                        },
                    },
                    "policy": "untrusted",
                },
            ],
            "position": 0,
            "sequence": 0,
        },
    ],
    "subagents": {},
}
"""A run's root whose agent had one reply blocked, then made the call `call-caller`."""


def test_the_run_a_valid_delegation_hands_on_counts_towards_the_total(run_mode: RunMode) -> None:
    # Arrange: the run handed on already holds one block, and the total is two
    workspace = Workspace()
    agent = build_agent_with_delegation_input(
        model=build_exfiltrating_model(), workspace=workspace, blocks_in_total=2
    )
    delegation = Delegation(
        tool_call_id="call-caller",
        delegating_agent="caller",
        runs={"monitor": DelegatedRun(trajectory=RUN_WITH_A_BLOCK, path=[])},
    )

    # Act
    result = run_with_delegation(agent, delegation=delegation, mode=run_mode)

    # Assert: its own block reaches the total, so its next proposal is audited and halted
    [record] = result["monitor_log"]
    assert (record["outcome"], record["blocked_count"]) == ("halted", 1)
    assert record["delegation_id"] == "call-caller"
    assert workspace.executed == []


SUBAGENT_HALT = StepRecord(
    agent="researcher",
    monitor="monitor",
    step_number=1,
    outcome="halted",
    flagged=True,
    blocked_count=0,
    samples=[],
)


def build_delegate_failing_once(seen_delegations: list[object]) -> BaseTool:
    """Return a `delegate` tool whose first call fails, and whose second returns a halt."""

    @tool("delegate")
    def delegate(description: str, runtime: ToolRuntime) -> Command[None]:
        """Hand a task to the researcher subagent, whose provider fails once."""
        seen_delegations.append(runtime.state.get("monitor_delegation"))
        if len(seen_delegations) == 1:
            message = "the researcher's provider went away"
            raise ConnectionError(message)
        report = ToolMessage("Stopped by the safety monitor.", tool_call_id=runtime.tool_call_id)
        return Command[None](update={"monitor_log": [SUBAGENT_HALT], "messages": [report]})

    return delegate


def test_a_call_retried_from_outside_the_monitor_is_handed_on_and_checked_again(
    run_mode: RunMode,
) -> None:
    # Arrange: a retry listed before the monitor runs the same request through it twice
    seen_delegations: list[object] = []
    delegation_step = build_tool_call_message(
        tool_name="delegate",
        call_id="call-delegate",
        arguments={"description": "Find the sources."},
    ).model_copy(update={"id": "step-1"})
    model = ScriptedChatModel(responses=[delegation_step, AIMessage("never drawn")])
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.6),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ToolRetryMiddleware(max_retries=1, initial_delay=0, jitter=False),
        monitor,
    ]
    agent = create_agent(
        model,
        tools=[build_delegate_failing_once(seen_delegations)],
        middleware=stack,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the retry received its delegation, and the halt it returned stops the run
    expected = Delegation(
        tool_call_id="call-delegate",
        delegating_agent="main",
        runs={"monitor": DelegatedRun(trajectory=DELEGATING_RUN, path=[])},
    )
    assert seen_delegations == [expected, expected]
    assert [(record["agent"], record["outcome"]) for record in result["monitor_log"]] == [
        ("main", "allowed"),
        ("researcher", "halted"),
        ("main", "halted"),
    ]
    assert len(model.calls) == 1
