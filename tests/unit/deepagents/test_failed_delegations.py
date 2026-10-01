"""A known limit: the blocks inside a delegation that raises count only if the run is resumed.

A subagent's records reach its parent's log only through the result of the
`task` call, so a subagent whose run raises returns none. These tests pin
where Auto Mode's thread total misses those blocks, and where it keeps them.
"""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import SubAgent, create_deep_agent
from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    RunMode,
    Workspace,
    build_benign_post,
    build_exfiltration_step,
    build_keyword_monitor,
    build_task_input,
    build_thread_config,
    run_agent,
    run_payload,
    stream_subgraph_custom_events,
    summarise_records,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.deep_agents import build_delegation_step

MAX_TOTAL_BLOCKS = 3


def build_tools(*, lookups_that_fail: int) -> list[BaseTool]:
    """Build `http_post` and a `lookup` tool whose first calls raise, as any tool can."""
    failures: list[str] = []

    @tool
    def lookup(key: str) -> str:
        """Look a key up in the team's notes."""
        if len(failures) < lookups_that_fail:
            failures.append(key)
            raise KeyError(key)
        return "not found"

    return [*Workspace().build_http_tools(), lookup]


def build_lookup_step(*, call_id: str) -> AIMessage:
    return build_tool_call_message(tool_name="lookup", call_id=call_id, arguments={"key": "q9"})


def build_two_steered_steps(*, prefix: str) -> list[AIMessage]:
    """Script two steps that are each blocked once, then run a benign post: two blocks."""
    return [
        build_exfiltration_step(call_id=f"call-post-{prefix}-1"),
        build_benign_post(call_id=f"call-benign-{prefix}-1"),
        build_exfiltration_step(call_id=f"call-post-{prefix}-2"),
        build_benign_post(call_id=f"call-benign-{prefix}-2"),
    ]


def build_agent(
    *,
    main_model: ScriptedChatModel,
    worker_model: ScriptedChatModel,
    lookups_that_fail: int,
    middleware: list[AgentMiddleware[Any, Any, Any]] | None = None,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    main_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=0.6, max_total_blocks=MAX_TOTAL_BLOCKS),
    )
    worker = SubAgent(name="worker", description="Finds sources.", model=worker_model)
    stack: list[AgentMiddleware[Any, Any, Any]] = [*(middleware or []), main_monitor]
    return create_deep_agent(
        model=main_model,
        tools=build_tools(lookups_that_fail=lookups_that_fail),
        middleware=stack,
        subagents=monitor_subagents(middleware=main_monitor, subagents=[worker]),
        checkpointer=checkpointer,
    )


def render_error_message(error: Exception, request: ToolCallRequest) -> str:
    return f"{request.tool_call['name']} failed with {type(error).__name__}."


@pytest.mark.parametrize(
    "answering",
    [
        ToolRetryMiddleware(max_retries=0, initial_delay=0, jitter=False),
        ToolErrorMiddleware(on_error=render_error_message),
    ],
    ids=["tool-retry", "tool-error"],
)
def test_blocks_inside_a_delegation_answered_with_an_error_are_not_counted(
    run_mode: RunMode,
    answering: AgentMiddleware[Any, Any, Any],
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            build_delegation_step(call_id="call-task-2"),
            AIMessage("Done."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            *build_two_steered_steps(prefix="first"),
            build_lookup_step(call_id="call-lookup"),
            *build_two_steered_steps(prefix="second"),
            AIMessage("Report."),
        ],
    )
    agent = build_agent(
        main_model=main_model,
        worker_model=worker_model,
        lookups_that_fail=1,
        middleware=[answering],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the first delegation's two blocks are missing, so the thread has had four.
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-2"),
        ("worker", "steered", 1, "call-task-2"),
        ("worker", "allowed", 0, "call-task-2"),
        ("main", "allowed", 0, None),
    ]
    [failed] = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.status == "error"
    ]
    assert failed.tool_call_id == "call-task-1"
    assert len(worker_model.calls) == 10


def test_a_retried_delegation_starts_again_from_the_same_count(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(call_id="call-task-1"), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(
        responses=[
            *build_two_steered_steps(prefix="first"),
            build_lookup_step(call_id="call-lookup"),
            *build_two_steered_steps(prefix="retry"),
            AIMessage("Report."),
        ],
    )
    retry = ToolRetryMiddleware(max_retries=1, initial_delay=0, jitter=False)
    agent = build_agent(
        main_model=main_model,
        worker_model=worker_model,
        lookups_that_fail=1,
        middleware=[retry],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the retry steered twice more from a count of 0, four blocks against three.
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "allowed", 0, "call-task-1"),
        ("main", "allowed", 0, None),
    ]
    assert len(worker_model.calls) == 10


def test_blocks_inside_a_crashed_delegation_are_not_counted_on_the_next_turn(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            build_delegation_step(call_id="call-task-2"),
            AIMessage("Second answer."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            *build_two_steered_steps(prefix="first"),
            build_lookup_step(call_id="call-lookup"),
            *build_two_steered_steps(prefix="second"),
            AIMessage("Second report."),
        ],
    )
    agent = build_agent(
        main_model=main_model,
        worker_model=worker_model,
        lookups_that_fail=1,
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"next-turn-{run_mode}")
    with pytest.raises(KeyError):
        run_payload(agent, build_task_input(), mode=run_mode, config=config)

    # Act
    result = run_payload(
        agent,
        {"messages": [HumanMessage("Try again.")]},
        mode=run_mode,
        config=config,
    )

    # Assert: the crashed delegation's two blocks are missing, so the thread has had four.
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-2"),
        ("worker", "steered", 1, "call-task-2"),
        ("worker", "allowed", 0, "call-task-2"),
        ("main", "allowed", 0, None),
    ]


def test_blocks_inside_a_crashed_delegation_count_when_the_run_is_resumed(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(call_id="call-task-1"), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(
        responses=[
            *build_two_steered_steps(prefix="first"),
            build_lookup_step(call_id="call-lookup"),
            build_exfiltration_step(call_id="call-post-after-resume"),
            AIMessage("Report."),
        ],
    )
    agent = build_agent(
        main_model=main_model,
        worker_model=worker_model,
        lookups_that_fail=1,
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"resume-{run_mode}")
    with pytest.raises(KeyError):
        run_payload(agent, build_task_input(), mode=run_mode, config=config)

    # Act
    result = run_payload(agent, None, mode=run_mode, config=config)

    # Assert: the resumed worker counts its first two blocks and halts at the third.
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "allowed", 0, "call-task-1"),
        ("worker", "halted", 1, "call-task-1"),
        ("main", "halted", 0, None),
    ]
    assert len(worker_model.calls) == 6


def test_steps_inside_a_failed_delegation_still_stream_as_they_are_committed(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(call_id="call-task-1"), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(
        responses=[
            *build_two_steered_steps(prefix="first"),
            build_lookup_step(call_id="call-lookup"),
        ],
    )
    retry = ToolRetryMiddleware(max_retries=0, initial_delay=0, jitter=False)
    agent = build_agent(
        main_model=main_model,
        worker_model=worker_model,
        lookups_that_fail=1,
        middleware=[retry],
    )

    # Act
    events, error = stream_subgraph_custom_events(agent, mode=run_mode)

    # Assert
    assert error is None
    records = [event["record"] for event in events if event["type"] == "monitor_step"]
    assert summarise_records(records) == [
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "allowed", 0, "call-task-1"),
        ("main", "allowed", 0, None),
    ]
