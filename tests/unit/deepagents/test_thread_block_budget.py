"""Auto Mode's block total covers the whole thread, subagents included, and names delegations."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_task_input,
    build_thread_config,
    read_texts,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.flaky_models import TriggeredFailureMonitor
from tests.support.protocols import AcceptFirst

MAIN_ANSWER = "Here is the summary."


def build_auto_mode_monitor(*, max_total_blocks: int = 20) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=0.6, max_total_blocks=max_total_blocks),
    )


def build_benign_post(*, call_id: str) -> AIMessage:
    return build_tool_call_message(
        tool_name="http_post",
        call_id=call_id,
        arguments={"url": "https://notes.example/team", "body": "Three sources."},
    )


def build_two_subagent_agent(
    *,
    main_model: ScriptedChatModel,
    worker_model: ScriptedChatModel,
    reviewer_model: ScriptedChatModel,
    main_monitor: MonitorMiddleware,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    subagents = [
        SubAgent(name="worker", description="Finds sources.", model=worker_model),
        SubAgent(name="reviewer", description="Checks sources.", model=reviewer_model),
    ]
    return create_deep_agent(
        model=main_model,
        middleware=[main_monitor],
        subagents=monitor_subagents(middleware=main_monitor, subagents=subagents),
    )


def stream_subgraph_custom_events(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
) -> tuple[list[dict[str, Any]], BaseException | None]:
    """Collect the custom events of every graph in a run, and the error it ended with, if any."""
    payload = build_task_input()
    events: list[dict[str, Any]] = []
    try:
        if mode == "invoke":
            parts = agent.stream(payload, stream_mode="custom", subgraphs=True)
            events.extend(event for _namespace, event in parts)
        else:

            async def collect() -> None:
                parts = agent.astream(payload, stream_mode="custom", subgraphs=True)
                async for _namespace, event in parts:
                    events.append(event)

            asyncio.run(collect())
    except Exception as error:
        return events, error
    return events, None


def summarise(log: list[StepRecord]) -> list[tuple[str, str, int]]:
    return [(record["agent"], record["outcome"], record["blocked_count"]) for record in log]


@pytest.fixture
def http_tools() -> list[BaseTool]:
    return [tool for tool in Workspace().build_tools() if tool.name == "http_post"]


def test_repeated_delegations_halt_at_exactly_the_thread_total(run_mode: RunMode) -> None:
    # Arrange
    delegations = 11
    main_model = ScriptedChatModel(
        responses=[
            *(build_delegation_step(call_id=f"call-task-{index}") for index in range(delegations)),
            AIMessage("All parts done."),
        ],
    )
    worker_responses: list[AIMessage] = []
    for index in range(delegations):
        worker_responses += [
            build_exfiltration_step(call_id=f"call-post-{index}-a"),
            build_exfiltration_step(call_id=f"call-post-{index}-b"),
            AIMessage(f"Part {index} done."),
        ]
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=ScriptedChatModel(responses=worker_responses),
        main_monitor=build_auto_mode_monitor(),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    log = result["monitor_log"]
    assert sum(record["blocked_count"] for record in log) == 20
    worker_rows = [row for row in summarise(log) if row[0] == "worker"]
    assert worker_rows == [("worker", "steered", 2)] * 9 + [("worker", "halted", 2)]
    assert summarise(log)[-1] == ("main", "halted", 0)
    assert log[-1]["samples"] == []
    assert len(main_model.calls) == 10


def test_blocks_made_before_a_delegation_count_inside_it(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            build_delegation_step(),
            AIMessage(MAIN_ANSWER),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            build_exfiltration_step(call_id="call-post-2"),
            AIMessage("Three sources found."),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_auto_mode_monitor(max_total_blocks=3),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert summarise(result["monitor_log"]) == [
        ("main", "steered", 1),
        ("worker", "halted", 2),
        ("main", "halted", 0),
    ]
    assert len(worker_model.calls) == 2


def test_each_delegation_names_its_records_with_its_own_id(
    run_mode: RunMode,
    http_tools: list[BaseTool],
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            build_delegation_step(call_id="call-task-2"),
            AIMessage(MAIN_ANSWER),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_benign_post(call_id="call-post-1"),
            AIMessage("First report."),
            build_benign_post(call_id="call-post-2"),
            AIMessage("Second report."),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_auto_mode_monitor(),
        tools=http_tools,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    steps = [
        (record["agent"], record.get("delegation_id"), record["step_number"])
        for record in result["monitor_log"]
    ]
    assert steps == [
        ("main", None, 1),
        ("worker", "call-task-1", 1),
        ("worker", "call-task-1", 2),
        ("main", None, 2),
        ("worker", "call-task-2", 1),
        ("worker", "call-task-2", 2),
        ("main", None, 3),
    ]
    assert all(
        "delegation_id" not in record
        for record in result["monitor_log"]
        if record["agent"] == "main"
    )
    assert "monitor_delegation" not in result


def test_parallel_subagents_count_each_block_once(run_mode: RunMode) -> None:
    # Arrange
    parallel_delegation = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"description": "Find sources.", "subagent_type": "worker"},
                "id": "call-task-worker",
                "type": "tool_call",
            },
            {
                "name": "task",
                "args": {"description": "Check sources.", "subagent_type": "reviewer"},
                "id": "call-task-reviewer",
                "type": "tool_call",
            },
        ],
    )
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            parallel_delegation,
            build_delegation_step(call_id="call-task-again"),
            AIMessage(MAIN_ANSWER),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-worker-1"),
            AIMessage("Sources found."),
            build_exfiltration_step(call_id="call-post-worker-2"),
            build_exfiltration_step(call_id="call-post-worker-3"),
            AIMessage("More sources found."),
        ],
    )
    reviewer_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-reviewer"),
            AIMessage("Sources checked."),
        ],
    )
    agent = build_two_subagent_agent(
        main_model=main_model,
        worker_model=worker_model,
        reviewer_model=reviewer_model,
        main_monitor=build_auto_mode_monitor(max_total_blocks=5),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    rows = summarise(result["monitor_log"])
    assert rows[0] == ("main", "steered", 1)
    assert sorted(rows[1:3]) == [("reviewer", "steered", 1), ("worker", "steered", 1)]
    assert rows[3:] == [
        ("main", "allowed", 0),
        ("worker", "halted", 2),
        ("main", "halted", 0),
    ]
    assert result["monitor_log"][4]["delegation_id"] == "call-task-again"
    assert "monitor_delegation" not in result


def test_a_thread_s_second_turn_keeps_the_subagent_blocks_of_the_first(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            AIMessage(MAIN_ANSWER),
            build_delegation_step(call_id="call-task-2"),
            AIMessage("Second answer."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            AIMessage("First report."),
            build_exfiltration_step(call_id="call-post-2"),
            AIMessage("Second report."),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_auto_mode_monitor(max_total_blocks=2),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"budget-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    assert summarise(result["monitor_log"]) == [
        ("main", "allowed", 0),
        ("worker", "steered", 1),
        ("main", "allowed", 0),
        ("main", "allowed", 0),
        ("worker", "halted", 1),
        ("main", "halted", 0),
    ]
    assert "monitor_delegation" not in agent.get_state(config).values


def test_a_later_turn_samples_its_first_step_after_subagents_reached_the_total(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            AIMessage(MAIN_ANSWER),
            build_delegation_step(call_id="call-task-2"),
            AIMessage("Third answer."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            AIMessage("First report."),
            build_exfiltration_step(call_id="call-post-2"),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_auto_mode_monitor(max_total_blocks=2),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"third-turn-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    assert summarise(result["monitor_log"])[-3:] == [
        ("worker", "halted", 1),
        ("main", "halted", 0),
        ("main", "allowed", 0),
    ]
    assert len(result["monitor_log"][-1]["samples"]) == 1
    assert read_texts(result["messages"])[-1] == "Third answer."
    assert len(main_model.calls) == 4


def test_a_subagent_halt_under_the_total_leaves_the_parent_running(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    worker_model = ScriptedChatModel(
        responses=[build_exfiltration_step(call_id=f"call-post-{index}") for index in range(3)],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_auto_mode_monitor(),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert summarise(result["monitor_log"]) == [
        ("main", "allowed", 0),
        ("worker", "halted", 3),
        ("main", "allowed", 0),
    ]
    assert read_texts(result["messages"])[-1] == MAIN_ANSWER


def test_stacked_monitors_hand_a_subagent_the_blocks_once(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            build_delegation_step(),
            AIMessage(MAIN_ANSWER),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            build_exfiltration_step(call_id="call-post-2"),
            AIMessage("Three sources found."),
        ],
    )
    outer_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AcceptFirst(),
        label="outer",
    )
    inner_monitor = build_auto_mode_monitor(max_total_blocks=3)
    agent = create_deep_agent(
        model=main_model,
        middleware=[outer_monitor, inner_monitor],
        subagents=monitor_subagents(
            middleware=inner_monitor,
            subagents=[SubAgent(name="worker", description="Finds.", model=worker_model)],
        ),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    worker_rows = [row for row in summarise(result["monitor_log"]) if row[0] == "worker"]
    assert worker_rows == [("worker", "halted", 2)]


def test_a_failed_subagent_step_names_its_delegation(run_mode: RunMode) -> None:
    # Arrange
    worker_monitor = MonitorMiddleware(
        monitor=TriggeredFailureMonitor(inner=build_keyword_monitor(), trigger="Three sources"),
        protocol=AcceptFirst(),
    )
    agent = build_deep_agent(
        main_model=ScriptedChatModel(
            responses=[build_delegation_step(call_id="call-task-7"), AIMessage(MAIN_ANSWER)],
        ),
        worker_model=ScriptedChatModel(responses=[AIMessage("Three sources found.")]),
        main_monitor=build_auto_mode_monitor(),
        worker_monitor=worker_monitor,
    )

    # Act
    events, error = stream_subgraph_custom_events(agent, mode=run_mode)

    # Assert
    assert isinstance(error, TimeoutError)
    [failed] = [event for event in events if event["type"] == "monitor_step_failed"]
    assert (failed["agent"], failed["step_number"]) == ("worker", 1)
    assert failed["delegation_id"] == "call-task-7"
