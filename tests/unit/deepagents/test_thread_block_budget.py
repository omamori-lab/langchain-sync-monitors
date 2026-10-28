"""The run's block total covers its subagents in the order recorded, and names delegations.

Block and feed back counts the blocks of every trajectory in the run
[@hyoshigi2026]: a subagent counts its delegators' blocks, and the parent
counts each subagent's once its trajectory joins. At the total, the next
proposal is audited, and so halted, and the audit resets the total.
"""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import BlockAndFeedBack
from tests.support.agents import (
    RunMode,
    Workspace,
    build_benign_post,
    build_exfiltration_step,
    build_keyword_monitor,
    build_thread_config,
    read_texts,
    run_agent,
    stream_subgraph_custom_events,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import (
    build_deep_agent,
    build_delegation_step,
    build_parallel_delegation,
)
from tests.support.flaky_models import TriggeredFailureMonitor
from tests.support.protocols import RUN_EVERY_STEP

MAIN_ANSWER = "Here is the summary."


def build_blocking_monitor(*, blocks_in_total: int = 20) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.6, blocks_in_total=blocks_in_total),
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


def summarise(log: list[StepRecord]) -> list[tuple[str, str, int]]:
    return [(record["agent"], record["outcome"], record["blocked_count"]) for record in log]


@pytest.fixture
def http_tools() -> list[BaseTool]:
    return Workspace().build_http_tools()


def test_repeated_delegations_are_audited_at_exactly_the_run_s_total(run_mode: RunMode) -> None:
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
        main_monitor=build_blocking_monitor(),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the tenth worker's second block is the twentieth, so its report is audited and
    # halted; the audit resets the total, so the eleventh delegation blocks twice again
    log = result["monitor_log"]
    assert sum(record["blocked_count"] for record in log) == 22
    worker_rows = [row for row in summarise(log) if row[0] == "worker"]
    assert worker_rows == [
        *[("worker", "steered", 2)] * 9,
        ("worker", "halted", 2),
        ("worker", "steered", 2),
    ]
    assert summarise(log)[-1] == ("main", "allowed", 0)
    assert len(main_model.calls) == 12


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
        main_monitor=build_blocking_monitor(blocks_in_total=3),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the main agent's block and the worker's two reach the total of three, so the
    # worker's report is audited and halted, which resets the total
    assert summarise(result["monitor_log"]) == [
        ("main", "steered", 1),
        ("worker", "halted", 2),
        ("main", "allowed", 0),
    ]
    assert len(worker_model.calls) == 3


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
        main_monitor=build_blocking_monitor(),
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
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            build_parallel_delegation(),
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
        main_monitor=build_blocking_monitor(blocks_in_total=5),
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
        ("main", "allowed", 0),
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
        main_monitor=build_blocking_monitor(blocks_in_total=2),
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
        ("main", "allowed", 0),
    ]
    assert "monitor_delegation" not in agent.get_state(config).values


def test_a_subagent_halt_under_the_total_leaves_the_parent_running(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    worker_model = ScriptedChatModel(
        responses=[
            *(build_exfiltration_step(call_id=f"call-post-{index}") for index in range(3)),
            AIMessage("Report."),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_blocking_monitor(),
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


def test_a_failed_subagent_step_names_its_delegation(run_mode: RunMode) -> None:
    # Arrange
    worker_monitor = MonitorMiddleware(
        monitor=TriggeredFailureMonitor(inner=build_keyword_monitor(), trigger="Three sources"),
        protocol=RUN_EVERY_STEP,
    )
    agent = build_deep_agent(
        main_model=ScriptedChatModel(
            responses=[build_delegation_step(call_id="call-task-7"), AIMessage(MAIN_ANSWER)],
        ),
        worker_model=ScriptedChatModel(responses=[AIMessage("Three sources found.")]),
        main_monitor=build_blocking_monitor(),
        worker_monitor=worker_monitor,
    )

    # Act
    events, error = stream_subgraph_custom_events(agent, mode=run_mode)

    # Assert
    assert isinstance(error, TimeoutError)
    [failed] = [event for event in events if event["type"] == "monitor_step_failed"]
    assert (failed["agent"], failed["step_number"]) == ("worker", 1)
    assert failed["delegation_id"] == "call-task-7"
