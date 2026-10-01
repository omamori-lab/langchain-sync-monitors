"""A subagent keeps its delegation across an interrupt and counts under its monitor's label.

A parent counts none of the blocks a subagent's monitor records under another
label, made or returned. A subagent without a monitor runs without a delegation.
"""

from __future__ import annotations

from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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
    read_texts,
    run_agent,
    run_payload,
    summarise_records,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step


def build_auto_mode_monitor() -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=0.6),
    )


def test_a_subagent_keeps_its_delegation_when_resumed_after_an_interrupt(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(call_id="call-task-1"), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post"),
            build_benign_post(call_id="call-benign"),
            AIMessage("Report."),
        ],
    )
    main_monitor = build_auto_mode_monitor()
    worker = SubAgent(
        name="worker",
        description="Finds sources.",
        model=worker_model,
        interrupt_on={"http_post": True},
    )
    agent = create_deep_agent(
        model=main_model,
        tools=Workspace().build_http_tools(),
        middleware=[main_monitor],
        subagents=monitor_subagents(middleware=main_monitor, subagents=[worker]),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"interrupt-{run_mode}")
    interrupted = run_payload(agent, build_task_input(), mode=run_mode, config=config)
    assert "__interrupt__" in interrupted

    # Act
    approval = Command(resume={"decisions": [{"type": "approve"}]})
    result = run_payload(agent, approval, mode=run_mode, config=config)

    # Assert
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task-1"),
        ("worker", "allowed", 0, "call-task-1"),
        ("main", "allowed", 0, None),
    ]
    assert "monitor_delegation" not in result
    assert "monitor_delegation" not in agent.get_state(config).values


def test_an_unmonitored_subagent_runs_under_a_monitored_parent(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(call_id="call-task-1"), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(responses=[AIMessage("Report.")])
    agent = create_deep_agent(
        model=main_model,
        middleware=[build_auto_mode_monitor()],
        subagents=[SubAgent(name="worker", description="Finds sources.", model=worker_model)],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("main", "allowed", 0, None),
    ]
    assert read_texts(result["messages"])[-1] == "Done."
    assert "monitor_delegation" not in result


def test_a_subagent_monitor_with_a_label_of_its_own_counts_apart(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main-1"),
            build_exfiltration_step(call_id="call-post-main-2"),
            build_delegation_step(call_id="call-task-1"),
            AIMessage("Done."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-worker"),
            AIMessage("Report."),
        ],
    )
    worker_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=0.6, max_total_blocks=3),
        label="worker-monitor",
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=AutoMode(block_threshold=0.6, max_total_blocks=3),
        ),
        worker_monitor=worker_monitor,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the worker's first block would reach a shared total of three.
    rows = [
        (record["agent"], record["monitor"], record["outcome"], record["blocked_count"])
        for record in result["monitor_log"]
    ]
    assert rows == [
        ("main", "monitor", "steered", 2),
        ("worker", "worker-monitor", "steered", 1),
        ("main", "monitor", "allowed", 0),
    ]


def test_a_parent_counts_apart_from_a_subagent_monitor_with_a_label_of_its_own(
    run_mode: RunMode,
) -> None:
    # Arrange: the worker blocks twice under its own label, and the parent's total is two
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            build_exfiltration_step(call_id="call-post-main"),
            AIMessage("Done."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-worker-1"),
            build_exfiltration_step(call_id="call-post-worker-2"),
            AIMessage("Report."),
        ],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=AutoMode(block_threshold=0.6, max_total_blocks=2),
        ),
        worker_monitor=MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=AutoMode(block_threshold=0.6, max_total_blocks=3),
            label="worker-monitor",
        ),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the parent's own first block is its first, so it is steered, not halted
    rows = [
        (record["agent"], record["monitor"], record["outcome"], record["blocked_count"])
        for record in result["monitor_log"]
    ]
    assert rows == [
        ("main", "monitor", "allowed", 0),
        ("worker", "worker-monitor", "steered", 2),
        ("main", "monitor", "steered", 1),
    ]


def test_a_parent_at_its_total_still_samples_after_blocks_returned_under_another_label(
    run_mode: RunMode,
) -> None:
    # Arrange: the parent reached its total of one on the first turn, and a new turn lifts
    # its halt
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            build_delegation_step(call_id="call-task-1"),
            AIMessage("Done."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[build_exfiltration_step(call_id="call-post-worker"), AIMessage("Report.")],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=AutoMode(block_threshold=0.6, max_total_blocks=1),
        ),
        worker_monitor=MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=AutoMode(block_threshold=0.6, max_total_blocks=3),
            label="worker-monitor",
        ),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"returned-by-label-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert: the worker's block is not the parent monitor's, so its next step is sampled
    rows = [
        (record["agent"], record["monitor"], record["outcome"], len(record["samples"]))
        for record in result["monitor_log"]
    ]
    assert rows == [
        ("main", "monitor", "halted", 1),
        ("main", "monitor", "allowed", 1),
        ("worker", "worker-monitor", "steered", 2),
        ("main", "monitor", "allowed", 1),
    ]
