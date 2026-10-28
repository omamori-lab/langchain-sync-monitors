"""A deep agent and its subagent are both monitored, and a subagent halt can stop the run."""

from __future__ import annotations

import pytest
from hyoshigi import Author
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors import DEFAULT_HALT_MESSAGE
from langchain_sync_monitors.contracts import SubagentHalt
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    build_task_input,
    build_thread_config,
    read_texts,
    run_agent,
    run_stopping_audits,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import HALT_EVERY_STEP, RUN_EVERY_STEP

WORKER_REPORT = "Three sources found."
MAIN_ANSWER = "Here is the summary."


@pytest.fixture
def monitor() -> KeywordMonitor:
    return KeywordMonitor()


@pytest.fixture
def halting_worker_monitor(monitor: KeywordMonitor) -> MonitorMiddleware:
    return MonitorMiddleware(monitor=monitor, protocol=HALT_EVERY_STEP)


def test_the_main_agent_and_its_subagent_both_record_their_steps(
    run_mode: RunMode,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    worker_model = ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)])
    main_monitor = MonitorMiddleware(monitor=monitor, protocol=RUN_EVERY_STEP)
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=main_monitor,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_texts(result["messages"])[-1] == MAIN_ANSWER
    log = result["monitor_log"]
    assert [(record["agent"], record["step_number"]) for record in log] == [
        ("main", 1),
        ("worker", 1),
        ("main", 2),
    ]
    authors = [rating.task_author for rating in monitor.ratings]
    assert authors == [Author.USER, Author.DELEGATOR, Author.USER]


def test_a_subagent_halt_stops_the_whole_run_when_asked(
    run_mode: RunMode,
    monitor: KeywordMonitor,
    halting_worker_monitor: MonitorMiddleware,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    main_monitor = MonitorMiddleware(
        monitor=monitor,
        protocol=RUN_EVERY_STEP,
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)]),
        main_monitor=main_monitor,
        worker_monitor=halting_worker_monitor,
        checkpointer=InMemorySaver(),
    )

    # Act
    result = run_stopping_audits(
        agent,
        build_task_input(),
        mode=run_mode,
        config=build_thread_config("whole-run"),
    )

    # Assert
    assert len(main_model.calls) == 1
    assert "halted the subagent worker" in result["messages"][-1].text
    outcomes = [(record["agent"], record["outcome"]) for record in result["monitor_log"]]
    assert outcomes == [("main", "allowed"), ("worker", "halted"), ("main", "halted")]


def test_a_subagent_halt_only_ends_the_subagent_by_default(
    run_mode: RunMode,
    monitor: KeywordMonitor,
    halting_worker_monitor: MonitorMiddleware,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)]),
        main_monitor=MonitorMiddleware(monitor=monitor, protocol=RUN_EVERY_STEP),
        worker_monitor=halting_worker_monitor,
        checkpointer=InMemorySaver(),
    )

    # Act
    result = run_stopping_audits(
        agent,
        build_task_input(),
        mode=run_mode,
        config=build_thread_config("subagent-only"),
    )

    # Assert
    assert read_texts(result["messages"])[-1] == MAIN_ANSWER
    reports = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [report.text for report in reports] == [DEFAULT_HALT_MESSAGE]
    assert [record["outcome"] for record in result["monitor_log"]] == [
        "allowed",
        "halted",
        "allowed",
    ]


def test_subagent_records_reach_the_checkpointed_parent_state(
    run_mode: RunMode,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    agent = build_deep_agent(
        main_model=ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)]),
        worker_model=ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)]),
        main_monitor=MonitorMiddleware(monitor=monitor, protocol=RUN_EVERY_STEP),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"deep-{run_mode}")

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    stored = agent.get_state(config).values["monitor_log"]
    assert stored == result["monitor_log"]
    assert [record["agent"] for record in stored] == ["main", "worker", "main"]
