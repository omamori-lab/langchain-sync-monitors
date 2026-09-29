"""Subagents a monitor cannot watch correctly are refused, with advice that works."""

from __future__ import annotations

import pytest
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import SubagentHalt, TaskAuthor
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


def build_compiled_worker(*, worker_monitor: MonitorMiddleware | None = None) -> CompiledSubAgent:
    runnable = create_agent(
        ScriptedChatModel(responses=[AIMessage("Three sources found.")]),
        middleware=[worker_monitor] if worker_monitor else [],
    )
    return CompiledSubAgent(name="worker", description="Finds sources.", runnable=runnable)


def test_a_forked_subagent_is_refused(middleware: MonitorMiddleware) -> None:
    # Arrange
    fork = SubAgent(name="forker", description="Continues.", system_prompt="Go on.", mode="fork")

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"mode='fork'.*#35"):
        monitor_subagents(middleware=middleware, subagents=[fork])


def test_a_compiled_fork_is_refused_as_a_fork(middleware: MonitorMiddleware) -> None:
    # Arrange
    fork = build_compiled_worker()
    fork["mode"] = "fork"

    # Act / Assert
    with pytest.raises(ConfigurationError, match="mode='fork'"):
        monitor_subagents(middleware=middleware, subagents=[fork])


def test_an_isolated_subagent_is_still_monitored(middleware: MonitorMiddleware) -> None:
    # Arrange
    isolated = SubAgent(name="helper", description="Helps.", system_prompt="Help.", mode="isolated")

    # Act
    specs = monitor_subagents(middleware=middleware, subagents=[isolated])

    # Assert
    assert [spec["name"] for spec in specs] == ["helper", "general-purpose"]


def test_the_compiled_subagent_advice_names_the_agent_and_its_task_author(
    middleware: MonitorMiddleware,
) -> None:
    # Act
    with pytest.raises(ConfigurationError) as raised:
        monitor_subagents(middleware=middleware, subagents=[build_compiled_worker()])

    # Assert
    assert "agent_name='worker'" in str(raised.value)
    assert "task_author=TaskAuthor.PARENT_AGENT" in str(raised.value)


def test_following_the_compiled_subagent_advice_keeps_stop_whole_run_working(
    run_mode: RunMode,
) -> None:
    # Arrange: the worker's own graph carries the monitor the error message describes
    worker_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=HaltAfterOneSample(),
        agent_name="worker",
        task_author=TaskAuthor.PARENT_AGENT,
    )
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    main_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    agent = create_deep_agent(
        model=main_model,
        middleware=[main_monitor],
        subagents=[build_compiled_worker(worker_monitor=worker_monitor)],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    outcomes = [(record["agent"], record["outcome"]) for record in result["monitor_log"]]
    assert outcomes == [("main", "allowed"), ("worker", "halted"), ("main", "halted")]
    assert len(main_model.calls) == 1
    assert "halted the subagent worker" in result["messages"][-1].text
