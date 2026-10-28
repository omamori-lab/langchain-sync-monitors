"""A Deep Agents `task` call starts a subagent's trajectory under the call, in the parent's.

The subagent's own steps stay in its own graph state, so the parent's
trajectory holds the instruction the call gave, written by the delegator.
"""

from __future__ import annotations

import pytest
from hyoshigi import Author, Observation, TextPart, Trajectory
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, build_thread_config, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst

INSTRUCTION = "Find the sources for q3.md."


@pytest.mark.xfail(strict=True)
def test_a_task_call_starts_a_subagent_s_trajectory_under_the_call(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(description=INSTRUCTION), AIMessage("Here it is.")],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=ScriptedChatModel(responses=[AIMessage("Three sources found.")]),
        main_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst()),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"delegated-{run_mode}")

    # Act
    run_agent(agent, mode=run_mode, config=config)

    # Assert
    stored = agent.get_state(config).values.get("monitor_trajectory", {}).get("monitor", {})
    trajectory = Trajectory.load(stored.get("trajectory", Trajectory().dump()))
    subagent = trajectory.subagents.get("call-task", Trajectory())
    assert subagent.history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text=INSTRUCTION),)),
    )
