"""A subagent's trajectory joins its parent's run under the call that started it.

Each subagent here is compiled with `create_agent` and no system prompt, so
its trajectory holds the instruction the call gave, written by the
delegator, then exactly the events its own messages read as. A subagent's
monitor of a label the parent has no trajectory for returns a trajectory
that is dropped with a warning.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from deepagents import CompiledSubAgent, create_deep_agent
from hyoshigi import (
    Action,
    Author,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.contracts import SubagentHalt, TaskAuthor
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_read_step,
    build_thread_config,
    read_texts,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step, build_parallel_delegation
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst

type Agent = CompiledStateGraph[Any, Any, Any, Any]

LABEL = "monitor"
READ_ACTION = Action(
    id="step-1",
    tool_calls=(ToolCall(id="call-read", name="read_file", arguments='{"path": "q3.md"}'),),
)
READ_RESULT = ToolObservation(call_id="call-read", parts=(TextPart(text="contents of q3.md"),))


def build_main_monitor() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


def build_subagent_monitor(agent_name: str, *, label: str = LABEL) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=AcceptFirst(),
        label=label,
        agent_name=agent_name,
        task_author=TaskAuthor.PARENT_AGENT,
    )


def build_reader(name: str, *, answer: str, monitor: MonitorMiddleware) -> CompiledSubAgent:
    """Build a compiled subagent that reads q3.md as `step-1`, then answers as `step-2`."""
    model = ScriptedChatModel(
        responses=[
            build_read_step().model_copy(update={"id": "step-1"}),
            AIMessage(answer, id="step-2"),
        ],
    )
    runnable = create_agent(model, tools=Workspace().build_tools(), middleware=[monitor])
    return CompiledSubAgent(name=name, description=f"The {name}.", runnable=runnable)


def read_run(agent: Agent, *, thread_id: str) -> Trajectory:
    """Return the main agent's trajectory as the checkpoint holds it, empty before any write."""
    values = agent.get_state(build_thread_config(thread_id)).values
    stored = values.get("monitor_trajectory", {}).get(LABEL, {})
    return Trajectory.load(stored.get("trajectory", Trajectory().dump()))


@pytest.mark.xfail(strict=True, reason="a subagent's trajectory joins its parent's run (#146)")
def test_a_nested_subagent_s_steps_sit_under_both_calls_in_the_root_s_trajectory(
    run_mode: RunMode,
) -> None:
    # Arrange: main delegates to middle, a deep agent that delegates to inner
    inner = build_reader("inner", answer="Inner done.", monitor=build_subagent_monitor("inner"))
    middle = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(
                    description="Check the sources.",
                    call_id="call-task-middle",
                    subagent_type="inner",
                ),
                AIMessage("Middle done."),
            ],
        ),
        middleware=[build_subagent_monitor("middle")],
        subagents=[inner],
    )
    agent = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(call_id="call-task-main", subagent_type="middle"),
                AIMessage("Done."),
            ],
        ),
        middleware=[build_main_monitor()],
        subagents=[CompiledSubAgent(name="middle", description="Delegates.", runnable=middle)],
        checkpointer=InMemorySaver(),
    )
    thread_id = f"nested-run-{run_mode}"

    # Act
    run_agent(agent, mode=run_mode, config=build_thread_config(thread_id))

    # Assert
    middle_run = read_run(agent, thread_id=thread_id).subagents.get("call-task-main", Trajectory())
    assert middle_run.subagents.get("call-task-middle", Trajectory()).history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text="Check the sources."),)),
        READ_ACTION,
        READ_RESULT,
        Action(id="step-2", parts=(TextPart(text="Inner done."),)),
    )


@pytest.mark.xfail(strict=True, reason="a subagent's trajectory joins its parent's run (#146)")
def test_parallel_subagents_with_the_same_ids_join_apart_under_their_own_calls(
    run_mode: RunMode,
) -> None:
    # Arrange: both subagents name their steps and their read alike
    agent = create_deep_agent(
        model=ScriptedChatModel(responses=[build_parallel_delegation(), AIMessage("Done.")]),
        middleware=[build_main_monitor()],
        subagents=[
            build_reader(
                "worker", answer="Three sources.", monitor=build_subagent_monitor("worker")
            ),
            build_reader(
                "reviewer",
                answer="Sources check out.",
                monitor=build_subagent_monitor("reviewer"),
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    thread_id = f"parallel-run-{run_mode}"

    # Act
    run_agent(agent, mode=run_mode, config=build_thread_config(thread_id))

    # Assert
    subagents = read_run(agent, thread_id=thread_id).subagents
    assert subagents.get("call-task-worker", Trajectory()).history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text="Find sources."),)),
        READ_ACTION,
        READ_RESULT,
        Action(id="step-2", parts=(TextPart(text="Three sources."),)),
    )
    assert subagents.get("call-task-reviewer", Trajectory()).history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text="Check sources."),)),
        READ_ACTION,
        READ_RESULT,
        Action(id="step-2", parts=(TextPart(text="Sources check out."),)),
    )


@pytest.mark.xfail(strict=True, reason="a subagent's trajectory joins its parent's run (#146)")
def test_a_trajectory_under_a_label_the_parent_lacks_is_dropped_with_a_warning_alone(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the worker's monitor has a label of its own, and a halt would stop the run
    worker = build_reader(
        "worker",
        answer="Three sources.",
        monitor=build_subagent_monitor("worker", label="worker-monitor"),
    )
    main_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    agent = create_deep_agent(
        model=ScriptedChatModel(
            responses=[build_delegation_step(subagent_type="worker"), AIMessage("Done.")],
        ),
        middleware=[main_monitor],
        subagents=[worker],
    )

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.returned_records"):
        result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_texts(result["messages"])[-1] == "Done."
    assert (
        "a trajectory under the monitor label 'worker-monitor', which the agent 'main' has no "
        "trajectory for"
    ) in caplog.text
