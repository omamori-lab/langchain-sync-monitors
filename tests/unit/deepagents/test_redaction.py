"""A subagent's trajectory is redacted before it joins its parent's, as the parent's is.

The subagent's monitor takes the redaction of the monitor it was copied from or
given, and everything it keeps under the call that started it, the instruction
included, holds the redacted text when the parent's channel joins it.
"""

from __future__ import annotations

import json

import pytest
from deepagents import CompiledSubAgent, create_deep_agent
from hyoshigi import Action, Author, Observation, TextPart, ToolCall, ToolObservation, Trajectory
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, read_stored_trajectory, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import RUN_EVERY_STEP
from tests.support.redaction import (
    LAST_REPLY_WITH_SECRET,
    MASK,
    MASK_SECRET,
    MASKED_FIRST_REPLY,
    MASKED_LAST_REPLY,
    MASKED_READ_RESULT,
    SECRET,
    TASK_WITH_SECRET,
    build_secret_steps,
    build_secret_tools,
)

INSTRUCTION_WITH_SECRET = f"Find the sources, key {SECRET}."
READ_CALL = ToolCall(id="call-read", name="read_file", arguments='{"path": "q3.md"}')


def build_monitor(agent_name: str, *, task_author: TaskAuthor) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=RUN_EVERY_STEP,
        agent_name=agent_name,
        task_author=task_author,
        redaction=MASK_SECRET,
    )


@pytest.mark.xfail(strict=True, reason="PR 8 redacts the trajectory at entry")
def test_a_subagent_s_joined_trajectory_holds_what_the_redaction_leaves(run_mode: RunMode) -> None:
    # Arrange: the worker's instruction, tool result and replies all hold the token
    worker = CompiledSubAgent(
        name="worker",
        description="Finds sources.",
        runnable=create_agent(
            ScriptedChatModel(responses=build_secret_steps()),
            tools=build_secret_tools(),
            middleware=[build_monitor("worker", task_author=TaskAuthor.PARENT_AGENT)],
        ),
    )
    agent = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(
                    description=INSTRUCTION_WITH_SECRET,
                    call_id="call-task",
                    subagent_type="worker",
                ),
                AIMessage("Done."),
            ],
        ),
        middleware=[build_monitor("main", task_author=TaskAuthor.USER)],
        subagents=[worker],
    )

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    assert Trajectory.load(stored).subagents["call-task"].history == (
        Observation(
            author=Author.DELEGATOR,
            parts=(TextPart(text=f"Find the sources, key {MASK}."),),
        ),
        Action(id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(READ_CALL,)),
        ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
        Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
    )


@pytest.mark.xfail(strict=True, reason="PR 8 redacts the trajectory at entry")
def test_a_subagent_monitored_by_monitor_subagents_takes_the_redaction_of_its_parent(
    run_mode: RunMode,
) -> None:
    # Arrange
    agent = build_deep_agent(
        main_model=ScriptedChatModel(
            responses=[
                build_delegation_step(
                    description=INSTRUCTION_WITH_SECRET,
                    call_id="call-task",
                ),
                AIMessage("Done."),
            ],
        ),
        worker_model=ScriptedChatModel(responses=[AIMessage(LAST_REPLY_WITH_SECRET, id="step-1")]),
        main_monitor=build_monitor("main", task_author=TaskAuthor.USER),
    )

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    worker = Trajectory.load(stored).subagents["call-task"]
    assert worker.history[0] == Observation(
        author=Author.DELEGATOR,
        parts=(TextPart(text=f"Find the sources, key {MASK}."),),
    )
    assert worker.history[-1] == Action(id="step-1", parts=(TextPart(text=MASKED_LAST_REPLY),))


@pytest.mark.xfail(strict=True, reason="PR 8 redacts the trajectory at entry")
def test_a_nested_subagent_s_joined_trajectory_holds_what_the_redaction_leaves(
    run_mode: RunMode,
) -> None:
    # Arrange: main delegates to middle, a deep agent that delegates to inner, all one redaction
    main_monitor = build_monitor("main", task_author=TaskAuthor.USER)
    inner = CompiledSubAgent(
        name="inner",
        description="Reads the file.",
        runnable=create_agent(
            ScriptedChatModel(responses=build_secret_steps()),
            tools=build_secret_tools(),
            middleware=[main_monitor.copy_for_subagent(subagent_name="inner")],
        ),
    )
    middle = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(
                    description=INSTRUCTION_WITH_SECRET,
                    call_id="call-task-middle",
                    subagent_type="inner",
                ),
                AIMessage("Middle done."),
            ],
        ),
        middleware=[main_monitor.copy_for_subagent(subagent_name="middle")],
        subagents=[inner],
    )
    agent = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(call_id="call-task-main", subagent_type="middle"),
                AIMessage("Done."),
            ],
        ),
        middleware=[main_monitor],
        subagents=[CompiledSubAgent(name="middle", description="Delegates.", runnable=middle)],
    )

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    middle_run = Trajectory.load(stored).subagents["call-task-main"]
    assert middle_run.subagents["call-task-middle"].history == (
        Observation(
            author=Author.DELEGATOR,
            parts=(TextPart(text=f"Find the sources, key {MASK}."),),
        ),
        Action(id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(READ_CALL,)),
        ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
        Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
    )
