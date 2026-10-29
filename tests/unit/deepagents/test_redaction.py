"""A subagent's trajectory is redacted before it joins its parent's, as the parent's is.

The subagent's monitor takes the redaction of the monitor it was copied from or
given, and everything it keeps under the call that started it, the instruction
included, holds the redacted text when the parent's channel joins it.
"""

from __future__ import annotations

import json
from typing import Any

from deepagents import CompiledSubAgent, create_deep_agent
from hyoshigi import Action, Author, Observation, TextPart, ToolObservation, Trajectory
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors import Channel, Redaction, View
from langchain_sync_monitors.contracts import SampleRecord, TaskAuthor
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, read_stored_trajectory, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor, RenderingMonitor
from tests.support.protocols import RUN_EVERY_STEP
from tests.support.redaction import (
    LAST_REPLY_WITH_SECRET,
    MASK,
    MASK_SECRET,
    MASKED_FIRST_REPLY,
    MASKED_LAST_REPLY,
    MASKED_READ_CALL,
    MASKED_READ_RESULT,
    SECRET,
    SUSPICION_BY_SECRET,
    TASK_WITH_SECRET,
    build_secret_steps,
    build_secret_tools,
)

type Agent = CompiledStateGraph[Any, Any, Any, Any]

INSTRUCTION_WITH_SECRET = f"Find the sources, key {SECRET}."


def build_monitor(
    agent_name: str,
    *,
    task_author: TaskAuthor,
    redaction: Redaction | None = MASK_SECRET,
) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET),
        protocol=RUN_EVERY_STEP,
        agent_name=agent_name,
        task_author=task_author,
        redaction=redaction,
    )


def build_agent_with_worker(*, worker_redaction: Redaction | None) -> Agent:
    """Build a monitored main agent that delegates to a compiled, monitored worker.

    The instruction, the worker's tool result and its replies all hold the token.
    """
    worker = CompiledSubAgent(
        name="worker",
        description="Finds sources.",
        runnable=create_agent(
            ScriptedChatModel(responses=build_secret_steps()),
            tools=build_secret_tools(),
            middleware=[
                build_monitor(
                    "worker",
                    task_author=TaskAuthor.PARENT_AGENT,
                    redaction=worker_redaction,
                ),
            ],
        ),
    )
    return create_deep_agent(
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


MASKED_WORKER_HISTORY = (
    Observation(
        author=Author.DELEGATOR,
        parts=(TextPart(text=f"Find the sources, key {MASK}."),),
    ),
    Action(id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(MASKED_READ_CALL,)),
    ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
    Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
)
"""The worker's trajectory as the mask leaves it."""

MASKED_REASON = f"mentions {MASK}"
MASKED_WORKER_SAMPLES = [
    [
        SampleRecord(
            suspicion=0.9,
            reason=MASKED_REASON,
            proposal={
                "kind": "action",
                "id": "step-1",
                "thought": [],
                "parts": [{"kind": "text", "text": MASKED_FIRST_REPLY}],
                "tool_calls": [
                    {
                        "id": "call-read",
                        "name": "read_file",
                        "arguments": MASKED_READ_CALL.arguments,
                    },
                ],
            },
            executed=True,
        ),
    ],
    [
        SampleRecord(
            suspicion=0.9,
            reason=MASKED_REASON,
            proposal={
                "kind": "action",
                "id": "step-2",
                "thought": [],
                "parts": [{"kind": "text", "text": MASKED_LAST_REPLY}],
                "tool_calls": [],
            },
            executed=True,
        ),
    ],
]
"""The samples of the worker's two records as the mask leaves them."""


def test_a_subagent_s_joined_trajectory_holds_what_the_redaction_leaves(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent_with_worker(worker_redaction=MASK_SECRET)

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    assert Trajectory.load(stored).subagents["call-task"].history == MASKED_WORKER_HISTORY


def test_a_parent_redacts_the_trajectory_of_a_subagent_whose_monitor_has_no_redaction(
    run_mode: RunMode,
) -> None:
    # Arrange: the worker's own monitor keeps its trajectory as it read it
    agent = build_agent_with_worker(worker_redaction=None)

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    assert Trajectory.load(stored).subagents["call-task"].history == MASKED_WORKER_HISTORY


def test_a_parent_redacts_the_records_of_a_subagent_whose_monitor_has_no_redaction(
    run_mode: RunMode,
) -> None:
    # Arrange: the worker's own monitor keeps its records as it rated them
    agent = build_agent_with_worker(worker_redaction=None)

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    log = result["monitor_log"]
    assert SECRET not in json.dumps(log)
    worker_records = [record for record in log if record["agent"] == "worker"]
    assert [record["samples"] for record in worker_records] == MASKED_WORKER_SAMPLES


def test_a_subagent_s_monitor_is_asked_redacted_content(run_mode: RunMode) -> None:
    # Arrange: monitor_subagents copies the main monitor for the worker, rendering model and all
    view = View(channels=Channel.DEFAULT | Channel.TOOL_OBSERVATIONS | Channel.AGENT_PARTS)
    monitor = RenderingMonitor(view=view)
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
        main_monitor=MonitorMiddleware(
            monitor=monitor,
            protocol=RUN_EVERY_STEP,
            redaction=MASK_SECRET,
        ),
    )

    # Act
    run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert: the main agent rates twice and the worker once
    assert len(monitor.readings) == 3
    read = "\n".join(f"{each.trajectory}\n{each.proposed_action}" for each in monitor.readings)
    assert SECRET not in read
    assert f"Find the sources, key {MASK}." in read
    assert MASKED_LAST_REPLY in read


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
        Action(
            id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(MASKED_READ_CALL,)
        ),
        ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
        Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
    )
