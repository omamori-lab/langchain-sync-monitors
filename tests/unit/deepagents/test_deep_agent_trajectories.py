"""A Deep Agent's monitor keeps its trajectory as any agent's does, and starts its subagents'.

Deep Agents keeps a run's input given as a string without an id, which the
monitor gives one at the start of the run. It keeps a message a hook writes
as a string without an id too, which the monitor gives one at the end of the
run, and the trajectory reads it once, not again under its new id. A `task`
call starts a subagent's trajectory under the call, in the parent's: the
subagent's own steps stay in its own graph state, so the parent's trajectory
holds the instruction the call gave, written by the delegator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from deepagents import create_deep_agent
from hyoshigi import (
    Action,
    Author,
    Event,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    TASK,
    RunMode,
    Workspace,
    build_read_step,
    build_thread_config,
    run_agent,
    run_messages,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst
from tests.support.written_human_messages import (
    NoteAfterAnswerMiddleware,
    NoteBeforeModelMiddleware,
)

INSTRUCTION = "Find the sources for q3.md."
SYSTEM_PROMPT = "You summarise files for the team."
NOTE = "Remember to cite every source you read."
NEXT_TURN = "Go on."
THIRD_TURN = "Now post the summary."
NOTE_ID = "note-with-an-id"

TASK_OBSERVATION = Observation(author=Author.USER, parts=(TextPart(text=TASK),))
NEXT_OBSERVATION = Observation(author=Author.USER, parts=(TextPart(text=NEXT_TURN),))
THIRD_OBSERVATION = Observation(author=Author.USER, parts=(TextPart(text=THIRD_TURN),))
SYSTEM_OBSERVATION = Observation(
    author=Author.SCAFFOLD,
    source="system",
    parts=(TextPart(text=SYSTEM_PROMPT),),
)
NOTE_OBSERVATION = Observation(
    author=Author.SCAFFOLD,
    source="application",
    parts=(TextPart(text=NOTE),),
)
FIRST_ANSWER = Action(id="step-1", parts=(TextPart(text="First."),))
SECOND_ANSWER = Action(id="step-2", parts=(TextPart(text="Second."),))
THIRD_ANSWER = Action(id="step-3", parts=(TextPart(text="Third."),))

type Agent = CompiledStateGraph[Any, Any, Any, Any]


def build_answers(*texts: str) -> ScriptedChatModel:
    """Script one final answer per text, as `step-1`, `step-2` and on."""
    replies = [AIMessage(text, id=f"step-{number}") for number, text in enumerate(texts, start=1)]
    return ScriptedChatModel(responses=replies)


def build_noted_agent(
    model: ScriptedChatModel,
    *,
    writers: Sequence[AgentMiddleware[Any, Any, Any]],
    labels: Sequence[str] = ("monitor",),
    tools: Sequence[BaseTool] = (),
) -> Agent:
    """Build a checkpointed Deep Agent with the writers, then one monitor for each label."""
    monitors: list[AgentMiddleware[Any, Any, Any]] = [
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst(), label=label)
        for label in labels
    ]
    return create_deep_agent(
        model=model,
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        middleware=[*writers, *monitors],
        checkpointer=InMemorySaver(),
    )


def read_history(
    agent: Agent, config: RunnableConfig, *, label: str = "monitor"
) -> tuple[Event, ...]:
    """Return the history of the label's trajectory, as the checkpoint holds it."""
    stored = agent.get_state(config).values.get("monitor_trajectory", {}).get(label, {})
    return Trajectory.load(stored.get("trajectory", Trajectory().dump())).history


def test_a_string_input_enters_the_trajectory_once_as_the_user_s(run_mode: RunMode) -> None:
    # Arrange
    agent = build_deep_agent(
        main_model=ScriptedChatModel(responses=[AIMessage("Done.")]),
        worker_model=ScriptedChatModel(responses=[]),
        main_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst()),
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"string-input-{run_mode}")

    # Act
    run_messages(agent, TASK, mode=run_mode, config=config)

    # Assert
    stored = agent.get_state(config).values.get("monitor_trajectory", {}).get("monitor", {})
    history = Trajectory.load(stored.get("trajectory", Trajectory().dump())).history
    task = Observation(author=Author.USER, parts=(TextPart(text=TASK),))
    assert history[:1] == (task,)
    assert history.count(task) == 1


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


@pytest.mark.xfail(strict=True)
def test_a_message_a_hook_writes_without_an_id_enters_the_trajectory_once_across_runs(
    run_mode: RunMode,
) -> None:
    # Arrange: the first run's step reads the note before the run's end gives it an id
    agent = build_noted_agent(
        build_answers("First.", "Second."),
        writers=[NoteBeforeModelMiddleware(text=NOTE)],
    )
    config = build_thread_config(f"hook-note-{run_mode}")
    run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)

    # Act
    run_messages(agent, [HumanMessage(NEXT_TURN)], mode=run_mode, config=config)

    # Assert
    assert read_history(agent, config) == (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        NOTE_OBSERVATION,
        FIRST_ANSWER,
        NEXT_OBSERVATION,
        SECOND_ANSWER,
    )


@pytest.mark.xfail(strict=True)
def test_every_stacked_monitor_reads_a_hook_written_message_once_across_runs(
    run_mode: RunMode,
) -> None:
    # Arrange: the run's end gives the id once, and both monitors read the note before it
    agent = build_noted_agent(
        build_answers("First.", "Second."),
        writers=[NoteBeforeModelMiddleware(text=NOTE)],
        labels=("outer", "inner"),
    )
    config = build_thread_config(f"stacked-hook-note-{run_mode}")
    run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)

    # Act
    run_messages(agent, [HumanMessage(NEXT_TURN)], mode=run_mode, config=config)

    # Assert
    expected = (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        NOTE_OBSERVATION,
        FIRST_ANSWER,
        NEXT_OBSERVATION,
        SECOND_ANSWER,
    )
    assert read_history(agent, config, label="outer") == expected
    assert read_history(agent, config, label="inner") == expected


def test_two_hook_written_messages_with_the_same_text_each_enter_the_trajectory_once(
    run_mode: RunMode,
) -> None:
    # Arrange: each note is written after a run's last step, so no step reads it without an id
    agent = build_noted_agent(
        build_answers("First.", "Second.", "Third."),
        writers=[NoteAfterAnswerMiddleware(text=NOTE)],
    )
    config = build_thread_config(f"same-text-{run_mode}")
    run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)
    run_messages(agent, [HumanMessage(NEXT_TURN)], mode=run_mode, config=config)

    # Act
    run_messages(agent, [HumanMessage(THIRD_TURN)], mode=run_mode, config=config)

    # Assert
    assert read_history(agent, config) == (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        FIRST_ANSWER,
        NOTE_OBSERVATION,
        NEXT_OBSERVATION,
        SECOND_ANSWER,
        NOTE_OBSERVATION,
        THIRD_OBSERVATION,
        THIRD_ANSWER,
    )


@pytest.mark.xfail(strict=True)
def test_a_message_with_an_id_all_along_enters_beside_a_hook_written_one_with_its_text(
    run_mode: RunMode,
) -> None:
    # Arrange: the first run reads a note without an id, then a note with an id is written
    agent = build_noted_agent(
        build_answers("First.", "Second."),
        writers=[
            NoteBeforeModelMiddleware(text=NOTE),
            NoteAfterAnswerMiddleware(text=NOTE, message_id=NOTE_ID),
        ],
    )
    config = build_thread_config(f"id-all-along-{run_mode}")
    run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)

    # Act
    run_messages(agent, [HumanMessage(NEXT_TURN)], mode=run_mode, config=config)

    # Assert
    assert read_history(agent, config) == (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        NOTE_OBSERVATION,
        FIRST_ANSWER,
        NOTE_OBSERVATION,
        NEXT_OBSERVATION,
        SECOND_ANSWER,
    )


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "labels",
    [("monitor",), ("outer", "inner")],
    ids=["one-monitor", "two-monitors"],
)
def test_a_message_read_in_a_run_that_did_not_end_enters_the_trajectory_once(
    run_mode: RunMode,
    labels: tuple[str, ...],
) -> None:
    # Arrange: the first run's second model call fails after a tool answered, so no call dangles
    first_step = build_read_step().model_copy(update={"id": "step-1"})
    model = ScriptedChatModel(responses=[first_step])
    agent = build_noted_agent(
        model,
        writers=[NoteBeforeModelMiddleware(text=NOTE)],
        labels=labels,
        tools=Workspace().build_tools(),
    )
    config = build_thread_config(f"unended-{len(labels)}-{run_mode}")
    with pytest.raises(AssertionError, match="ran out of responses"):
        run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)
    model.responses.append(AIMessage("Second.", id="step-2"))

    # Act
    run_messages(agent, [HumanMessage(NEXT_TURN)], mode=run_mode, config=config)

    # Assert
    call = ToolCall(id="call-read", name="read_file", arguments='{"path": "q3.md"}')
    expected = (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        NOTE_OBSERVATION,
        Action(id="step-1", tool_calls=(call,)),
        ToolObservation(call_id="call-read", parts=(TextPart(text="contents of q3.md"),)),
        Observation(
            author=Author.UNKNOWN,
            source="unconfirmed_input",
            parts=(TextPart(text=NEXT_TURN),),
        ),
        SECOND_ANSWER,
    )
    histories = {label: read_history(agent, config, label=label) for label in labels}
    assert histories == dict.fromkeys(labels, expected)


@pytest.mark.xfail(strict=True)
def test_a_string_input_with_the_text_of_an_earlier_note_enters_as_the_user_s(
    run_mode: RunMode,
) -> None:
    # Arrange: the first run reads a note without an id, and the next input repeats its text
    agent = build_noted_agent(
        build_answers("First.", "Second."),
        writers=[NoteBeforeModelMiddleware(text=NOTE)],
    )
    config = build_thread_config(f"repeated-text-{run_mode}")
    run_messages(agent, [HumanMessage(TASK)], mode=run_mode, config=config)

    # Act
    run_messages(agent, NOTE, mode=run_mode, config=config)

    # Assert
    assert read_history(agent, config) == (
        TASK_OBSERVATION,
        SYSTEM_OBSERVATION,
        NOTE_OBSERVATION,
        FIRST_ANSWER,
        Observation(author=Author.USER, parts=(TextPart(text=NOTE),)),
        SECOND_ANSWER,
    )
