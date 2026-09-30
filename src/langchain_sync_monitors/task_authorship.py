"""Which human messages the task author wrote: only those that arrive as a run's input.

A monitor reads an untagged human message as the words of the user, or of the
agent that delegated the task, and only those words authorise an action. Deep
Agents writes some human messages during a run without a tag: `read_file`
attaches the frames of a video in one whose text holds the path the agent
chose, and its Nemotron profile writes nudges, one of which quotes an argument
the agent chose [@deepagents2026]. So the middleware records, in the graph
state, which human messages were a run's input, and tags every other one as a
context note, in the state as well as in what the monitor reads.

- At the start of a run, an untagged human message the monitor has not seen
  is the run's input, and is recorded under `TASK_MESSAGES_KEY`.
- Before each step, when it commits and when the run ends, the monitor
  records the untagged human messages then in the state as seen, under
  `SEEN_HUMAN_MESSAGES_KEY`, and writes each one that is not a run's input
  back, by id, tagged as a note, so a replayed history keeps the tag.
- A human message a tool writes is tagged where it is written.
- `RUN_OPEN_KEY` is set at the start of a run and of each step, and cleared
  when the run reaches the monitor's `after_agent` hook. A run that starts
  while it is still set follows one that stopped early, or a fork from a
  checkpoint inside a run, and the state may hold messages the monitor never
  saw. The monitor cannot tell them from the new input, so it reads all of
  them as notes, which they stay for the rest of the thread.

Two paths stay open, both through another middleware listed before the
monitor. Its `before_agent` hook runs before the monitor's own, so an untagged
human message it writes counts as the run's input. Its `after_agent` hook runs
after the monitor's own, since LangChain runs those hooks in reverse order
[@langchain2026], so an untagged human message it writes counts as the next
run's input, unless the hook sends the run back to the model.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TypeGuard

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, convert_to_messages
from langgraph.types import Command

from langchain_sync_monitors._langchain import AgentStateUpdate, ToolCallResult, ToolCallResults
from langchain_sync_monitors.transcript import MONITOR_FEEDBACK_SOURCE, read_message_source

logger = logging.getLogger(__name__)

TASK_MESSAGES_KEY = "monitor_task_messages"
"""The state key that holds the ids of the human messages that arrived as a run's input."""

SEEN_HUMAN_MESSAGES_KEY = "monitor_seen_human_messages"
"""The state key that holds the ids of every untagged human message the monitor has seen."""

RUN_OPEN_KEY = "monitor_run_open"
"""The state key that is true from the start of a run until the run reaches its end."""

APPLICATION_SOURCE = "application"
"""The source of a context note made from a human message that names no source of its own."""


def merge_message_ids(  # lanorme: ignore[KWARG-001]
    recorded: list[str],
    new: list[str],
) -> list[str]:
    """Add newly recorded message ids to the recorded ones, keeping each id once, in order.

    Stacked monitors commit in the same model node, so both may record one
    id. LangGraph calls a reducer with both values by position [@langgraph2026].
    """
    return list(dict.fromkeys([*recorded, *new]))


def keep_latest_flag(earlier: bool, latest: bool) -> bool:  # lanorme: ignore[KWARG-001]
    """Keep the latest value of a flag that stacked monitors may write in the same node."""
    del earlier
    return latest


def is_untagged_human_message(message: BaseMessage) -> TypeGuard[HumanMessage]:
    """Tell whether a message is a human message that no part of the application tagged."""
    return isinstance(message, HumanMessage) and read_message_source(message) is None


def find_untagged_human_message_ids(messages: Iterable[BaseMessage]) -> list[str]:
    """Return the ids of the untagged human messages, in order, leaving out any without an id."""
    return [message.id for message in messages if is_untagged_human_message(message) and message.id]


def tag_as_context_note(message: HumanMessage, *, source: str) -> HumanMessage:
    """Return a copy of a human message with an `lc_source` tag, which makes it a context note.

    The source comes from a tool's or a message's name, which the monitor
    does not choose, so the monitor's own source becomes `application`: a
    tool called `monitor` must not write the monitor's feedback.
    """
    note_source = APPLICATION_SOURCE if source == MONITOR_FEEDBACK_SOURCE else source
    additional_kwargs = {**message.additional_kwargs, "lc_source": note_source}
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def mark_context_note(message: HumanMessage) -> HumanMessage:
    """Return a copy of a human message tagged as a context note.

    The note's source is the message's `name`, as Deep Agents' Nemotron
    profile names its nudges [@deepagents2026], or else `application`.
    """
    return tag_as_context_note(message, source=message.name or APPLICATION_SOURCE)


def is_note_to_mark(message: BaseMessage, *, task_message_ids: Collection[str]) -> bool:
    """Tell whether a message is an untagged human message that was not a run's input."""
    return is_untagged_human_message(message) and message.id not in task_message_ids


def mark_context_notes(
    history: Sequence[BaseMessage],
    *,
    task_message_ids: Collection[str],
) -> tuple[BaseMessage, ...]:
    """Tag every untagged human message the task author did not write as a context note.

    `task_message_ids` holds the ids of the human messages that arrived as a
    run's input. Any other untagged human message was written during a run,
    by a tool, a middleware or the application, and a tool can put the
    agent's own words in it, so it must not speak as the user. This tags the
    monitor's copy of a model request, which can hold messages the state
    lacks. A message without an id is never the task author's.
    """
    return tuple(
        mark_context_note(message)
        if isinstance(message, HumanMessage)
        and is_note_to_mark(message, task_message_ids=task_message_ids)
        else message
        for message in history
    )


def read_state_messages(state: object) -> list[BaseMessage]:
    """Return the messages in an agent state, which LangChain leaves untyped for tools."""
    messages = state.get("messages") if isinstance(state, Mapping) else None
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, BaseMessage)]


def read_message_ids(state: object, *, key: str) -> frozenset[str]:
    """Return the message ids recorded under a state key, or none when the key is missing.

    A step whose state holds no recorded input reads every human message as
    a note, so it fails closed.
    """
    ids = state.get(key) if isinstance(state, Mapping) else None
    if not isinstance(ids, list):
        return frozenset()
    return frozenset(value for value in ids if isinstance(value, str))


def is_run_open(state: object) -> bool:
    """Tell whether the state belongs to a run that has not reached its end."""
    return isinstance(state, Mapping) and state.get(RUN_OPEN_KEY) is True


def find_unseen_human_message_ids(state: object) -> list[str]:
    """Return the ids of the untagged human messages in the state the monitor has not seen."""
    seen_ids = read_message_ids(state, key=SEEN_HUMAN_MESSAGES_KEY)
    messages = read_state_messages(state)
    return [
        message_id
        for message_id in find_untagged_human_message_ids(messages)
        if message_id not in seen_ids
    ]


def build_run_input_update(state: object) -> AgentStateUpdate:
    """Return the update that records a run's input as the task author's messages.

    At the start of a run, an untagged human message the monitor has not seen
    is the run's input: the monitor saw every earlier one, at the step or the
    end that followed it. After a run that stopped before its end, or from a
    fork inside a run, it may not have, so it records no input and warns.
    """
    new_ids = find_unseen_human_message_ids(state)
    update: AgentStateUpdate = {RUN_OPEN_KEY: True}
    if not new_ids:
        return update
    if is_run_open(state):
        logger.warning(
            "This run starts from a run that did not reach its end, so the monitor cannot "
            "tell its input from messages the earlier run left behind; it reads %d human "
            "message(s) as context notes, which authorise nothing.",
            len(new_ids),
        )
        return {**update, SEEN_HUMAN_MESSAGES_KEY: new_ids}
    return {**update, TASK_MESSAGES_KEY: new_ids, SEEN_HUMAN_MESSAGES_KEY: new_ids}


def build_note_update(state: object) -> AgentStateUpdate:
    """Return the update that records the human messages in the state and tags the notes.

    Every untagged human message is recorded as seen, and each one that was
    not a run's input is written back, with its id, tagged as a note. The
    reducer replaces a message that has the same id [@langgraph2026], so
    the conversation keeps its order.
    """
    task_message_ids = read_message_ids(state, key=TASK_MESSAGES_KEY)
    unseen_ids = find_unseen_human_message_ids(state)
    notes = [
        mark_context_note(message)
        for message in read_state_messages(state)
        if isinstance(message, HumanMessage)
        and message.id
        and is_note_to_mark(message, task_message_ids=task_message_ids)
    ]
    update: AgentStateUpdate = {}
    if unseen_ids:
        update[SEEN_HUMAN_MESSAGES_KEY] = unseen_ids
    if notes:
        update["messages"] = notes
    return update


def build_step_start_update(state: object) -> AgentStateUpdate:
    """Return the update a step starts with: the notes so far, and the run marked open.

    An `after_agent` hook can send the run back to the model after the
    monitor's own `after_agent` hook marked it closed, so each step opens it
    again.
    """
    return {**build_note_update(state), RUN_OPEN_KEY: True}


def build_run_end_update(state: object) -> AgentStateUpdate:
    """Return the update a run ends with: the notes so far, and the run marked closed."""
    return {**build_note_update(state), RUN_OPEN_KEY: False}


def relabel_tool_written_message(message: BaseMessage, *, tool_name: str) -> BaseMessage:
    """Return a message a tool wrote so that it speaks neither as the task author nor the monitor.

    A human message becomes a note named after the tool. Only the model node
    writes the monitor's feedback, so any other message a tool tags with the
    monitor's source loses the tag.
    """
    source = read_message_source(message)
    if isinstance(message, HumanMessage) and source in {None, MONITOR_FEEDBACK_SOURCE}:
        return tag_as_context_note(message, source=tool_name)
    if source != MONITOR_FEEDBACK_SOURCE:
        return message
    additional_kwargs = {
        key: value for key, value in message.additional_kwargs.items() if key != "lc_source"
    }
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def relabel_tool_command(
    command: Command,
    *,
    tool_name: str,
    existing_ids: frozenset[str],
) -> Command:
    """Relabel the new messages in a tool's `Command` update.

    Messages given as dictionaries, tuples or strings, one or a list, are
    converted first, as LangGraph's message reducer would convert them
    [@langgraph2026]. A message whose id the state already holds replaces one
    there, as a tool that rewrites the history does, and keeps its author.
    """
    if not isinstance(command.update, dict) or "messages" not in command.update:
        return command
    written = command.update["messages"]
    messages = convert_to_messages(written if isinstance(written, list) else [written])
    relabelled = [
        message
        if message.id in existing_ids
        else relabel_tool_written_message(message, tool_name=tool_name)
        for message in messages
    ]
    return replace(command, update={**command.update, "messages": relabelled})


def relabel_tool_result(
    result: ToolCallResult,
    *,
    tool_name: str,
    existing_ids: frozenset[str],
) -> ToolCallResult:
    """Relabel what one item of a tool's result writes, a tool message or a `Command`."""
    if isinstance(result, Command):
        return relabel_tool_command(result, tool_name=tool_name, existing_ids=existing_ids)
    if read_message_source(result) != MONITOR_FEEDBACK_SOURCE or result.id in existing_ids:
        return result
    relabelled = relabel_tool_written_message(result, tool_name=tool_name)
    return relabelled if isinstance(relabelled, ToolMessage) else result


def mark_tool_written_notes(
    results: ToolCallResults,
    *,
    tool_name: str,
    state: object,
) -> ToolCallResults:
    """Tag the new human messages a tool writes as notes, and strip the monitor's source.

    A tool can return a tool message, a `Command`, or a list of both, which
    LangGraph's tool node accepts [@langgraph2026]. Tagged where it is
    written, with the tool's name as its source, a human message stays a note
    in every later run, even one that starts before the monitor has seen it,
    and in a history the application stores and replays.
    """
    existing_ids = frozenset(message.id for message in read_state_messages(state) if message.id)
    if isinstance(results, list):
        return [
            relabel_tool_result(result, tool_name=tool_name, existing_ids=existing_ids)
            for result in results
        ]
    return relabel_tool_result(results, tool_name=tool_name, existing_ids=existing_ids)
