"""Which human messages the task author wrote: only those that arrive as a run's input.

A monitor reads an untagged human message as the words of the user, or of the
agent that delegated the task, and only those words authorise an action. Deep
Agents writes some human messages during a run without a tag: `read_file`
attaches the frames of a video in one whose text holds the path the agent
chose, and its Nemotron profile writes named nudges [@deepagents2026]. So the
middleware records, in the graph state, which human messages were a run's
input, and the monitor reads every other one as a context note.

- At the start of a run, an untagged human message the monitor has not seen
  is the run's input, and is recorded under `TASK_MESSAGES_KEY`.
- Every step records the untagged human messages then in the state under
  `SEEN_HUMAN_MESSAGES_KEY`, so a message written during one run is never
  taken for the next run's input.
- A new untagged human message a tool writes is tagged where it is written,
  so it stays a note even when the run fails before the next step, and in a
  history the application stores and replays.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import replace

from langchain_core.messages import BaseMessage, HumanMessage, convert_to_messages
from langgraph.types import Command

from langchain_sync_monitors._langchain import AgentStateUpdate, ToolCallResult
from langchain_sync_monitors.transcript import (
    find_untagged_human_message_ids,
    is_untagged_human_message,
    tag_as_context_note,
)

TASK_MESSAGES_KEY = "monitor_task_messages"
"""The state key that holds the ids of the human messages that arrived as a run's input."""

SEEN_HUMAN_MESSAGES_KEY = "monitor_seen_human_messages"
"""The state key that holds the ids of every untagged human message the monitor has seen."""


def merge_message_ids(  # lanorme: ignore[KWARG-001]
    recorded: list[str],
    new: list[str],
) -> list[str]:
    """Add newly recorded message ids to the recorded ones, keeping each id once, in order.

    Stacked monitors commit in the same model node, so both may record one
    id. LangGraph calls a reducer with both values by position [@langgraph2026].
    """
    return list(dict.fromkeys([*recorded, *new]))


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


def find_unseen_human_message_ids(state: object) -> list[str]:
    """Return the ids of the untagged human messages in the state the monitor has not seen."""
    seen_ids = read_message_ids(state, key=SEEN_HUMAN_MESSAGES_KEY)
    messages = read_state_messages(state)
    return [
        message_id
        for message_id in find_untagged_human_message_ids(messages)
        if message_id not in seen_ids
    ]


def build_run_input_update(state: object) -> AgentStateUpdate | None:
    """Return the update that records a run's input as the task author's messages.

    At the start of a run, an untagged human message the monitor has not seen
    is the run's input: the monitor saw every earlier one, at the step that
    followed it, and recorded it as seen.
    """
    new_ids = find_unseen_human_message_ids(state)
    if not new_ids:
        return None
    return {TASK_MESSAGES_KEY: new_ids, SEEN_HUMAN_MESSAGES_KEY: new_ids}


def build_seen_messages_update(state: object) -> AgentStateUpdate:
    """Return the update that records the untagged human messages a step sees, if any are new."""
    unseen_ids = find_unseen_human_message_ids(state)
    return {SEEN_HUMAN_MESSAGES_KEY: unseen_ids} if unseen_ids else {}


def is_new_untagged_human_message(message: BaseMessage, *, existing_ids: frozenset[str]) -> bool:
    """Tell whether a message a tool writes is an untagged human message the state lacks."""
    return is_untagged_human_message(message) and message.id not in existing_ids


def mark_tool_written_notes(
    result: ToolCallResult,
    *,
    tool_name: str,
    state: object,
) -> ToolCallResult:
    """Tag the new untagged human messages a tool writes through a `Command` as context notes.

    The tool's name is the note's source. Messages given as dictionaries,
    tuples or strings are converted first, as LangGraph's message reducer
    would convert them [@langgraph2026]. A message whose id is already in the
    state replaces one there, as a tool that rewrites the history does, and
    keeps its author.
    """
    if not isinstance(result, Command) or not isinstance(result.update, dict):
        return result
    written = result.update.get("messages")
    if not isinstance(written, list):
        return result
    messages = convert_to_messages(written)
    existing_ids = frozenset(message.id for message in read_state_messages(state) if message.id)
    is_new_note = functools.partial(is_new_untagged_human_message, existing_ids=existing_ids)
    if not any(map(is_new_note, messages)):
        return result
    tagged = [
        tag_as_context_note(message, source=tool_name)
        if isinstance(message, HumanMessage) and is_new_note(message)
        else message
        for message in messages
    ]
    return replace(result, update={**result.update, "messages": tagged})
