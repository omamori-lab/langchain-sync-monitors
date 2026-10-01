"""Untagged human messages that reach the state without an id, and the end of a run.

Deep Agents keeps `messages` in a channel whose reducer adds a message
without an id as it is, and LangGraph gives an id to a message written as a
message or a dictionary, not to one written as a string or a `(role, text)`
tuple [@deepagents2026; @langgraph2026]. The monitor records human messages
by id, so a run's input given that way would never be recorded: the judge
would read the user's task as a note, and a halt would stand. So the
monitor gives each untagged human message without an id a fresh one at the
start of a run, before recording the input (`run_inputs`), and at the end
of a run, where one still without an id was written during the run, tagged
as a note, so the next run cannot take it for input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Overwrite

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.state_keys import RUN_OPEN_KEY
from langchain_sync_monitors.task_authorship import (
    build_note_update,
    is_untagged_human_message,
    read_state_messages,
    tag_as_context_note_from_name,
)


def is_human_message_without_id(message: BaseMessage) -> TypeGuard[HumanMessage]:
    """Tell whether a message is an untagged human message without an id."""
    return is_untagged_human_message(message) and not message.id


def assign_human_message_id(message: HumanMessage, *, as_notes: bool) -> HumanMessage:
    """Return the message with a fresh id, and tagged as a context note with `as_notes`."""
    # The plain uuid4 LangGraph gives a message it writes; the monitor did not write it.
    message_with_id = message.model_copy(update={"id": str(uuid4())})
    return tag_as_context_note_from_name(message_with_id) if as_notes else message_with_id


def assign_human_message_ids(state: object, *, as_notes: bool) -> list[BaseMessage] | None:
    """Return the state's messages with a fresh id on each untagged human message without one.

    With `as_notes`, each such message is also tagged as a context note.
    `None` means no message lacks an id.
    """
    messages = read_state_messages(state)
    if not any(is_human_message_without_id(message) for message in messages):
        return None
    return [
        assign_human_message_id(message, as_notes=as_notes)
        if is_human_message_without_id(message)
        else message
        for message in messages
    ]


def replace_state_messages(state: object, *, messages: list[BaseMessage]) -> dict[str, object]:
    """Return a copy of the state that holds `messages`, for reading it as it will be."""
    values = dict(state) if isinstance(state, Mapping) else {}
    return {**values, "messages": messages}


def build_assigned_ids_update(
    messages_with_ids: list[BaseMessage],
    *,
    update: AgentStateUpdate,
) -> AgentStateUpdate:
    """Return `update` writing the whole history back as `messages_with_ids`, in an `Overwrite`.

    A message without an id cannot be replaced in place, since the reducer
    would add its copy as a new message, so the history is written back
    whole. An `Overwrite`, unlike a removal of every message, puts nothing
    on `stream_mode="messages"`. The messages `update` already writes, each
    under an id `messages_with_ids` holds, replace theirs.
    """
    written = {message.id: message for message in update.get("messages", []) if message.id}
    messages = [
        written.get(message.id, message) if message.id else message for message in messages_with_ids
    ]
    return {**update, "messages": Overwrite(messages)}


def build_run_end_update(state: object) -> AgentStateUpdate:
    """Return the update a run ends with: the notes so far, and the run marked closed.

    An untagged human message still without an id was written during the
    run, since the run's start gave its input one, so it gets an id and is
    tagged as a note: left as it is, the next run would take it for input.
    """
    messages_with_ids = assign_human_message_ids(state, as_notes=True)
    if messages_with_ids is None:
        return {**build_note_update(state), RUN_OPEN_KEY: False}
    update = build_note_update(replace_state_messages(state, messages=messages_with_ids))
    return {**build_assigned_ids_update(messages_with_ids, update=update), RUN_OPEN_KEY: False}
