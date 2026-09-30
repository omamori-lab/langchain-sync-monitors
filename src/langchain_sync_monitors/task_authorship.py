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
- A human message a tool writes is tagged where it is written, and no
  message a tool writes keeps a source only the monitor writes, whatever
  shape the tool's update takes, a command it raises as a `ParentCommand`
  included; a message the tool writes back unchanged, under its id, is left
  as it was, and one it writes under the id of a human message the monitor
  saw and the state no longer holds loses that id. A tool's `Command` writes
  to the state keys only the monitor writes, `MONITOR_STATE_KEYS`, are
  dropped with a warning, so no command a tool returns or raises can record
  a message as a run's input or keep words of its own as the user's. A
  `Send` a tool's command carries reaches another node's input, not these
  writes, and stays open (#76).
- `RUN_OPEN_KEY` is set at the start of a run and of each step, and cleared
  when the run reaches the monitor's `after_agent` hook. A run that starts
  while it is still set follows one that stopped early, or a fork from a
  checkpoint inside a run, and the state may hold messages the monitor never
  saw. The monitor cannot tell them from the new input, so it tags all of
  them as notes from `unconfirmed_input`, which they stay for the rest of
  the thread. The prompt tells the judge that such a note may be the user's
  own words: it authorises nothing, and only a limit it sets that narrows
  what the agent may do still applies, since no note removes a safeguard.

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
from dataclasses import dataclass
from typing import TypeGuard

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import ParentCommand
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    AgentStateUpdate,
    ToolCallResult,
    ToolCallResults,
    read_update_pairs,
    replace_update_pairs,
    rewrite_update_messages,
)
from langchain_sync_monitors.state_keys import (
    MONITOR_STATE_KEYS,
    RUN_OPEN_KEY,
    SEEN_HUMAN_MESSAGES_KEY,
    TASK_MESSAGES_KEY,
)
from langchain_sync_monitors.transcript import MONITOR_FEEDBACK_SOURCE, read_message_source

logger = logging.getLogger(__name__)

APPLICATION_SOURCE = "application"
"""The source of a context note made from a human message that names no source of its own."""

UNCONFIRMED_INPUT_SOURCE = "unconfirmed_input"
"""The source of a note made from a human message that may be a run's input, but cannot be told
from one an earlier run that stopped early left behind."""

RESERVED_SOURCES = frozenset({MONITOR_FEEDBACK_SOURCE, UNCONFIRMED_INPUT_SOURCE})
"""The sources only the monitor writes: its feedback, and input it cannot confirm."""


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
    """Return a copy of a human message with an `lc_source` tag, which makes it a context note."""
    additional_kwargs = {**message.additional_kwargs, "lc_source": source}
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def build_note_source(name: str) -> str:
    """Return the source of a note named after a tool or a message's name.

    The monitor does not choose those names, so one of its own sources
    becomes `application`: a tool called `monitor` must not write the
    monitor's feedback, nor one called `unconfirmed_input` a note that may be
    the user's.
    """
    return APPLICATION_SOURCE if name in RESERVED_SOURCES else name


def mark_context_note(message: HumanMessage) -> HumanMessage:
    """Return a copy of a human message tagged as a context note.

    The note's source is the message's `name`, as Deep Agents' Nemotron
    profile names its nudges [@deepagents2026], or else `application`.
    """
    return tag_as_context_note(
        message, source=build_note_source(message.name or APPLICATION_SOURCE)
    )


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
    fork inside a run, it may not have, so it records no input, tags each of
    those messages as a note from `unconfirmed_input`, and warns with their
    ids, so the application can tell the user.
    """
    new_ids = find_unseen_human_message_ids(state)
    update: AgentStateUpdate = {RUN_OPEN_KEY: True}
    if not new_ids:
        return update
    if not is_run_open(state):
        return {**update, TASK_MESSAGES_KEY: new_ids, SEEN_HUMAN_MESSAGES_KEY: new_ids}
    logger.warning(
        "This run starts from a run that did not reach its end, so the monitor cannot tell "
        "its input from messages the earlier run left behind. It reads the human messages "
        "%s as notes from unconfirmed input: they authorise nothing, and only a limit they "
        "set that narrows what the agent may do still applies.",
        new_ids,
    )
    unconfirmed = [
        tag_as_context_note(message, source=UNCONFIRMED_INPUT_SOURCE)
        for message in read_state_messages(state)
        if isinstance(message, HumanMessage) and message.id in new_ids
    ]
    return {**update, SEEN_HUMAN_MESSAGES_KEY: new_ids, "messages": unconfirmed}


def build_note_update(state: object) -> AgentStateUpdate:
    """Return the update that records the human messages in the state and tags the notes.

    Every untagged human message is recorded as seen, and each one that was
    not a run's input is written back, with its id, tagged as a note. The
    reducer replaces a message that has the same id [@langgraph2026], so
    the conversation keeps its order.
    """
    task_message_ids = read_message_ids(state, key=TASK_MESSAGES_KEY)
    unseen_ids = find_unseen_human_message_ids(state)
    # Only a message with an id can be replaced in place; one without would be added again.
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

    A human message becomes a note named after the tool. Only the monitor
    writes its own sources, its feedback in the model node and unconfirmed
    input at the start of a run, so any other message a tool tags with one of
    them loses the tag.
    """
    source = read_message_source(message)
    if isinstance(message, HumanMessage) and (source is None or source in RESERVED_SOURCES):
        return tag_as_context_note(message, source=build_note_source(tool_name))
    if source not in RESERVED_SOURCES:
        return message
    additional_kwargs = {
        key: value for key, value in message.additional_kwargs.items() if key != "lc_source"
    }
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def is_unchanged_write_back(
    message: BaseMessage,
    *,
    existing_messages: Mapping[str, BaseMessage],
) -> bool:
    """Tell whether a tool writes back, under its id, a message the state holds unchanged.

    Such a message keeps its author, as when a tool rewrites the history.
    One the tool changed in any field does not: a tool that edits a message
    by id must not put the agent's words in the user's. Metadata can change
    what a message says, too: Deep Agents' `FilesystemMiddleware` shows a
    human message that carries `additional_kwargs["lc_evicted_to"]` as a stub
    that names that path [@deepagents2026]. So the whole message is compared,
    its exact type included, field by field through `model_dump`: a subclass
    whose own `__eq__` calls it equal to anything is still told apart, since
    Python asks a subclass first.
    """
    existing = existing_messages.get(message.id) if message.id else None
    return (
        existing is not None
        and type(message) is type(existing)
        and message.model_dump() == existing.model_dump()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class StateBeforeTool:
    """What the state held when a tool ran, as the relabelling of its writes reads it.

    `existing_messages` holds its messages by id. `removed_seen_ids` holds
    the ids of the human messages the monitor has seen that it no longer
    holds, such as a run's input a tool removed.
    """

    existing_messages: Mapping[str, BaseMessage]
    removed_seen_ids: frozenset[str]


def read_state_before_tool(state: object) -> StateBeforeTool:
    """Return what the state held when a tool ran: its messages, and the seen ids it lost."""
    existing_messages = read_existing_messages(state)
    seen_ids = read_message_ids(state, key=SEEN_HUMAN_MESSAGES_KEY)
    return StateBeforeTool(
        existing_messages=existing_messages,
        removed_seen_ids=seen_ids - existing_messages.keys(),
    )


def reissue_removed_seen_id(message: BaseMessage, *, before: StateBeforeTool) -> BaseMessage:
    """Return a tool's message without an id when it names a seen human message the state lost.

    LangGraph adds a message under an id the state lacks at the end
    [@langgraph2026], far from where the message that had the id stood. The
    monitor places a kept run input just before a message under its id, which
    is right for a rewrite in place, so such a message must not keep the id:
    a tool could otherwise move the user's answer after a question the agent
    asked later. Without an id, LangGraph gives it a fresh one. A
    `RemoveMessage` keeps its id, which is what it removes.
    """
    if isinstance(message, RemoveMessage) or message.id not in before.removed_seen_ids:
        return message
    return message.model_copy(update={"id": None})


def relabel_unless_written_back(
    message: BaseMessage,
    *,
    tool_name: str,
    before: StateBeforeTool,
) -> BaseMessage:
    """Relabel a message a tool writes, unless the tool writes it back unchanged."""
    if is_unchanged_write_back(message, existing_messages=before.existing_messages):
        return message
    reissued = reissue_removed_seen_id(message, before=before)
    return relabel_tool_written_message(reissued, tool_name=tool_name)


def relabel_tool_command(
    command: Command,
    *,
    tool_name: str,
    before: StateBeforeTool,
) -> Command:
    """Relabel the messages a tool's `Command` writes, and drop its monitor state writes.

    The messages are read as LangGraph writes them, from an update given as
    a dict, as pairs of key and value, or as an object whose class annotates
    its keys, such as a dataclass or a pydantic model, and converted as its
    message reducer converts them, whether one message or a list, given as
    messages, dictionaries, tuples or strings, or wrapped in an `Overwrite`
    [@langgraph2026]. A dict stays a dict, and any other update becomes the
    pairs LangGraph would write, so the state receives the same writes with
    the messages relabelled. A command that writes no messages, such as one
    with only a `goto`, is returned as it is. The reader is private to
    LangGraph; without it, an update other than a dict or pairs raises
    `MonitorError`. The command's writes to `MONITOR_STATE_KEYS`, in any
    update shape, are dropped first, by `drop_monitor_state_writes`.
    """
    return rewrite_update_messages(
        drop_monitor_state_writes(command, tool_name=tool_name),
        rewrite=lambda message: relabel_unless_written_back(
            message, tool_name=tool_name, before=before
        ),
    )


def drop_monitor_state_writes(command: Command, *, tool_name: str) -> Command:
    """Return a tool's command without its writes to the state keys only the monitor writes.

    Through them a tool could record a message it wrote as a run's input,
    keep words of its own as the user's, or change a halt's count, so each
    such write is dropped, and a warning names the tool and the keys. The
    pairs are read as LangGraph writes them, so every update shape is
    covered, and a value goes with its key whatever its form, an `Overwrite`
    included. Only the command's own update is read: a `Send` it carries
    reaches another node's input and stays open (#76). `monitor_log` is not
    among the keys: Deep Agents' `task` tool returns a subagent's records
    through it [@deepagents2026].
    """
    pairs = read_update_pairs(command)
    dropped = [str(key) for key, _ in pairs if is_monitor_state_key(key)]
    if not dropped:
        return command
    logger.warning(
        "The tool %s wrote the state keys %s, which only the monitor writes, in a command's "
        "update, so the monitor dropped those writes.",
        tool_name,
        dropped,
    )
    kept = [pair for pair in pairs if not is_monitor_state_key(pair[0])]
    return replace_update_pairs(command, pairs=kept)


def is_monitor_state_key(key: str) -> bool:
    """Tell whether an update's key names a state key only the monitor writes.

    A key is compared with `==`, as LangGraph finds its channel, so a string
    subclass that only its own `__ne__` sets apart is still caught.
    """
    return any(key == name for name in MONITOR_STATE_KEYS)


def relabel_tool_result(
    result: ToolCallResult,
    *,
    tool_name: str,
    before: StateBeforeTool,
) -> ToolCallResult:
    """Relabel what one item of a tool's result writes, a tool message or a `Command`."""
    if isinstance(result, Command):
        return relabel_tool_command(result, tool_name=tool_name, before=before)
    if is_unchanged_write_back(result, existing_messages=before.existing_messages):
        return result
    reissued = reissue_removed_seen_id(result, before=before)
    if read_message_source(reissued) not in RESERVED_SOURCES:
        return reissued if isinstance(reissued, ToolMessage) else result
    relabelled = relabel_tool_written_message(reissued, tool_name=tool_name)
    # A tool message stays one; the check only narrows the type for the type checker.
    return relabelled if isinstance(relabelled, ToolMessage) else result


def read_existing_messages(state: object) -> dict[str, BaseMessage]:
    """Return the messages in the state by id, leaving out any without an id."""
    return {message.id: message for message in read_state_messages(state) if message.id}


def relabel_parent_command(bubble: ParentCommand, *, tool_name: str, state: object) -> None:
    """Relabel, in place, what the command in a `ParentCommand` a tool call raises writes.

    A tool can raise one, or call a graph whose node returns a command for
    its parent graph, and LangGraph applies that command as the tools node's
    own writes, or hands it on to the graph it names [@langgraph2026]. Its
    messages are relabelled as a returned command's are, and it keeps its
    `graph`, `goto` and `resume`. The command is replaced in the exception,
    as LangGraph itself replaces it on the way up.
    """
    [command] = bubble.args
    relabelled = relabel_tool_command(
        command, tool_name=tool_name, before=read_state_before_tool(state)
    )
    bubble.args = (relabelled,)


def mark_tool_written_notes(
    results: ToolCallResults,
    *,
    tool_name: str,
    state: object,
) -> ToolCallResults:
    """Tag the human messages a tool writes as notes, and strip the monitor's own sources.

    A tool can return a tool message, a `Command`, or a list of both, which
    LangGraph's tool node accepts [@langgraph2026], and a `Command`'s update
    can take any shape LangGraph accepts; `relabel_tool_command` reads each,
    and `relabel_parent_command` a command the tool raises instead.
    Every new or changed message the tool writes loses a source only the
    monitor writes, and a human message left without a source becomes a
    note named after the tool, or `application` for a tool named after one
    of the monitor's sources. A write to a state key only the monitor
    writes is dropped, and a message written under the id of a human message
    the monitor saw and the state no longer holds loses that id, by
    `reissue_removed_seen_id`. A message the tool writes back under its id,
    unchanged in every field, keeps its author and its source. Tagged where
    it is written, a human message stays a note in every later run, even one
    that starts before the monitor has seen it, and in a history the
    application stores and replays.
    """
    before = read_state_before_tool(state)
    if isinstance(results, list):
        return [
            relabel_tool_result(result, tool_name=tool_name, before=before) for result in results
        ]
    return relabel_tool_result(results, tool_name=tool_name, before=before)
