"""The text of every run's input, kept so the judge reads each one after it leaves the history.

Every human message recorded as a run's input, under `TASK_MESSAGES_KEY`,
reaches the judge as the task author's words, verbatim and in its original
order, even once the model request no longer holds it. It leaves when the
agent summarises: LangChain's `SummarizationMiddleware` replaces the earlier
messages in the state with a summary, and Deep Agents replaces them in the
model request [@langchain2026; @deepagents2026]. It also leaves when a tool
removes it by id, or writes another message under its id, which becomes the
tool's note [@langgraph2026]. The summary stays a context note, which
authorises nothing, beside the kept turns: the agent's model wrote it.

- At the start of a run, each new input's text is kept under
  `RUN_INPUTS_KEY`, in the same update that records its id. That is the one
  hook sure to see it: a summariser listed before the monitor runs its own
  `before_model` hook before the monitor's, and on a run's first step it can
  already remove an earlier run's input. Input a run cannot confirm, after
  one that stopped early, is kept too, but it goes back as a note from
  `unconfirmed_input`, never as the task author's words, and is never
  recorded as a run's input, so it lifts no halt.
- Only the text is kept, which is what the judge reads, so the state grows
  by the text of every run's input and no more. A turn put back carries its
  text alone: a monitor that reads `MonitorInput.history` itself gets no
  image the turn held.
- A kept input follows its message in the state. Whenever the state holds
  it under its id as the monitor renders it, untagged or as a note from
  `unconfirmed_input`, the kept text becomes that message's text: at the
  start of a run and of each step, at each commit, and, in memory, before
  each judgement. So a trusted rewrite, such as `PIIMiddleware`'s redaction
  [@langchain2026] or the user's own `update_state` edit, reaches the judge
  as the agent reads it. A tool cannot write such a message: `task_authorship`
  tags every human message a tool writes, and drops its writes to this key.
- Before a monitor judges a step, each kept input its copy of the
  conversation lacks is put back. One the copy holds under its id, as the
  monitor renders it, with other text, such as the preview Deep Agents shows
  the agent in place of a large message while the state keeps the whole
  [@deepagents2026], is replaced by the kept text. The agent's own request
  is never changed.

A kept input goes back just before the message that took its id, else just
after the nearest of the three messages before it that the conversation
still holds, else at the start, just before the summary that replaced it and
its neighbours. Either way it comes after the input before it and before the
next one the conversation holds. So an input a tool removed together with
the three messages before it goes back right after the input before it,
ahead of that input's surviving steps.

A redaction and a summarisation that both reach a turn in the same
`before_model` pass, before the monitor has seen the redaction, leave the
kept copy unredacted: the redaction never reaches a state the monitor reads.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.task_authorship import (
    UNCONFIRMED_INPUT_SOURCE,
    build_run_input_update,
    find_unseen_human_message_ids,
    is_run_open,
    read_state_messages,
    tag_as_context_note,
)
from langchain_sync_monitors.transcript import read_message_source

RUN_INPUTS_KEY = "monitor_run_inputs"
"""The state key that holds the text of every human message a run received as its input."""

ANCHOR_COUNT = 3
"""How many of the messages before an input are kept as the places it can go back to."""


class RunInput(TypedDict):
    """The kept copy of one human message a run received as its input.

    `text` is the message's text, which is what the judge reads.
    `previous_message_ids` holds the ids of up to `ANCHOR_COUNT` messages
    before it when it was recorded, nearest first, and is empty when it
    opened the thread. `confirmed` is false for input a run could not confirm,
    which goes back as a note from `unconfirmed_input`.
    """

    id: str
    text: str
    previous_message_ids: list[str]
    confirmed: bool


def is_run_input(value: object) -> bool:
    """Tell whether a value from the state has the shape of a kept run input."""
    if not isinstance(value, Mapping):
        return False
    previous_message_ids = value.get("previous_message_ids")
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("text"), str)
        and isinstance(value.get("confirmed"), bool)
        and isinstance(previous_message_ids, list)
        and all(isinstance(previous_id, str) for previous_id in previous_message_ids)
    )


def read_run_input_entries(value: object) -> list[RunInput]:
    """Return a copy of each well-formed kept input in a value, leaving out anything else."""
    if not isinstance(value, list):
        return []
    return [
        RunInput(
            id=entry["id"],
            text=entry["text"],
            previous_message_ids=list(entry["previous_message_ids"]),
            confirmed=entry["confirmed"],
        )
        for entry in value
        if is_run_input(entry)
    ]


def merge_run_inputs(  # lanorme: ignore[KWARG-001]
    recorded: list[RunInput],
    new: list[RunInput],
) -> list[RunInput]:
    """Keep one copy of each input: its latest text, in the place it was first recorded.

    Stacked monitors may keep the same input in one node, and a refresh
    writes it again. An entry of another shape is left out, so a malformed
    write cannot break the thread. LangGraph calls a reducer with both values
    by position [@langgraph2026].
    """
    entries = [*read_run_input_entries(recorded), *read_run_input_entries(new)]
    latest = {entry["id"]: entry for entry in entries}
    return list(latest.values())


def read_run_inputs(state: object) -> tuple[RunInput, ...]:
    """Return the kept run inputs in a state, in order, leaving out any entry of another shape."""
    entries = state.get(RUN_INPUTS_KEY) if isinstance(state, Mapping) else None
    return tuple(read_run_input_entries(entries))


def read_rendered_source(entry: RunInput) -> str | None:
    """Return the `lc_source` a kept input is rendered with: none for the task author's words."""
    return None if entry["confirmed"] else UNCONFIRMED_INPUT_SOURCE


def is_rendered_form(message: BaseMessage, *, entry: RunInput) -> bool:
    """Tell whether a message under a kept input's id is that input as the monitor renders it."""
    return isinstance(message, HumanMessage) and read_message_source(message) == (
        read_rendered_source(entry)
    )


def build_kept_inputs(
    messages: Sequence[BaseMessage],
    *,
    input_ids: Collection[str],
    confirmed: bool,
) -> list[RunInput]:
    """Return a copy of each message with an id in `input_ids`, with the ids of those before it."""
    kept: list[RunInput] = []
    for index, message in enumerate(messages):
        if not message.id or message.id not in input_ids:
            continue
        earlier = messages[max(0, index - ANCHOR_COUNT) : index]
        kept.append(
            RunInput(
                id=message.id,
                text=message.text,
                previous_message_ids=[previous.id for previous in reversed(earlier) if previous.id],
                confirmed=confirmed,
            ),
        )
    return kept


def refresh_run_inputs(
    kept: Sequence[RunInput],
    *,
    messages: Sequence[BaseMessage],
) -> list[RunInput]:
    """Return the kept inputs, each with the text of its message where the state renders it."""
    by_id = {message.id: message for message in messages if message.id}
    refreshed: list[RunInput] = []
    for entry in kept:
        message = by_id.get(entry["id"])
        text = entry["text"]
        if message is not None and is_rendered_form(message, entry=entry):
            text = message.text
        refreshed.append(
            RunInput(
                id=entry["id"],
                text=text,
                previous_message_ids=entry["previous_message_ids"],
                confirmed=entry["confirmed"],
            ),
        )
    return refreshed


def find_changed_inputs(state: object) -> list[RunInput]:
    """Return each kept input whose message the state now holds with other text, refreshed."""
    kept = read_run_inputs(state)
    refreshed = refresh_run_inputs(kept, messages=read_state_messages(state))
    return [entry for entry, before in zip(refreshed, kept, strict=True) if entry != before]


def build_refresh_update(state: object) -> AgentStateUpdate:
    """Return the update that gives each kept input the text its message has in the state now."""
    changed = find_changed_inputs(state)
    return {RUN_INPUTS_KEY: changed} if changed else {}


def read_current_run_inputs(state: object) -> tuple[RunInput, ...]:
    """Return the kept run inputs with the text their messages have in the state now."""
    return tuple(refresh_run_inputs(read_run_inputs(state), messages=read_state_messages(state)))


def build_run_start_update(state: object) -> AgentStateUpdate:
    """Return the update a run starts with: its input recorded, and the text of each input kept.

    After a run that stopped early, the new input is kept unconfirmed, to go
    back as a note.
    """
    update = build_run_input_update(state)
    new_inputs = build_kept_inputs(
        read_state_messages(state),
        input_ids=find_unseen_human_message_ids(state),
        confirmed=not is_run_open(state),
    )
    kept = [*find_changed_inputs(state), *new_inputs]
    return {**update, RUN_INPUTS_KEY: kept} if kept else update


def build_input_message(entry: RunInput) -> HumanMessage:
    """Return a kept input as the monitor renders it: the task author's words, or a note."""
    message = HumanMessage(content=entry["text"], id=entry["id"])
    if entry["confirmed"]:
        return message
    return tag_as_context_note(message, source=UNCONFIRMED_INPUT_SOURCE)


def find_input_slot(entry: RunInput, *, index_by_id: Mapping[str, int]) -> int:
    """Return where a missing input goes, as the index of the message it goes before.

    It goes before the message that took its id, else just after the nearest
    message before it that is still there, else at the start.
    """
    index = index_by_id.get(entry["id"])
    if index is not None:
        return index
    for previous_id in entry["previous_message_ids"]:
        if previous_id in index_by_id:
            return index_by_id[previous_id] + 1
    return 0


def find_missing_input_slots(
    inputs: Sequence[RunInput],
    *,
    history: Sequence[BaseMessage],
    present_ids: Collection[str],
) -> dict[str, int]:
    """Return, for each input the history lacks, the index of the message it goes before.

    Each one comes after the input before it and before the next input the
    history holds, so the inputs keep their order.
    """
    index_by_id = {message.id: index for index, message in enumerate(history) if message.id}
    slots: dict[str, int] = {}
    lowest = 0
    for position, entry in enumerate(inputs):
        if entry["id"] in present_ids:
            lowest = max(lowest, index_by_id[entry["id"]] + 1)
            continue
        later_present = [
            index_by_id[later["id"]]
            for later in inputs[position + 1 :]
            if later["id"] in present_ids
        ]
        highest = min(later_present, default=len(history))
        slot = min(max(find_input_slot(entry, index_by_id=index_by_id), lowest), highest)
        slots[entry["id"]] = slot
        lowest = slot
    return slots


def replace_changed_inputs(
    history: Sequence[BaseMessage],
    *,
    inputs: Sequence[RunInput],
) -> list[BaseMessage]:
    """Return the history with each input it renders, but with other text, made verbatim."""
    entries = {entry["id"]: entry for entry in inputs}
    return [
        build_input_message(entries[message.id])
        if message.id in entries
        and is_rendered_form(message, entry=entries[message.id])
        and message.text != entries[message.id]["text"]
        else message
        for message in history
    ]


def find_present_input_ids(
    history: Sequence[BaseMessage],
    *,
    inputs: Sequence[RunInput],
) -> set[str]:
    """Return the ids of the inputs the history holds as the monitor renders them."""
    entries = {entry["id"]: entry for entry in inputs}
    return {
        message.id
        for message in history
        if message.id in entries and is_rendered_form(message, entry=entries[message.id])
    }


def restore_run_inputs(
    history: Sequence[BaseMessage],
    *,
    run_inputs: Sequence[RunInput],
    task_message_ids: Collection[str],
) -> tuple[BaseMessage, ...]:
    """Return the monitor's copy of a conversation with every kept input in it, verbatim.

    A confirmed input goes back as the task author's words only while its id
    is recorded as a run's input; an unconfirmed one always goes back as a
    note. One the history holds as the monitor renders it, with the kept
    text, is left where it is, so no input is read twice.
    """
    inputs = [
        entry for entry in run_inputs if not entry["confirmed"] or entry["id"] in task_message_ids
    ]
    replaced = replace_changed_inputs(history, inputs=inputs)
    present_ids = find_present_input_ids(replaced, inputs=inputs)
    slots = find_missing_input_slots(inputs, history=replaced, present_ids=present_ids)
    restored: list[BaseMessage] = []
    for index in range(len(replaced) + 1):
        restored.extend(
            build_input_message(entry) for entry in inputs if slots.get(entry["id"]) == index
        )
        restored.extend(replaced[index : index + 1])
    return tuple(restored)
