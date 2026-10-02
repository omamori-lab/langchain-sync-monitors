"""The text of every run's input, kept so the monitor reads each one after it leaves the history.

Every human message recorded as a run's input, under `TASK_MESSAGES_KEY`,
reaches the monitor as the task author's words, verbatim, even once the model
request no longer holds it: after LangChain's `SummarizationMiddleware`
replaces earlier messages in the state, Deep Agents replaces them in the
request [@langchain2026; @deepagents2026], or a tool removes it by id
[@langgraph2026]. The inputs put back keep their order. The summary stays a
context note beside the kept turns, since the agent's model wrote it.

- Each new input's text is kept under `RUN_INPUTS_KEY` at the start of its
  run, in the update that records its id: the one hook sure to see it, since
  a summariser listed before the monitor runs its `before_model` hook first.
  Input a run cannot confirm, after one that stopped early, is kept too, and
  goes back as a note from `unconfirmed_input`, never as a run's input.
- Only the text is kept, which is what the monitor reads, so the state grows
  by the text of every run's input, and a turn put back holds no image.
- A kept input follows its message in the state: whenever the state holds it
  under its id as the monitor renders it, the kept text becomes its text, at
  the start of a run and of each step, at each commit, and in memory before
  each judgement. So a redaction such as `PIIMiddleware`'s [@langchain2026],
  or the user's `update_state` edit, reaches the monitor as the agent reads it.
  A tool cannot write such a message: `task_authorship` tags the human
  messages a tool writes, and drops its writes to this key.
- Before a monitor judges a step, each kept input its copy lacks is put
  back, and one it holds under its id with other text, such as the preview
  Deep Agents shows the agent of a large message [@deepagents2026], is made
  verbatim. The agent's own request is never changed, so a rewrite made in
  the request alone, not in the state, is not followed.

A kept input goes back just after the nearest of the three messages before
it still there, else before a message that took its id, unless a tool wrote
it (`task_authorship` records those ids, since LangGraph adds a message under
an absent id at the end), else at the start, before the summary. It comes
before the next input still there and any message under its id, and after
the input before it, which wins where they disagree, so the inputs put back
keep their order. Inputs the state holds are read where they stand: a tool
that reorders them, writing the history back in a new order or removing an
input and writing it back in a parallel call, reorders them for the monitor.

The monitor reads every input whole on every step, even one Deep Agents shows
the agent only as a preview, over 50,000 tokens by default, so a very large
input costs its full size at each step and can exceed the context of a
small monitor model, and the step then fails. One case stays open: with a redacting
middleware and a summariser both listed before the monitor, one pass can
redact a turn and summarise it away before any monitor hook sees the
redaction, and the monitor then reads the turn as it arrived.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.message_ids import (
    assign_human_message_ids,
    build_assigned_ids_update,
    replace_state_messages,
)
from langchain_sync_monitors.state_keys import RUN_INPUTS_KEY
from langchain_sync_monitors.task_authorship import (
    UNCONFIRMED_INPUT_SOURCE,
    build_run_input_update,
    find_unseen_human_message_ids,
    is_run_open,
    read_state_messages,
    tag_as_context_note,
)
from langchain_sync_monitors.transcript import read_message_source

PREVIOUS_MESSAGE_COUNT = 3
"""How many of the messages before an input are kept as the places it can go back to."""


class RunInput(TypedDict):
    """The kept copy of one human message a run received as its input.

    `text` is what the monitor reads. `previous_message_ids` holds the ids of up
    to `PREVIOUS_MESSAGE_COUNT` messages before it when recorded, nearest first.
    `confirmed` is false for input a run could not confirm, which goes back as
    a note from `unconfirmed_input`.
    """

    id: str
    text: str
    previous_message_ids: list[str]
    confirmed: bool


def is_run_input(value: object) -> bool:
    """Tell whether a value from the state has the shape of a kept run input."""
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("text"), str)
        and isinstance(value.get("confirmed"), bool)
        and is_id_list(value.get("previous_message_ids"))
    )


def is_id_list(value: object) -> bool:
    """Tell whether a value from the state is a list of message ids."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


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

    Stacked monitors and refreshes write an input again, and an entry of
    another shape is left out. LangGraph calls a reducer by position [@langgraph2026].
    """
    entries = [*read_run_input_entries(recorded), *read_run_input_entries(new)]
    latest = {entry["id"]: entry for entry in entries}
    return list(latest.values())


def read_run_inputs(state: object) -> tuple[RunInput, ...]:
    """Return the kept run inputs in a state, in order, leaving out any entry of another shape."""
    entries = state.get(RUN_INPUTS_KEY) if isinstance(state, Mapping) else None
    return tuple(read_run_input_entries(entries))


def is_rendered_form(message: BaseMessage, *, entry: RunInput) -> bool:
    """Tell whether a message under a kept input's id is that input as the monitor renders it.

    The task author's words carry no `lc_source`; unconfirmed input carries its own.
    """
    source = None if entry["confirmed"] else UNCONFIRMED_INPUT_SOURCE
    return isinstance(message, HumanMessage) and read_message_source(message) == source


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
        earlier = messages[max(0, index - PREVIOUS_MESSAGE_COUNT) : index]
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
        refreshed.append({**entry, "text": text})
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
    back as a note. Input that reached the state without an id, as a string
    or a `(role, text)` tuple does in a Deep Agent, is first given one, and
    the history is written back with it, as `assign_human_message_ids` says.
    """
    messages_with_ids = assign_human_message_ids(state, as_notes=False)
    if messages_with_ids is not None:
        state = replace_state_messages(state, messages=messages_with_ids)
    update = build_run_input_update(state)
    new_inputs = build_kept_inputs(
        read_state_messages(state),
        input_ids=find_unseen_human_message_ids(state),
        confirmed=not is_run_open(state),
    )
    kept = [*find_changed_inputs(state), *new_inputs]
    update = {**update, RUN_INPUTS_KEY: kept} if kept else update
    if messages_with_ids is None:
        return update
    return build_assigned_ids_update(messages_with_ids, update=update)


def build_input_message(entry: RunInput) -> HumanMessage:
    """Return a kept input as the monitor renders it: the task author's words, or a note.

    It keeps its input's id, not a fresh `monitor-` one: it never reaches the
    state, and the rest of the monitor's copy is read against that id.
    """
    message = HumanMessage(content=entry["text"], id=entry["id"])
    if entry["confirmed"]:
        return message
    return tag_as_context_note(message, source=UNCONFIRMED_INPUT_SOURCE)


def find_input_insertion_point(
    entry: RunInput,
    *,
    index_by_id: Mapping[str, int],
    rewritten_input_ids: Collection[str],
) -> int:
    """Return where a missing input goes, as the index of the message it goes before.

    It goes just after the nearest message before it that is still there,
    else just before a message that took its id, unless a tool wrote that
    message, else at the start.
    """
    for previous_id in entry["previous_message_ids"]:
        if previous_id in index_by_id:
            return index_by_id[previous_id] + 1
    return 0 if entry["id"] in rewritten_input_ids else index_by_id.get(entry["id"], 0)


def find_missing_input_insertion_points(
    inputs: Sequence[RunInput],
    *,
    history: Sequence[BaseMessage],
    present_ids: Collection[str],
    rewritten_input_ids: Collection[str],
) -> dict[str, int]:
    """Return, for each input the history lacks, the index of the message it goes before.

    Each one comes after the input before it, and before both the next input
    the history holds and a message that took its id, which was written after
    it. Where the two bounds cross, the lower one wins, so the inputs put back
    keep their order among themselves and with the ones the history holds.
    """
    index_by_id = {message.id: index for index, message in enumerate(history) if message.id}
    insertion_points: dict[str, int] = {}
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
        # A message under its id came after it.
        highest = min([*later_present, index_by_id.get(entry["id"], len(history))])
        candidate = find_input_insertion_point(
            entry, index_by_id=index_by_id, rewritten_input_ids=rewritten_input_ids
        )
        insertion_point = max(min(candidate, highest), lowest)
        insertion_points[entry["id"]] = insertion_point
        lowest = insertion_point
    return insertion_points


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
    rewritten_input_ids: Collection[str] = frozenset(),
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
    insertion_points = find_missing_input_insertion_points(
        inputs, history=replaced, present_ids=present_ids, rewritten_input_ids=rewritten_input_ids
    )
    restored: list[BaseMessage] = []
    for index in range(len(replaced) + 1):
        restored.extend(
            build_input_message(entry)
            for entry in inputs
            if insertion_points.get(entry["id"]) == index
        )
        restored.extend(replaced[index : index + 1])
    return tuple(restored)
