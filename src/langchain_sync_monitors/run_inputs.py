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
  one that stopped early, is recorded as no one's and is not kept.
- Only the text is kept, which is what the judge reads, so the state grows
  by the text of every run's input and no more.
- An input the state still holds untagged under its id, with other text, at
  the start of a run that follows a finished one, is kept with its new text:
  the user edited it between runs, as `update_state` can. A middleware that
  rewrites it under its id during a run is trusted code, and its version is
  taken the same way at the next run's start.
- Before a monitor judges a step, each kept input its copy of the
  conversation lacks is put back as an untagged human message. One the copy
  holds under its id with other text, such as the preview Deep Agents shows
  the agent in place of a large message [@deepagents2026], is replaced by the
  kept text. The agent's own request is never changed.

A kept input goes back just before the message that took its id, else just
after the message it followed when it was recorded, else at the start, just
before the summary that replaced it and its neighbours. Either way it comes
after the input before it and before the next one the conversation holds.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.task_authorship import (
    build_run_input_update,
    find_unseen_human_message_ids,
    find_untagged_human_message_ids,
    is_run_open,
    is_untagged_human_message,
    read_state_messages,
)

RUN_INPUTS_KEY = "monitor_run_inputs"
"""The state key that holds the text of every human message recorded as a run's input."""


class RunInput(TypedDict):
    """The kept copy of one human message recorded as a run's input.

    `text` is the message's text, which is what the judge reads.
    `previous_message_id` is the id of the message it followed when it was
    recorded, or None when it opened the thread.
    """

    id: str
    text: str
    previous_message_id: str | None


def merge_run_inputs(  # lanorme: ignore[KWARG-001]
    recorded: list[RunInput],
    new: list[RunInput],
) -> list[RunInput]:
    """Keep one copy of each input: its latest text, in the place it was first recorded.

    Stacked monitors may keep the same input in one node, and an edit writes
    it again. LangGraph calls a reducer with both values by position
    [@langgraph2026].
    """
    latest = {entry["id"]: entry for entry in [*recorded, *new]}
    return list(latest.values())


def is_run_input(value: object) -> bool:
    """Tell whether a value from the state has the shape of a kept run input."""
    if not isinstance(value, Mapping):
        return False
    previous_message_id = value.get("previous_message_id")
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("text"), str)
        and (previous_message_id is None or isinstance(previous_message_id, str))
    )


def read_run_inputs(state: object) -> tuple[RunInput, ...]:
    """Return the kept run inputs in a state, in order, leaving out any entry of another shape."""
    entries = state.get(RUN_INPUTS_KEY) if isinstance(state, Mapping) else None
    if not isinstance(entries, list):
        return ()
    return tuple(
        RunInput(
            id=entry["id"],
            text=entry["text"],
            previous_message_id=entry.get("previous_message_id"),
        )
        for entry in entries
        if is_run_input(entry)
    )


def build_kept_inputs(
    messages: Sequence[BaseMessage],
    *,
    input_ids: Collection[str],
) -> list[RunInput]:
    """Return a copy of each message recorded as a run's input, with the id of the one before it."""
    previous_ids = [None, *(message.id for message in messages[:-1])]
    return [
        RunInput(id=message.id, text=message.text, previous_message_id=previous_id)
        for message, previous_id in zip(messages, previous_ids, strict=True)
        if message.id and message.id in input_ids
    ]


def build_edited_inputs(
    messages: Sequence[BaseMessage],
    *,
    kept: Sequence[RunInput],
) -> list[RunInput]:
    """Return each kept input the state holds untagged under its id, with its new text."""
    texts = {
        message.id: message.text
        for message in messages
        if is_untagged_human_message(message) and message.id
    }
    return [
        RunInput(
            id=entry["id"],
            text=texts[entry["id"]],
            previous_message_id=entry["previous_message_id"],
        )
        for entry in kept
        if entry["id"] in texts and texts[entry["id"]] != entry["text"]
    ]


def build_run_start_update(state: object) -> AgentStateUpdate:
    """Return the update a run starts with: its input recorded, and the text of each input kept.

    A run that follows one that stopped early confirms no input, so it keeps
    nothing and takes no edit.
    """
    update = build_run_input_update(state)
    if is_run_open(state):
        return update
    messages = read_state_messages(state)
    kept = [
        *build_edited_inputs(messages, kept=read_run_inputs(state)),
        *build_kept_inputs(messages, input_ids=find_unseen_human_message_ids(state)),
    ]
    return {**update, RUN_INPUTS_KEY: kept} if kept else update


def build_input_message(entry: RunInput) -> HumanMessage:
    """Return a kept input as the untagged human message the judge reads."""
    return HumanMessage(content=entry["text"], id=entry["id"])


def find_input_slot(entry: RunInput, *, index_by_id: Mapping[str, int]) -> int:
    """Return where a missing input goes, as the index of the message it goes before.

    It goes before the message that took its id, else just after the one it
    followed, else at the start.
    """
    index = index_by_id.get(entry["id"])
    if index is not None:
        return index
    previous_message_id = entry["previous_message_id"]
    if previous_message_id is not None and previous_message_id in index_by_id:
        return index_by_id[previous_message_id] + 1
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
    """Return the history with each input it holds untagged, but with other text, made verbatim."""
    entries = {entry["id"]: entry for entry in inputs}
    return [
        build_input_message(entries[message.id])
        if is_untagged_human_message(message)
        and message.id in entries
        and message.text != entries[message.id]["text"]
        else message
        for message in history
    ]


def restore_run_inputs(
    history: Sequence[BaseMessage],
    *,
    run_inputs: Sequence[RunInput],
    task_message_ids: Collection[str],
) -> tuple[BaseMessage, ...]:
    """Return the monitor's copy of a conversation with every kept run input in it, verbatim.

    Only an input whose id is recorded as a run's input is put back. One the
    history holds untagged under its id, with the kept text, is left where it
    is, so no input is read twice.
    """
    inputs = [entry for entry in run_inputs if entry["id"] in task_message_ids]
    replaced = replace_changed_inputs(history, inputs=inputs)
    present_ids = find_untagged_human_message_ids(replaced)
    slots = find_missing_input_slots(inputs, history=replaced, present_ids=present_ids)
    restored: list[BaseMessage] = []
    for index in range(len(replaced) + 1):
        restored.extend(
            build_input_message(entry) for entry in inputs if slots.get(entry["id"]) == index
        )
        restored.extend(replaced[index : index + 1])
    return tuple(restored)
