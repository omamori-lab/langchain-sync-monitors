"""The subagent halts and blocks an agent's tool calls returned, kept until its next step.

`returned_records` checks the records a tool call returns where they are
written, and stores what they hold under `SUBAGENT_RETURNS_KEY`, a private
key no tool can write: one `SubagentReturn` per call whose records show a
subagent halted or hold blocks. The agent's next step reads them there,
rather than from where records sit in `monitor_log`, where a tool's own
records could hide them: its halt decision reads the halted subagents, and
Auto Mode the new subagent blocks. When the step commits, it marks them
answered, and the store removes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from pydantic import TypeAdapter, ValidationError

from langchain_sync_monitors._langchain import AgentStateUpdate, read_delegation
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.records import describe_record_error, render_value
from langchain_sync_monitors.state_keys import SUBAGENT_RETURNS_KEY


class SubagentReturn(TypedDict):
    """The halts and blocks one tool call returned, for the calling agent's next step to answer.

    `id` is the entry's own, so the step that answers it removes it.
    `delegation_id` is the calling agent's own delegation, or None in an
    agent no monitored agent started: a forked subagent starts with its
    parent's private state [@deepagents2026], and must not answer its
    parent's returns. `tool_call_id` is the call's id, `halted_subagents`
    names each subagent the call's records show halted, and `blocks` holds
    the blocks they record by monitor label. `answered` marks the copy with
    which a step removes the entry.
    """

    id: str
    delegation_id: str | None
    tool_call_id: str | None
    halted_subagents: list[str]
    blocks: dict[str, int]
    answered: bool


RETURNS_ADAPTER = TypeAdapter(list[SubagentReturn])
"""Checks the entries read from the state, which only the monitor writes."""


def merge_subagent_returns(  # lanorme: ignore[KWARG-001]
    recorded: list[SubagentReturn],
    new: list[SubagentReturn],
) -> list[SubagentReturn]:
    """Add the new entries, and remove every entry a step marks answered.

    LangGraph calls a reducer with both values by position [@langgraph2026].
    """
    answered = {entry["id"] for entry in new if entry["answered"]}
    return [
        entry
        for entry in [*recorded, *new]
        if not entry["answered"] and entry["id"] not in answered
    ]


def read_stored_returns(state: object) -> list[SubagentReturn]:
    """Return every entry in the state, raising `MonitorError` for a value that does not fit.

    Skipping an entry could hide a halt, so none is skipped. The error names
    the value and the fields at fault, and is raised from None, since
    pydantic's own error quotes the value, which the traceback would print.
    """
    value = state.get(SUBAGENT_RETURNS_KEY) if isinstance(state, Mapping) else None
    if value is None:
        return []
    try:
        return RETURNS_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        message = (
            f"{SUBAGENT_RETURNS_KEY} holds a value the monitor cannot read "
            f"({describe_record_error(error)}): {render_value(value)}. Only the monitor "
            "writes this key; remove the value."
        )
        raise MonitorError(message) from None


def read_subagent_returns(state: object) -> list[SubagentReturn]:
    """Return the entries this agent's next step has to answer, in the order they were stored.

    Only the entries of this agent's own delegation count.
    """
    delegation = read_delegation(state)
    delegation_id = None if delegation is None else delegation["tool_call_id"]
    return [
        entry for entry in read_stored_returns(state) if entry["delegation_id"] == delegation_id
    ]


def build_answered_update(state: object) -> AgentStateUpdate:
    """Return the update with which a committed step removes the entries it answered."""
    returns = read_subagent_returns(state)
    if not returns:
        return {}
    return {SUBAGENT_RETURNS_KEY: [{**entry, "answered": True} for entry in returns]}


def find_halted_subagents(returns: Sequence[SubagentReturn]) -> list[str]:
    """Return the names of the subagents the entries show halted, each once, in order."""
    return list(dict.fromkeys(name for entry in returns for name in entry["halted_subagents"]))


def count_returned_blocks(returns: Sequence[SubagentReturn], *, monitor: str) -> int:
    """Return the blocks one monitor label recorded in the entries' records."""
    return sum(entry["blocks"].get(monitor, 0) for entry in returns)
