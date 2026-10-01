"""The subagent halts and blocks a tool call returned wait for the agent's next step, then go."""

from __future__ import annotations

import pytest

from langchain_sync_monitors.contracts import Delegation
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.subagent_returns import (
    SubagentReturn,
    build_answered_update,
    count_returned_blocks,
    find_halted_subagents,
    merge_subagent_returns,
    read_subagent_returns,
)


def build_entry(
    *,
    entry_id: str,
    delegation_id: str | None = None,
    halted: list[str] | None = None,
    blocks: dict[str, int] | None = None,
    answered: bool = False,
) -> SubagentReturn:
    return SubagentReturn(
        id=entry_id,
        delegation_id=delegation_id,
        tool_call_id="call-task",
        halted_subagents=halted or [],
        blocks=blocks or {},
        answered=answered,
    )


def test_the_store_adds_entries_and_removes_the_ones_a_step_answered() -> None:
    # Arrange
    first, second = build_entry(entry_id="a"), build_entry(entry_id="b")
    stored = merge_subagent_returns([], [first, second])

    # Act
    merged = merge_subagent_returns(stored, [{**first, "answered": True}])

    # Assert
    assert merged == [second]


def test_an_agent_reads_only_the_entries_of_its_own_delegation() -> None:
    # Arrange: a fork starts with its parent's private state
    parent_entry = build_entry(entry_id="a", halted=["worker"], blocks={"monitor": 3})
    fork_entry = build_entry(entry_id="b", delegation_id="call-fork", blocks={"monitor": 1})
    delegation = Delegation(tool_call_id="call-fork", delegating_agent="main", blocks_before={})
    state = {
        "monitor_subagent_returns": [parent_entry, fork_entry],
        "monitor_delegation": delegation,
    }

    # Act
    returns = read_subagent_returns(state)

    # Assert
    assert returns == [fork_entry]
    assert find_halted_subagents(returns) == []
    assert count_returned_blocks(returns, monitor="monitor") == 1


def test_a_step_answers_only_its_own_delegation_s_entries() -> None:
    # Arrange
    parent_entry = build_entry(entry_id="a", halted=["worker"])
    own_entry = build_entry(entry_id="b", delegation_id="call-fork", halted=["reviewer"])
    delegation = Delegation(tool_call_id="call-fork", delegating_agent="main", blocks_before={})
    state = {
        "monitor_subagent_returns": [parent_entry, own_entry],
        "monitor_delegation": delegation,
    }

    # Act
    update = build_answered_update(state)

    # Assert
    assert update == {"monitor_subagent_returns": [{**own_entry, "answered": True}]}


def test_halted_subagents_are_named_once_in_order() -> None:
    # Arrange
    returns = [
        build_entry(entry_id="a", halted=["worker", "reviewer"]),
        build_entry(entry_id="b", halted=["worker"]),
    ]

    # Act / Assert
    assert find_halted_subagents(returns) == ["worker", "reviewer"]


def test_an_entry_the_monitor_cannot_read_raises() -> None:
    # Arrange
    state = {"monitor_subagent_returns": [{"id": "a", "halted_subagents": "worker"}]}

    # Act
    with pytest.raises(MonitorError) as raised:
        read_subagent_returns(state)

    # Assert: the fields at fault are named, the value only by its type and length, and
    # pydantic's error, which quotes the value, is neither chained nor shown as the context
    assert str(raised.value) == (
        "monitor_subagent_returns holds a value the monitor cannot read "
        "(0.delegation_id: Field required; 0.tool_call_id: Field required; "
        "0.halted_subagents: Input should be a valid list): a list of length 1. "
        "Only the monitor writes this key; remove the value."
    )
    assert "worker" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize(
    "state",
    [{}, {"monitor_subagent_returns": None}, "not a state"],
    ids=["no-entries", "entries-none", "not-a-mapping"],
)
def test_a_state_without_entries_answers_nothing(state: object) -> None:
    # Arrange: the parametrised state holds no entries, or is no mapping at all

    # Act
    returns = read_subagent_returns(state)
    update = build_answered_update(state)

    # Assert
    assert returns == []
    assert update == {}


def test_an_entry_whose_count_is_a_string_raises_rather_than_being_converted() -> None:
    # Arrange
    entry = {**build_entry(entry_id="a"), "blocks": {"monitor": "3"}}

    # Act / Assert
    with pytest.raises(MonitorError, match="monitor_subagent_returns"):
        read_subagent_returns({"monitor_subagent_returns": [entry]})
