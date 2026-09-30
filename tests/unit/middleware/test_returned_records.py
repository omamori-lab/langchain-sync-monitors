"""The records a tool returns are checked where they are written, and wait there to be answered."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.errors import ParentCommand
from langgraph.types import Command, Overwrite

from langchain_sync_monitors._langchain import read_update_pairs
from langchain_sync_monitors.contracts import Delegation, StepRecord
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.returned_records import (
    check_command_records,
    check_parent_command_records,
    check_returned_records,
    read_tool_caller,
)
from langchain_sync_monitors.subagent_returns import SubagentReturn

LOGGER = "langchain_sync_monitors.returned_records"
OWN_STEP = StepRecord(
    agent="main",
    monitor="monitor",
    step_number=1,
    outcome="allowed",
    flagged=False,
    blocked_count=0,
    samples=[],
)


def build_record(**fields: Any) -> dict[str, Any]:
    record = {
        "agent": "worker",
        "monitor": "monitor",
        "step_number": 1,
        "outcome": "allowed",
        "flagged": False,
        "blocked_count": 0,
        "samples": [],
        "delegation_id": "call-task",
    }
    return {**record, **fields}


def build_tool_call(
    *, call_id: str | None = "call-task", subagent_type: str | None = None
) -> ToolCall:
    arguments = {"description": "Find the sources."}
    if subagent_type is not None:
        arguments["subagent_type"] = subagent_type
    return ToolCall(name="task", args=arguments, id=call_id, type="tool_call")


def build_state(
    *, delegation: Delegation | None = None, log: list[Any] | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {"messages": [], "monitor_log": [OWN_STEP] if log is None else log}
    if delegation is not None:
        state["monitor_delegation"] = delegation
    return state


def check(update: Any, *, state: dict[str, Any] | None = None, **call: Any) -> Command:
    caller = read_tool_caller(
        state or build_state(), agent="main", tool_call=build_tool_call(**call)
    )
    return check_command_records(Command(update=update), caller=caller)


def read_written(command: Command, *, key: str) -> list[Any]:
    return [value for pair_key, value in read_update_pairs(command) if pair_key == key]


def read_entry(command: Command) -> SubagentReturn:
    [[entry]] = read_written(command, key="monitor_subagent_returns")
    return entry


def test_a_subagent_s_records_reach_the_log_and_its_halt_and_blocks_are_stored() -> None:
    # Arrange
    halted = build_record(outcome="halted", blocked_count=2)
    other_label = build_record(monitor="guard", blocked_count=1)

    # Act
    checked = check({"monitor_log": [halted, other_label]}, subagent_type="worker")

    # Assert
    assert read_written(checked, key="monitor_log") == [[halted, other_label]]
    entry = read_entry(checked)
    assert entry["halted_subagents"] == ["worker"]
    assert entry["blocks"] == {"monitor": 2, "guard": 1}
    assert (entry["delegation_id"], entry["tool_call_id"], entry["answered"]) == (
        None,
        "call-task",
        False,
    )


def test_records_with_no_halt_and_no_block_store_nothing() -> None:
    # Act
    checked = check({"monitor_log": [build_record()]})

    # Assert
    assert read_written(checked, key="monitor_log") == [[build_record()]]
    assert read_written(checked, key="monitor_subagent_returns") == []


def test_a_command_that_writes_no_records_is_returned_as_it_is() -> None:
    # Arrange
    caller = read_tool_caller(build_state(), agent="main", tool_call=build_tool_call())
    command = Command(update={"messages": [ToolMessage("done", tool_call_id="call-task")]})

    # Act / Assert
    assert check_command_records(command, caller=caller) is command


@pytest.mark.parametrize(
    "overwrite",
    [
        lambda records: Overwrite(records),
        lambda records: {"__overwrite__": records},
        lambda records: {"type": "__overwrite__", "value": records},
    ],
    ids=["typed", "dict", "json"],
)
def test_an_overwrite_of_the_log_is_read_as_an_append(
    overwrite: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    halted = build_record(outcome="halted")

    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": overwrite([halted])})

    # Assert
    assert read_written(checked, key="monitor_log") == [[halted]]
    assert read_entry(checked)["halted_subagents"] == ["worker"]
    assert "Overwrite" in caplog.text


def test_an_empty_overwrite_erases_nothing(caplog: pytest.LogCaptureFixture) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": Overwrite([])})

    # Assert
    assert read_update_pairs(checked) == []
    assert "Overwrite" in caplog.text


UNREADABLE_WRITES = {
    "negative-count": [build_record(blocked_count=-100)],
    "string-count": [build_record(blocked_count="2")],
    "no-agent": [{key: value for key, value in build_record().items() if key != "agent"}],
    "not-a-list": "halted",
    "one-record-not-in-a-list": {"agent": "worker"},
}


@pytest.mark.parametrize("value", UNREADABLE_WRITES.values(), ids=UNREADABLE_WRITES.keys())
def test_an_unreadable_record_is_kept_out_of_the_log_and_counts_as_a_halt(
    value: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": value}, subagent_type="worker")

    # Assert
    assert read_written(checked, key="monitor_log") == []
    assert read_entry(checked)["halted_subagents"] == ["worker"]
    assert "not a step record the monitor can read" in caplog.text


def test_an_unreadable_record_leaves_the_readable_ones_in_the_log() -> None:
    # Arrange
    readable = build_record(blocked_count=1)

    # Act
    checked = check({"monitor_log": [build_record(step_number=-1), readable]})

    # Assert
    assert read_written(checked, key="monitor_log") == [[readable]]
    entry = read_entry(checked)
    assert entry["halted_subagents"] == ["that the task call call-task started"]
    assert entry["blocks"] == {"monitor": 1}


@pytest.mark.parametrize(
    ("call_id", "halted_name"),
    [("call-task", "that the task call call-task started"), (None, "that a task call started")],
    ids=["with-an-id", "without-an-id"],
)
def test_a_halt_claiming_the_caller_s_own_step_is_kept_out_of_the_log_and_counts(
    call_id: str | None,
    halted_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the caller is main, started by no monitored agent; a subagent that got no
    # delegation and shares the name writes exactly this
    forged = build_record(agent="main", outcome="halted", blocked_count=2)
    del forged["delegation_id"]

    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": [forged]}, call_id=call_id)

    # Assert
    assert read_written(checked, key="monitor_log") == []
    entry = read_entry(checked)
    assert entry["halted_subagents"] == [halted_name]
    assert entry["blocks"] == {}
    assert "claims to be a step of the agent 'main' itself" in caplog.text


def test_a_step_claiming_the_caller_s_own_that_is_no_halt_is_dropped_and_stores_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: step 2, since a copy of step 1, which the log holds, would be a write-back
    forged = build_record(agent="main", outcome="allowed", step_number=2)
    del forged["delegation_id"]

    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": [forged]})

    # Assert
    assert read_update_pairs(checked) == []
    assert "claims to be a step of the agent 'main' itself" in caplog.text


def test_a_same_named_subagent_s_record_is_kept_and_named_by_the_call() -> None:
    # Arrange: a fork records under main, with its own delegation
    fork_halt = build_record(agent="main", outcome="halted", delegation_id="call-fork")

    # Act
    checked = check({"monitor_log": [fork_halt]}, call_id="call-fork", subagent_type="forker")

    # Assert
    assert read_written(checked, key="monitor_log") == [[fork_halt]]
    assert read_entry(checked)["halted_subagents"] == ["forker"]


def test_a_call_reusing_the_caller_s_delegation_id_with_its_own_records_raises() -> None:
    # Arrange: a subagent named main, started by a call that reuses the id of main's own
    delegation = Delegation(tool_call_id="call_0", delegating_agent="main", blocks_before={})
    nested = build_record(agent="main", outcome="halted", delegation_id="call_0")

    # Act / Assert
    with pytest.raises(ConfigurationError, match="cannot tell a subagent's steps"):
        check(
            {"monitor_log": [nested]},
            state=build_state(delegation=delegation),
            call_id="call_0",
        )


def test_a_call_reusing_the_caller_s_delegation_id_keeps_another_agent_s_records() -> None:
    # Arrange
    delegation = Delegation(tool_call_id="call_0", delegating_agent="main", blocks_before={})
    nested = build_record(agent="inner", outcome="halted", delegation_id="call_0")

    # Act
    checked = check(
        {"monitor_log": [nested]},
        state=build_state(delegation=delegation),
        call_id="call_0",
    )

    # Assert
    assert read_written(checked, key="monitor_log") == [[nested]]
    assert read_entry(checked)["delegation_id"] == "call_0"


def test_a_state_written_back_whole_adds_only_the_records_after_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    earlier_halt = build_record(outcome="halted", delegation_id="call-earlier")
    log = [OWN_STEP, earlier_halt]
    new = build_record(blocked_count=1)

    # Act
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        checked = check({"monitor_log": [*log, new]}, state=build_state(log=log))

    # Assert
    assert read_written(checked, key="monitor_log") == [[new]]
    assert read_entry(checked)["halted_subagents"] == []
    assert caplog.records == []


def test_records_that_start_with_part_of_the_log_are_all_checked() -> None:
    # Arrange: a subagent's records never start with the whole log, which holds main's step
    log = [OWN_STEP, build_record(outcome="halted")]

    # Act
    checked = check({"monitor_log": [build_record(outcome="halted")]}, state=build_state(log=log))

    # Assert
    assert read_written(checked, key="monitor_log") == [[build_record(outcome="halted")]]
    assert read_entry(checked)["halted_subagents"] == ["worker"]


@dataclasses.dataclass
class RecordsUpdate:
    """An update as a dataclass, which LangGraph reads as pairs."""

    monitor_log: list[dict[str, Any]]
    messages: list[ToolMessage]


def test_the_records_of_every_update_shape_are_checked() -> None:
    # Arrange
    report = ToolMessage("done", tool_call_id="call-task")
    update = RecordsUpdate(monitor_log=[build_record(blocked_count=-1)], messages=[report])

    # Act
    checked = check(update)

    # Assert
    assert read_written(checked, key="messages") == [[report]]
    assert read_written(checked, key="monitor_log") == []
    assert read_entry(checked)["halted_subagents"] == ["that the task call call-task started"]


def test_a_command_for_the_parent_graph_is_left_to_the_parent_s_monitor() -> None:
    # Arrange
    caller = read_tool_caller(build_state(), agent="main", tool_call=build_tool_call())
    command = Command(graph=Command.PARENT, update={"monitor_log": [build_record(step_number=-1)]})

    # Act / Assert
    assert check_command_records(command, caller=caller) is command


def test_the_records_in_a_parent_command_a_call_raises_are_checked_in_place() -> None:
    # Arrange: LangGraph names the graph by the time the parent's monitor sees it
    caller = read_tool_caller(build_state(), agent="main", tool_call=build_tool_call())
    bubble = ParentCommand(
        Command(graph="parent", update={"monitor_log": [build_record(outcome="halted")]})
    )

    # Act
    check_parent_command_records(bubble, caller=caller)

    # Assert
    [command] = bubble.args
    assert command.graph == "parent"
    assert read_entry(command)["halted_subagents"] == ["worker"]


def test_each_command_of_a_list_result_is_checked() -> None:
    # Arrange
    caller = read_tool_caller(build_state(), agent="main", tool_call=build_tool_call())
    message = ToolMessage("done", tool_call_id="call-task")
    results = [message, Command(update={"monitor_log": [build_record(outcome="halted")]})]

    # Act
    checked = check_returned_records(results, caller=caller)

    # Assert
    assert isinstance(checked, list)
    assert checked[0] is message
    assert isinstance(checked[1], Command)
    assert read_entry(checked[1])["halted_subagents"] == ["worker"]
