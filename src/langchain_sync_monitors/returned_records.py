"""The records and trajectories a tool call writes, checked where they are written.

Deep Agents' `task` tool returns a subagent's records through `monitor_log`,
and its trajectories through `monitor_trajectory` [@deepagents2026], so
those are the monitor keys a tool may write. The monitor checks every write
to them, in every update shape LangGraph accepts and in a `ParentCommand`,
before LangGraph writes it. A record a tool writes to `monitor_log` is
checked as follows:

- A write that replaces the log, an `Overwrite` in any of its forms, would
  erase the thread's blocks and its audit evidence. It is read as an append
  of the records it holds, with a warning.
- A write that starts with the whole log the state holds, as when a tool
  writes the whole state back, has that log left out, since its records
  are there already. Records of the log written back any other way, in
  part or out of order, are added again, so their blocks and halts count
  twice, which fails closed.
- A record that is not a whole `StepRecord` with counts of zero or more is
  kept out of the log, where it would make every later step raise, and the
  call counts as a halted subagent, so a halt the monitor cannot read still
  counts. A warning names the record by the fields that identify it, never
  by the text its samples hold.
- A record that claims to be a step of the calling agent itself, with its
  name and its delegation, is kept out of the log with a warning: only the
  agent's own monitor records its steps. A subagent that received no
  delegation, from a tool that did not pass it the state, and shares the
  agent's name records such steps too, so a halt among them still counts
  as a halted subagent, though its blocks are lost to Auto Mode's total.
  When the call reuses the id of the call that started the calling agent,
  a subagent that shares the agent's name records exactly such steps, which
  the monitor cannot tell from the agent's own, so it raises
  `ConfigurationError`.

A write to `monitor_trajectory` is kept, for `trajectory_channel` to join
under the call, only as a mapping of one or more monitor labels to the
trajectory states of a subagent the call itself started, each once, with no
lost write, and whose trajectory loads. Anything else, an `Overwrite`, a delta
or an empty mapping included, is dropped with a warning and counts as a
halted subagent, since the subagent's steps would otherwise be missing from
the run. A state under a label the calling agent has no trajectory for, as a
compiled subagent's monitor of another label returns, is dropped with a
warning alone; its steps stay in `monitor_log`.

The halts in the records kept, with a halt for each record the monitor could
not read or kept out as the caller's own and for each trajectory dropped as
above, and the blocks the records kept hold by monitor label, go to
`subagent_returns`, which keeps them until the calling agent's next step.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast
from uuid import uuid4

from hyoshigi import FormatError, Trajectory
from langchain_core.messages import ToolCall
from langgraph.errors import ParentCommand
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    ToolCallResult,
    ToolCallResults,
    UpdatePairs,
    UpdateValue,
    read_delegation,
    read_overwrite,
    read_update_pairs,
    replace_update_pairs,
)
from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.records import (
    count_blocks_by_monitor,
    is_own_record,
    render_value,
    validate_step_record,
)
from langchain_sync_monitors.state_keys import SUBAGENT_RETURNS_KEY, TRAJECTORY_KEY
from langchain_sync_monitors.subagent_returns import SubagentReturn
from langchain_sync_monitors.trajectory_channel import (
    TrajectoryState,
    read_labels,
    read_returned_state,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCaller:
    """The agent whose tool call wrote records, as the check of those records reads it.

    `agent` is the agent's name and `delegation_id` its own delegation,
    `state` the agent's state the call ran in, `log` its `monitor_log` as it
    held it, and `tool_call` the call itself.
    """

    agent: str
    delegation_id: str | None
    state: object
    log: Sequence[object]
    tool_call: ToolCall

    def describe_subagent(self) -> str:
        """Name the subagent the call started: by Deep Agents' `subagent_type`, else by the call.

        A subagent that shares the agent's name, such as a fork, is named by
        the call rather than by the name its records carry.
        """
        subagent_type = self.tool_call["args"].get("subagent_type")
        if isinstance(subagent_type, str) and subagent_type.strip():
            return subagent_type
        if self.tool_call["id"] is None:
            return f"that a {self.tool_call['name']} call started"
        return f"that the {self.tool_call['name']} call {self.tool_call['id']} started"

    def name_halted_subagent(self, record: StepRecord) -> str:
        """Name a halted record's subagent by the record, unless it carries this agent's name.

        Such a record is the subagent's that the call started, named by the
        call, or, when it names another delegation, that of a subagent nested
        deeper, named by its own name and the call that started it.
        """
        if record["agent"] != self.agent:
            return record["agent"]
        delegation_id = record.get("delegation_id")
        if delegation_id is None or delegation_id == self.tool_call["id"]:
            return self.describe_subagent()
        return f"{record['agent']} that the call {delegation_id} started"

    def is_caller_record(self, record: StepRecord) -> bool:
        """Tell whether a record claims to be a step of the calling agent itself."""
        return is_own_record(record, agent=self.agent, delegation_id=self.delegation_id)

    def has_own_delegation_id(self) -> bool:
        """Tell whether the call carries the id of the call that started the calling agent."""
        return self.delegation_id is not None and self.tool_call["id"] == self.delegation_id

    def holds_trajectory(self, label: str) -> bool:
        """Tell whether the calling agent's state holds a trajectory under the monitor label."""
        return isinstance(self.state, Mapping) and label in read_labels(self.state)


@dataclass(slots=True, kw_only=True)
class ReturnedRecords:
    """What the records one command writes to `monitor_log` hold, as they are checked.

    `kept` holds the records that reach the log, `trajectories` the
    trajectory states that reach `monitor_trajectory`, by label, and
    `halted_subagents` the names of the halted subagents, a record the
    monitor could not read or a trajectory it dropped included.
    """

    kept: list[StepRecord] = field(default_factory=list)
    trajectories: dict[str, TrajectoryState] = field(default_factory=dict)
    halted_subagents: list[str] = field(default_factory=list)

    def build_pairs(self, *, caller: ToolCaller) -> list[tuple[str, UpdateValue]]:
        """Return the writes of what was kept, and of the entry for the agent's next step."""
        pairs: list[tuple[str, UpdateValue]] = []
        if self.kept:
            pairs.append((MONITOR_LOG_KEY, self.kept))
        if self.trajectories:
            pairs.append((TRAJECTORY_KEY, self.trajectories))
        entry = self.build_entry(caller=caller)
        if entry is not None:
            pairs.append((SUBAGENT_RETURNS_KEY, [entry]))
        return pairs

    def build_entry(self, *, caller: ToolCaller) -> SubagentReturn | None:
        """Return the entry the calling agent's next step answers, or None when there is nothing."""
        blocks = count_blocks_by_monitor(self.kept, earlier_blocks={})
        recorded_blocks = {label: count for label, count in blocks.items() if count}
        if not self.halted_subagents and not recorded_blocks:
            return None
        return SubagentReturn(
            id=str(uuid4()),
            delegation_id=caller.delegation_id,
            tool_call_id=caller.tool_call["id"],
            halted_subagents=list(dict.fromkeys(self.halted_subagents)),
            blocks=recorded_blocks,
            answered=False,
        )


def read_tool_caller(state: object, *, agent: str, tool_call: ToolCall) -> ToolCaller:
    """Return the calling agent as the state its tool call ran in describes it."""
    delegation = read_delegation(state)
    log = state.get(MONITOR_LOG_KEY) if isinstance(state, Mapping) else None
    return ToolCaller(
        agent=agent,
        delegation_id=None if delegation is None else delegation["tool_call_id"],
        state=state,
        log=log if isinstance(log, list) else [],
        tool_call=tool_call,
    )


def leave_out_written_back(
    written: Sequence[object],
    *,
    caller: ToolCaller,
) -> Sequence[object]:
    """Return the written records less the log they start with, when they start with all of it.

    A tool that writes the whole state back writes the log it was given,
    which already holds those records. That log holds the calling agent's
    own step that made the call, which no subagent writes, so a subagent's
    records never start with the whole log.
    """
    log = list(caller.log)
    if log and list(written[: len(log)]) == log:
        return written[len(log) :]
    return written


def check_written_record(item: object, *, caller: ToolCaller, returned: ReturnedRecords) -> None:
    """Keep one record a tool writes to `monitor_log`, or count or drop it, as the module says."""
    try:
        record = validate_step_record(item)
    except ValueError:
        logger.warning(
            "The tool %s wrote a record to %s that is not a step record the monitor can read: "
            "%s. The monitor keeps it out of the log and counts a halted subagent.",
            caller.tool_call["name"],
            MONITOR_LOG_KEY,
            render_value(item),
        )
        returned.halted_subagents.append(caller.describe_subagent())
        return
    if caller.is_caller_record(record):
        check_caller_record(record, caller=caller, returned=returned)
        return
    returned.kept.append(record)
    if record["outcome"] == "halted":
        returned.halted_subagents.append(caller.name_halted_subagent(record))


def check_caller_record(
    record: StepRecord,
    *,
    caller: ToolCaller,
    returned: ReturnedRecords,
) -> None:
    """Keep out a record that claims a step of the calling agent, counting a halt, or raise.

    A call that reuses the id of the call that started the calling agent
    hands its subagent the same delegation id, so a subagent that shares the
    agent's name records steps the monitor cannot tell from the agent's own.
    """
    if caller.has_own_delegation_id():
        message = (
            f"The tool call {caller.tool_call['id']} reuses the id of the tool call that "
            f"started the agent {caller.agent!r}, and returned records under that agent's "
            "name, so the monitor cannot tell a subagent's steps from the agent's own. Give "
            "the subagent's monitor an agent_name of its own, or use a model provider that "
            "gives every tool call its own id."
        )
        raise ConfigurationError(message)
    logger.warning(
        "The tool %s wrote a record to %s that claims to be a step of the agent %r itself: "
        "%s. Only the agent's own monitor records its steps, so the monitor keeps it out of "
        "the log, and counts a halted subagent if it is a halt.",
        caller.tool_call["name"],
        MONITOR_LOG_KEY,
        caller.agent,
        render_value(record),
    )
    if record["outcome"] == "halted":
        returned.halted_subagents.append(caller.describe_subagent())


def check_written_value(
    value: UpdateValue,
    *,
    caller: ToolCaller,
    returned: ReturnedRecords,
) -> None:
    """Check one value a command writes to `monitor_log`, record by record."""
    is_overwrite, overwritten = read_overwrite(value)
    written = overwritten if is_overwrite else value
    if is_overwrite:
        logger.warning(
            "The tool %s wrote an Overwrite of %s, which would erase the thread's records, so "
            "the monitor adds the records it holds instead.",
            caller.tool_call["name"],
            MONITOR_LOG_KEY,
        )
    if not isinstance(written, list | tuple):
        check_written_record(written, caller=caller, returned=returned)
        return
    for item in leave_out_written_back(written, caller=caller):
        check_written_record(item, caller=caller, returned=returned)


def check_written_trajectories(
    value: UpdateValue,
    *,
    caller: ToolCaller,
    returned: ReturnedRecords,
) -> None:
    """Check one value a command writes to `monitor_trajectory`, label by label."""
    is_overwrite, _ = read_overwrite(value)
    if is_overwrite or not isinstance(value, Mapping) or not value:
        drop_trajectory(caller=caller, returned=returned, problem="a value that maps no labels")
        return
    for label, state in value.items():
        check_written_trajectory(label, value=state, caller=caller, returned=returned)


def check_written_trajectory(
    label: object,
    *,
    value: object,
    caller: ToolCaller,
    returned: ReturnedRecords,
) -> None:
    """Keep one label's written state, drop it with a warning, or count it as a halt."""
    if isinstance(label, str) and not caller.holds_trajectory(label):
        logger.warning(
            "The tool %s wrote to %s a trajectory under the monitor label %r, which the agent "
            "%r has no trajectory for, so the monitor drops it; its steps stay in %s.",
            caller.tool_call["name"],
            TRAJECTORY_KEY,
            label,
            caller.agent,
            MONITOR_LOG_KEY,
        )
        return
    problem = find_trajectory_problem(label, value=value, caller=caller, returned=returned)
    if problem is None:
        # With no problem found, the label is text and the value a trajectory state.
        returned.trajectories[cast("str", label)] = cast("TrajectoryState", value)
    else:
        drop_trajectory(caller=caller, returned=returned, problem=problem)


def find_trajectory_problem(
    label: object,
    *,
    value: object,
    caller: ToolCaller,
    returned: ReturnedRecords,
) -> str | None:
    """Return what keeps one label's written state from joining the run, or None."""
    state = read_returned_state(value)
    if not isinstance(label, str) or state is None:
        return "a value that is not a subagent's trajectory state"
    if label in returned.trajectories:
        return f"a second trajectory for the monitor {label!r}"
    if state["delegation_id"] != caller.tool_call["id"]:
        return f"a trajectory for the monitor {label!r} that another call started"
    return find_state_problem(state, label=label)


def find_state_problem(state: TrajectoryState, *, label: str) -> str | None:
    """Return what keeps a subagent's own trajectory state from joining the run, or None."""
    if state["lost_writes"] > 0:
        return f"a trajectory for the monitor {label!r} that lost writes"
    try:
        Trajectory.load(state["trajectory"])
    except FormatError:
        return f"a trajectory for the monitor {label!r} that does not load"
    return None


def drop_trajectory(*, caller: ToolCaller, returned: ReturnedRecords, problem: str) -> None:
    """Drop a write to `monitor_trajectory` with a warning, and count a halted subagent."""
    logger.warning(
        "The tool %s wrote to %s %s. The monitor drops it and counts a halted subagent.",
        caller.tool_call["name"],
        TRAJECTORY_KEY,
        problem,
    )
    returned.halted_subagents.append(caller.describe_subagent())


def check_record_writes(pairs: UpdatePairs, *, caller: ToolCaller) -> UpdatePairs:
    """Return the pairs with every write to `monitor_log` and `monitor_trajectory` checked.

    The records kept go in one write to `monitor_log`, after the other pairs,
    the trajectories kept in one write to `monitor_trajectory`, and the entry
    for the calling agent's next step in one more.
    """
    returned = ReturnedRecords()
    for key, value in pairs:
        if is_log_key(key):
            check_written_value(value, caller=caller, returned=returned)
        elif is_trajectory_key(key):
            check_written_trajectories(value, caller=caller, returned=returned)
    kept_pairs = [(key, value) for key, value in pairs if not is_checked_key(key)]
    return [*kept_pairs, *returned.build_pairs(caller=caller)]


def is_log_key(key: str) -> bool:
    """Tell whether an update's key names `monitor_log`, compared with `==` as LangGraph does."""
    return key == MONITOR_LOG_KEY


def is_trajectory_key(key: str) -> bool:
    """Tell whether an update's key names `monitor_trajectory`, compared with `==`."""
    return key == TRAJECTORY_KEY


def is_checked_key(key: str) -> bool:
    """Tell whether an update's key names one of the monitor keys a tool may write."""
    return is_log_key(key) or is_trajectory_key(key)


def check_command_records(command: Command, *, caller: ToolCaller) -> Command:
    """Return a tool's command with its records checked, and their halts and blocks stored.

    A command bound for the parent graph is returned as it is: the parent
    agent's monitor checks it against its own state, once LangGraph has
    named that graph [@langgraph2026]. So is a command that writes neither key.
    """
    if command.graph == Command.PARENT:
        return command
    pairs = read_update_pairs(command)
    if not any(is_checked_key(key) for key, _ in pairs):
        return command
    return replace_update_pairs(command, pairs=check_record_writes(pairs, caller=caller))


def check_returned_records(results: ToolCallResults, *, caller: ToolCaller) -> ToolCallResults:
    """Check the records each command of a tool's result writes; a tool message writes none."""
    if isinstance(results, list):
        return [check_result_records(result, caller=caller) for result in results]
    return check_result_records(results, caller=caller)


def check_result_records(result: ToolCallResult, *, caller: ToolCaller) -> ToolCallResult:
    """Check the records one item of a tool's result writes."""
    if isinstance(result, Command):
        return check_command_records(result, caller=caller)
    return result


def check_parent_command_records(parent_command: ParentCommand, *, caller: ToolCaller) -> None:
    """Check, in place, the records the command in a `ParentCommand` a tool call raises writes."""
    [command] = parent_command.args
    parent_command.args = (check_command_records(command, caller=caller),)
