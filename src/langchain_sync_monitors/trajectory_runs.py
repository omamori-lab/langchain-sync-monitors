"""Where a monitor's trajectory sits in its run, when one monitored agent starts another.

The run goes down to a subagent, and the subagent's trajectory comes back
[@hyoshigi2026]. Before each tool call, the monitor hands any subagent the
call starts a `DelegatedRun` for each monitor label whose trajectory holds
the call: the run's root, as `Trajectory.dump()` gives it, with the calling
agent's own trajectory joined in, and the ids of the calls from the root to
that agent. The subagent's monitor places its own trajectory in that run. On
its first load it starts it under the call, with the subagent's first human
message as the instruction; afterwards it joins its stored trajectory there,
so the records it adds take the run's next numbers. A label without a run
keeps a trajectory at the root of a run of its own. Once the call returns
the subagent's state, `trajectory_channel` joins it into its parent's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import hyoshigi
from hyoshigi import Action, ToolCall, Trajectory
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_sync_monitors._langchain import MONITOR_DELEGATION_KEY, read_delegation
from langchain_sync_monitors.contracts import DelegatedRun
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.task_authorship import read_state_messages
from langchain_sync_monitors.trajectory_channel import read_labels, read_trajectory_state


@dataclass(frozen=True, slots=True, kw_only=True)
class RunPlace:
    """A label's trajectory, placed in its run.

    `trajectory` is the agent's own, `root` the root of the run that holds
    it, and `path` the ids of the calls from the root to it. `instruction` is
    the human message this load started a subagent's trajectory with, if any.
    """

    trajectory: Trajectory
    root: Trajectory
    path: tuple[str, ...]
    instruction: BaseMessage | None


def load_run(state: Mapping[str, object], *, label: str) -> RunPlace:
    """Load a label's trajectory from the state, placed in the run its delegation holds.

    A run that does not load, or cannot hold the trajectory under the call,
    raises `ConfigurationError`.
    """
    stored = read_trajectory_state(state, label=label)
    own = None if stored is None else Trajectory.load(stored["trajectory"])
    delegation = read_delegation(state)
    if delegation is None or label not in delegation["runs"]:
        root = own or Trajectory()
        return RunPlace(trajectory=root, root=root, path=(), instruction=None)
    run = delegation["runs"][label]
    try:
        return place_in_run(run, call_id=delegation["tool_call_id"], own=own, state=state)
    except (hyoshigi.FormatError, hyoshigi.ConfigurationError) as error:
        raise ConfigurationError(build_run_message(label)) from error


def place_in_run(
    run: DelegatedRun,
    *,
    call_id: str,
    own: Trajectory | None,
    state: Mapping[str, object],
) -> RunPlace:
    """Join the agent's stored trajectory under the call, or start it there with its task.

    A path that leads to no trajectory raises `ConfigurationError`.
    """
    root = Trajectory.load(run["trajectory"])
    parent = find_subagent(root, path=run["path"])
    if parent is None:
        message = f"{MONITOR_DELEGATION_KEY} holds a run whose path leads to no trajectory."
        raise ConfigurationError(message)
    path = (*run["path"], call_id)
    if own is not None:
        joined = parent.join(own, call_id=call_id)
        return RunPlace(trajectory=joined, root=root, path=path, instruction=None)
    instruction = find_instruction(state)
    started = parent.delegate("" if instruction is None else instruction.text, call_id=call_id)
    return RunPlace(trajectory=started, root=root, path=path, instruction=instruction)


def build_run_message(label: str) -> str:
    """Explain that a delegation's run cannot hold the agent's trajectory."""
    return (
        f"{MONITOR_DELEGATION_KEY} holds a run for the monitor {label!r} that cannot hold this "
        "agent's trajectory. Leave it out of an agent's input: the monitor sets it for each "
        "subagent it starts."
    )


def find_subagent(root: Trajectory, *, path: Sequence[str]) -> Trajectory | None:
    """Return the subagent a path of call ids leads to from the root, or None."""
    found: Trajectory | None = root
    for call_id in path:
        found = None if found is None else found.subagents.get(call_id)
    return found


def find_instruction(state: Mapping[str, object]) -> HumanMessage | None:
    """Return the first human message in the state, the task a subagent was given, if any."""
    messages = read_state_messages(state)
    return next((message for message in messages if isinstance(message, HumanMessage)), None)


def find_tool_call(trajectory: Trajectory, *, call_id: str) -> ToolCall | None:
    """Return the tool call with this id among the trajectory's actions, latest first."""
    for event in reversed(trajectory.history):
        if isinstance(event, Action):
            for call in event.tool_calls:
                if call.id == call_id:
                    return call
    return None


def build_delegated_runs(state: Mapping[str, object], *, call_id: str) -> dict[str, DelegatedRun]:
    """Return the run a call hands its subagent for each label whose trajectory holds the call."""
    runs: dict[str, DelegatedRun] = {}
    for label in read_labels(state):
        place = load_run(state, label=label)
        if find_tool_call(place.trajectory, call_id=call_id) is not None:
            runs[label] = DelegatedRun(trajectory=place.root.dump(), path=list(place.path))
    return runs
