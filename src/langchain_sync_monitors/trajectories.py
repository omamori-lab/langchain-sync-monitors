"""Each monitor's trajectory of the run, kept in the graph state, each message in it once.

A trajectory is hyoshigi's history of the agent's actions and the
observations that reached it [@hyoshigi2026]. Each monitor keeps its own
under its label, in the private state key `monitor_trajectory`, as a
`TrajectoryState`. Each hook that adds to it writes a `TrajectoryDelta`, the
entries beyond the lengths it loaded, and `merge_trajectories` folds the
deltas in. The key is a LangGraph `DeltaChannel`, so a checkpoint stores the
writes and rebuilds the value from them [@langgraph2026].

The monitor reads each message into its trajectory once, by its key: its id,
or a hash of its type and content when it has none. It reads them at
`before_agent`, so a run's input is
read before a summariser can remove it, and at the start of each step, over
the request's system message, the state's messages and the messages another
middleware added to the request alone. A human message no run received as
input is read as a context note, as `task_authorship` tags it. The system
message has no id; it is read whenever it differs from the latest system or
developer message in the trajectory. A tool message answering a Deep Agents
`task` call whose subagent the trajectory lacks first starts that
subagent's trajectory, with the call's description as its instruction
[@deepagents2026].

An id the start or the end of a run gives a message the trajectory read under
its key is marked seen as well, in the same write (`given_ids`), so the message
is not read again under its id.

Each message a step commits is marked seen in the step's own write, so it is
never read again. The step's events stand in for them: each
blocked attempt as its proposal and the monitor's feedback, whatever the
feedback visibility, then the reply that runs as the action the monitor
rated and the results of its server tools. A halt message adds nothing.

A message whose action reuses an action id or a tool-call id already in the
trajectory raises `MessageFormatError`, as hyoshigi's own readers refuse one
[@hyoshigi2026].
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TypedDict, cast

from hyoshigi import (
    JSON,
    Action,
    Author,
    ConfigurationError,
    Event,
    Observation,
    ToolCall,
    Trajectory,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import Messages, add_messages
from langgraph.types import Overwrite

from langchain_sync_monitors._langchain import AgentModelRequest, AgentStateUpdate
from langchain_sync_monitors.contracts import BlockedAttempt, Outcome, StepDecision, TaskAuthor
from langchain_sync_monitors.delegation import read_delegation_id
from langchain_sync_monitors.errors import MessageFormatError
from langchain_sync_monitors.feedback import build_feedback_messages
from langchain_sync_monitors.state_keys import TASK_MESSAGES_KEY, TRAJECTORY_KEY
from langchain_sync_monitors.task_authorship import (
    read_message_ids,
    read_state_messages,
    tag_context_notes,
)
from langchain_sync_monitors.trajectory_events import (
    read_events,
    read_observation,
    read_proposed_action,
)

DELEGATION_TOOL_NAME = "task"
"""The tool through which Deep Agents starts a subagent [@deepagents2026]."""

SYSTEM_SOURCES = frozenset({"system", "developer"})
"""The sources of the observation a system or developer message is."""

monitor_step_open: ContextVar[bool] = ContextVar("monitor_step_open", default=False)
"""True while a monitor drives its protocol, so a monitor its draws call is nested."""


class TrajectoryState(TypedDict):
    """One monitor label's trajectory, as the state key holds it.

    `trajectory` is `Trajectory.dump()`, `delegation_id` the id of the tool
    call that started this agent, or None at the root, and `seen` the key of
    every message the monitor has read or written, in order. `lost_writes`
    counts the stale writes the key dropped outside the stacked case that no
    step has yet answered with a halt.
    """

    trajectory: dict[str, JSON]
    delegation_id: str | None
    seen: list[str]
    lost_writes: int


class TrajectoryLengths(TypedDict):
    """The lengths of a label's lists that a write was computed against."""

    history: int
    ratings: int
    records: int
    seen: int


class TrajectoryDelta(TypedDict):
    """What one hook adds to one label's trajectory.

    `base` holds the lengths the hook loaded. The lists hold the entries of
    `Trajectory.dump()` beyond them, `subagents` the subagents the hook
    started, and `seen` the keys it read or wrote. `nested` is true for a
    monitor called inside another monitor's step, whose writes pile up when
    that monitor draws at once. `answered_lost_writes` is the count of lost
    writes the hook's step halted for.
    """

    base: TrajectoryLengths
    delegation_id: str | None
    history: list[JSON]
    ratings: list[JSON]
    records: list[JSON]
    subagents: dict[str, JSON]
    seen: list[str]
    nested: bool
    answered_lost_writes: int


def build_empty_state(*, delegation_id: str | None) -> TrajectoryState:
    """Return the state of a label with no trajectory yet."""
    return TrajectoryState(
        trajectory=Trajectory().dump(),
        delegation_id=delegation_id,
        seen=[],
        lost_writes=0,
    )


def measure_state(state: TrajectoryState) -> TrajectoryLengths:
    """Return the lengths of a label's history, ratings, records and seen keys."""
    stored = state["trajectory"]
    return TrajectoryLengths(
        history=len(read_list(stored, key="history")),
        ratings=len(read_list(stored, key="ratings")),
        records=len(read_list(stored, key="records")),
        seen=len(state["seen"]),
    )


def is_behind(base: TrajectoryLengths, *, lengths: TrajectoryLengths) -> bool:
    """Tell whether a write's base is behind the lengths: no greater in any list, and not equal."""
    pairs = (
        (base["history"], lengths["history"]),
        (base["ratings"], lengths["ratings"]),
        (base["records"], lengths["records"]),
        (base["seen"], lengths["seen"]),
    )
    return base != lengths and all(value <= length for value, length in pairs)


def count_lost_write(state: TrajectoryState) -> TrajectoryState:
    """Return the state with one more lost write, and nothing else changed."""
    return {**state, "lost_writes": state["lost_writes"] + 1}


def extend_state(state: TrajectoryState, *, delta: TrajectoryDelta) -> TrajectoryState:
    """Return the state with a delta's entries appended, as new lists, and its answer applied."""
    stored = state["trajectory"]
    subagents = stored.get("subagents")
    trajectory: dict[str, JSON] = {
        **stored,
        "history": [*read_list(stored, key="history"), *delta["history"]],
        "ratings": [*read_list(stored, key="ratings"), *delta["ratings"]],
        "records": [*read_list(stored, key="records"), *delta["records"]],
        "subagents": {**(subagents if isinstance(subagents, dict) else {}), **delta["subagents"]},
    }
    return TrajectoryState(
        trajectory=trajectory,
        delegation_id=state["delegation_id"],
        seen=[*state["seen"], *delta["seen"]],
        lost_writes=max(0, state["lost_writes"] - delta["answered_lost_writes"]),
    )


def read_list(stored: Mapping[str, JSON], *, key: str) -> list[JSON]:
    """Return one of a dump's lists, or an empty one where it holds none."""
    value = stored.get(key)
    return value if isinstance(value, list) else []


def read_delta(write: object) -> TrajectoryDelta | None:
    """Return a write as a delta when it has a delta's shape, else None."""
    match write:
        case {
            "base": {"history": int(), "ratings": int(), "records": int(), "seen": int()},
            "delegation_id": str() | None,
            "history": list(),
            "ratings": list(),
            "records": list(),
            "subagents": dict(),
            "seen": list(),
            "nested": bool(),
            "answered_lost_writes": int(),
        }:
            # The pattern checked every key a delta holds.
            return cast("TrajectoryDelta", write)
        case _:
            return None


def merge_write(state: TrajectoryState | None, *, write: object) -> TrajectoryState:
    """Apply one label's write to its state, which None means the label does not have yet.

    A write that is not a delta counts as a lost write.
    """
    delta = read_delta(write)
    if delta is None:
        return count_lost_write(state or build_empty_state(delegation_id=None))
    held = state or build_empty_state(delegation_id=delta["delegation_id"])
    return merge_delta(held, delta=delta)


def merge_delta(held: TrajectoryState, *, delta: TrajectoryDelta) -> TrajectoryState:
    """Apply a delta to a label's state.

    A delta whose base equals the label's lengths is applied. A nested delta
    behind them, a stacked monitor's piled-up write, is dropped. Any other
    delta counts as a lost write.
    """
    lengths = measure_state(held)
    if delta["base"] == lengths:
        return extend_state(held, delta=delta)
    if delta["nested"] and is_behind(delta["base"], lengths=lengths):
        return held
    return count_lost_write(held)


def merge_trajectories(
    current: dict[str, TrajectoryState],
    writes: Sequence[dict[str, TrajectoryDelta]],
    /,
) -> dict[str, TrajectoryState]:
    """Fold each write into the trajectories, label by label, as `merge_write` says.

    It is a pure left fold over the writes, copying what it changes, so it
    gives the same value however LangGraph batches the writes, as a
    `DeltaChannel` reducer must [@langgraph2026]. It never raises and never
    logs, since LangGraph runs it again whenever it rebuilds the value; a
    write that is not a mapping of labels is left out.
    """
    merged = dict(current)
    for write in writes:
        for label, value in read_labelled_values(write):
            merged[label] = merge_write(merged.get(label), write=value)
    return merged


def read_labelled_values(write: object) -> Iterator[tuple[str, object]]:
    """Yield each label of a write with its value, leaving out a key that is not text."""
    if isinstance(write, Mapping):
        yield from ((label, value) for label, value in write.items() if isinstance(label, str))


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedTrajectory:
    """One label's trajectory as a hook loaded it, and what the hook adds to it.

    `base` holds the lengths it was loaded with, and `subagent_ids` the
    subagents it held then. `nested` says whether the hook runs inside
    another monitor's step. `seen` and `new_seen` grow as the hook reads and
    writes messages.
    """

    label: str
    trajectory: Trajectory
    delegation_id: str | None
    task_author: TaskAuthor
    nested: bool
    base: TrajectoryLengths
    lost_writes: int
    subagent_ids: frozenset[str]
    seen: set[str] = field(default_factory=set)
    new_seen: list[str] = field(default_factory=list)


def read_trajectory_state(state: Mapping[str, object], *, label: str) -> TrajectoryState | None:
    """Return a label's stored trajectory state, or None when the state holds none."""
    stored = state.get(TRAJECTORY_KEY)
    value = stored.get(label) if isinstance(stored, Mapping) else None
    # Only merge_trajectories writes the key, so a stored value is a state.
    return cast("TrajectoryState", value) if isinstance(value, dict) else None


def load_trajectory(
    state: Mapping[str, object],
    *,
    label: str,
    task_author: TaskAuthor,
) -> LoadedTrajectory:
    """Load a label's trajectory from the state, or start one for a label that has none."""
    stored = read_trajectory_state(state, label=label) or build_empty_state(
        delegation_id=read_delegation_id(state),
    )
    trajectory = Trajectory.load(stored["trajectory"])
    return LoadedTrajectory(
        label=label,
        trajectory=trajectory,
        delegation_id=stored["delegation_id"],
        task_author=task_author,
        nested=monitor_step_open.get(),
        base=measure_state(stored),
        lost_writes=stored["lost_writes"],
        subagent_ids=frozenset(trajectory.subagents),
        seen=set(stored["seen"]),
    )


def build_message_key(message: BaseMessage) -> str:
    """Return a message's key: its id, or a hash of its type and content when it has none."""
    if message.id:
        return message.id
    text = json.dumps([type(message).__name__, message.content], sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def add_event(loaded: LoadedTrajectory, *, event: Event, path: str) -> None:
    """Act an action or observe an observation, raising `MessageFormatError` on a reused id."""
    if not isinstance(event, Action):
        loaded.trajectory.observe(event)
        return
    try:
        loaded.trajectory.act(event)
    except ConfigurationError:
        raise MessageFormatError(
            path=path,
            expected="an action whose id and tool call ids are new to the trajectory",
        ) from None


def add_events(loaded: LoadedTrajectory, *, events: Sequence[Event], path: str) -> None:
    """Add each event of one message, in order."""
    for event in events:
        add_event(loaded, event=event, path=path)


def mark_seen(loaded: LoadedTrajectory, *, key: str) -> None:
    """Record a message's key as read, once."""
    if key not in loaded.seen:
        loaded.seen.add(key)
        loaded.new_seen.append(key)


type GivenId = tuple[str, str]
"""The key a message had, then the id an update gave it."""


def mark_given_ids(loaded: LoadedTrajectory, *, given: Sequence[GivenId]) -> None:
    """Mark each given id as read where the trajectory read the message under its key."""
    for key, message_id in given:
        if key in loaded.seen:
            mark_seen(loaded, key=message_id)


def find_tool_call(trajectory: Trajectory, *, call_id: str) -> ToolCall | None:
    """Return the tool call with this id among the trajectory's actions, latest first."""
    for event in reversed(trajectory.history):
        if isinstance(event, Action):
            for call in event.tool_calls:
                if call.id == call_id:
                    return call
    return None


def read_task_description(call: ToolCall) -> str | None:
    """Return a `task` call's description, or None when its arguments hold none."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    match arguments:
        case {"description": str(description)}:
            return description
        case _:
            return None


def delegate_task(loaded: LoadedTrajectory, *, message: ToolMessage) -> None:
    """Start the subagent a `task` call's answer reports on, unless the trajectory holds it."""
    call_id = message.tool_call_id
    if call_id in loaded.trajectory.subagents:
        return
    call = find_tool_call(loaded.trajectory, call_id=call_id)
    if call is None or call.name != DELEGATION_TOOL_NAME:
        return
    description = read_task_description(call)
    if description is not None:
        loaded.trajectory.delegate(description, call_id=call_id)


def add_unread_messages(
    loaded: LoadedTrajectory,
    *,
    messages: Sequence[BaseMessage],
    task_message_ids: frozenset[str],
) -> None:
    """Add each message the trajectory has not read, once, by its key.

    Human messages no run received as input are read as context notes.
    """
    for message in tag_context_notes(messages, task_message_ids=task_message_ids):
        key = build_message_key(message)
        if key in loaded.seen:
            continue
        if isinstance(message, ToolMessage):
            delegate_task(loaded, message=message)
        events = read_events(message, task_author=loaded.task_author)
        add_events(loaded, events=events, path=type(message).__name__)
        mark_seen(loaded, key=key)


def find_latest_system_observation(trajectory: Trajectory) -> Observation | None:
    """Return the latest observation a system or developer message added, if any."""
    for event in reversed(trajectory.history):
        if (
            isinstance(event, Observation)
            and event.author is Author.SCAFFOLD
            and event.source in SYSTEM_SOURCES
        ):
            return event
    return None


def add_changed_system_message(loaded: LoadedTrajectory, *, message: SystemMessage | None) -> None:
    """Observe the request's system message when it differs from the latest one observed."""
    if message is None:
        return
    observation = read_observation(message, task_author=loaded.task_author)
    if observation != find_latest_system_observation(loaded.trajectory):
        loaded.trajectory.observe(observation)


def find_request_only_messages(
    request_messages: Sequence[BaseMessage],
    *,
    state_messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Return the request's messages that the state does not hold, by id."""
    state_ids = {message.id for message in state_messages if message.id}
    return [message for message in request_messages if message.id not in state_ids]


def load_step_trajectory(
    request: AgentModelRequest,
    *,
    label: str,
    task_author: TaskAuthor,
) -> LoadedTrajectory:
    """Load a label's trajectory at the start of a step, and sync what the step's request holds.

    The request's system message comes first, then the state's messages, then
    the messages only the request holds.
    """
    state = request.state
    loaded = load_trajectory(state, label=label, task_author=task_author)
    add_changed_system_message(loaded, message=request.system_message)
    state_messages = read_state_messages(state)
    request_only = find_request_only_messages(request.messages, state_messages=state_messages)
    add_unread_messages(
        loaded,
        messages=[*state_messages, *request_only],
        task_message_ids=read_message_ids(state, key=TASK_MESSAGES_KEY),
    )
    return loaded


def read_state_after(state: Mapping[str, object], *, update: AgentStateUpdate) -> dict[str, object]:
    """Return the state as an update to its messages and run inputs leaves it.

    The messages are merged by LangGraph's message reducer, or replaced by an
    `Overwrite` [@langgraph2026].
    """
    written = update.get("messages")
    messages: Messages = [*read_state_messages(state)]
    if isinstance(written, Overwrite):
        messages = written.value
    elif written:
        messages = add_messages(messages, written)
    task_message_ids = [
        *read_message_ids(state, key=TASK_MESSAGES_KEY),
        *update.get(TASK_MESSAGES_KEY, []),
    ]
    return {**state, "messages": messages, TASK_MESSAGES_KEY: task_message_ids}


def load_run_start_trajectory(
    state: Mapping[str, object],
    *,
    update: AgentStateUpdate,
    given: Sequence[GivenId],
    label: str,
    task_author: TaskAuthor,
) -> LoadedTrajectory:
    """Load a label's trajectory at the start of a run, and sync the state `update` leaves.

    Each id `update` gave a message is marked read first, where the label read
    the message under its key.
    """
    started = read_state_after(state, update=update)
    loaded = load_trajectory(started, label=label, task_author=task_author)
    mark_given_ids(loaded, given=given)
    add_unread_messages(
        loaded,
        messages=read_state_messages(started),
        task_message_ids=read_message_ids(started, key=TASK_MESSAGES_KEY),
    )
    return loaded


def add_reply(loaded: LoadedTrajectory, *, message: AIMessage) -> None:
    """Add a reply that runs: the action the monitor rated, then its server tools' results."""
    path = type(message).__name__
    add_event(loaded, event=read_proposed_action(message), path=path)
    add_events(loaded, events=read_events(message, task_author=loaded.task_author)[1:], path=path)


def add_step_events(loaded: LoadedTrajectory, *, decision: StepDecision) -> None:
    """Add a decided step's events.

    Each blocked attempt adds its proposal and the monitor's feedback on it.
    A halt adds nothing more. Otherwise the response's first AI message adds
    the action the monitor rated, and each other message its events.
    """
    for attempt in decision.blocked_attempts:
        add_blocked_attempt(loaded, attempt=attempt)
    if decision.outcome is Outcome.HALTED:
        return
    messages = decision.response.result
    reply = next((message for message in messages if isinstance(message, AIMessage)), None)
    for message in messages:
        if message is reply:
            add_reply(loaded, message=message)
        else:
            events = read_events(message, task_author=loaded.task_author)
            add_events(loaded, events=events, path=type(message).__name__)


def add_blocked_attempt(loaded: LoadedTrajectory, *, attempt: BlockedAttempt) -> None:
    """Add a blocked attempt: the action the monitor rated, then its feedback on it."""
    add_event(loaded, event=read_proposed_action(attempt.proposal), path="AIMessage")
    for feedback in build_feedback_messages(attempt=attempt)[1:]:
        events = read_events(feedback, task_author=loaded.task_author)
        add_events(loaded, events=events, path=type(feedback).__name__)


def mark_committed(loaded: LoadedTrajectory, *, committed: Sequence[BaseMessage]) -> None:
    """Mark each message a step commits as seen, so it is never read again."""
    for message in committed:
        mark_seen(loaded, key=build_message_key(message))


def build_trajectory_update(
    loaded: LoadedTrajectory,
    *,
    answered_lost_writes: int,
) -> AgentStateUpdate:
    """Return the update that writes what a hook added to its label's trajectory, or none.

    The delta holds the entries of `Trajectory.dump()` beyond the lengths the
    hook loaded, the subagents it started and the keys it marked seen.
    """
    dump = loaded.trajectory.dump()
    base = loaded.base
    subagents = dump["subagents"]
    started = {
        call_id: subagent
        for call_id, subagent in (subagents.items() if isinstance(subagents, dict) else ())
        if call_id not in loaded.subagent_ids
    }
    delta = TrajectoryDelta(
        base=base,
        delegation_id=loaded.delegation_id,
        history=read_list(dump, key="history")[base["history"] :],
        ratings=read_list(dump, key="ratings")[base["ratings"] :],
        records=read_list(dump, key="records")[base["records"] :],
        subagents=started,
        seen=list(loaded.new_seen),
        nested=loaded.nested,
        answered_lost_writes=answered_lost_writes,
    )
    added = (
        delta["history"],
        delta["ratings"],
        delta["records"],
        delta["subagents"],
        delta["seen"],
    )
    changed = any(added) or answered_lost_writes > 0
    return {TRAJECTORY_KEY: {loaded.label: delta}} if changed else {}
