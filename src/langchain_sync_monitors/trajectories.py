"""Each monitor's trajectory of the run, kept in the graph state, each message in it once.

A trajectory is hyoshigi's history of the agent's actions and the
observations that reached it [@hyoshigi2026]. Each monitor keeps its own
under its label, in the private state key `monitor_trajectory`;
`trajectory_channel` has what the key holds and how writes fold in.

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
never read again. The decisions the step records stand in for them: a block
adds the blocked reply's action and the monitor's feedback, and the decision
that runs a reply adds the action the monitor rated, which is then followed
by the results of its server tools and the messages after the reply in its
draw. A halt adds nothing.

A thread whose `monitor_log` holds steps this agent's protocol decided, while
its trajectory under the label is missing, raises `MonitorError`: a
checkpointer that prunes without walking the key's delta chain rebuilds it
empty, and a thread checkpointed before the key existed has none, and the
protocol would count its records from nothing. A trajectory that is there
but holds fewer records is not checked, since such a checkpointer leaves
none at all.

A message whose action reuses an action id or a tool-call id already in the
trajectory raises `MessageFormatError`, as hyoshigi's own readers refuse one
[@hyoshigi2026].
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field

from hyoshigi import (
    Action,
    Author,
    ConfigurationError,
    Event,
    Observation,
    ToolCall,
    Trajectory,
)
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import Messages, add_messages
from langgraph.types import Overwrite

from langchain_sync_monitors._langchain import AgentModelRequest, AgentStateUpdate
from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.delegation import read_delegation_id
from langchain_sync_monitors.errors import MessageFormatError, MonitorError
from langchain_sync_monitors.message_keys import build_message_key
from langchain_sync_monitors.records import find_monitor_records, read_step_records
from langchain_sync_monitors.state_keys import TASK_MESSAGES_KEY, TRAJECTORY_KEY
from langchain_sync_monitors.task_authorship import (
    read_message_ids,
    read_state_messages,
    tag_context_notes,
)
from langchain_sync_monitors.trajectory_channel import (
    TrajectoryDelta,
    TrajectoryLengths,
    build_empty_state,
    measure_state,
    read_list,
    read_trajectory_state,
)
from langchain_sync_monitors.trajectory_events import read_events, read_observation
from langchain_sync_monitors.trajectory_runs import find_tool_call, load_run

DELEGATION_TOOL_NAME = "task"
"""The tool through which Deep Agents starts a subagent [@deepagents2026]."""

SYSTEM_SOURCES = frozenset({"system", "developer"})
"""The sources of the observation a system or developer message is."""

monitor_step_open: ContextVar[bool] = ContextVar("monitor_step_open", default=False)
"""True while a monitor drives its protocol, so a monitor its draws call is nested."""


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


def load_trajectory(
    state: Mapping[str, object],
    *,
    label: str,
    task_author: TaskAuthor,
) -> LoadedTrajectory:
    """Load a label's trajectory from the state, placed in its run as `load_run` says.

    A label that has none starts one; in a subagent, the human message it
    starts with is marked seen.
    """
    stored = read_trajectory_state(state, label=label) or build_empty_state(
        delegation_id=read_delegation_id(state),
    )
    place = load_run(state, label=label)
    loaded = LoadedTrajectory(
        label=label,
        trajectory=place.trajectory,
        delegation_id=stored["delegation_id"],
        task_author=task_author,
        nested=monitor_step_open.get(),
        base=measure_state(stored),
        lost_writes=stored["lost_writes"],
        subagent_ids=frozenset(place.trajectory.subagents),
        seen=set(stored["seen"]),
    )
    if place.instruction is not None:
        mark_seen(loaded, key=build_message_key(place.instruction))
    return loaded


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


def add_reply_results(loaded: LoadedTrajectory, *, messages: Sequence[BaseMessage]) -> None:
    """Add what follows a reply that ran: its server tools' results, then each later message.

    `messages` are the reply, whose action the step recorded, and the
    messages after it in its draw, such as the tool message of tool-strategy
    structured output.
    """
    reply, *later = messages
    add_events(
        loaded, events=read_events(reply, task_author=loaded.task_author)[1:], path="AIMessage"
    )
    for message in later:
        events = read_events(message, task_author=loaded.task_author)
        add_events(loaded, events=events, path=type(message).__name__)


def check_trajectory_kept(state: Mapping[str, object], *, label: str, agent: str) -> None:
    """Raise `MonitorError` when this agent's decided steps outlive its trajectory under `label`.

    A decided step is one of the agent's records that holds a sample.
    """
    if read_trajectory_state(state, label=label) is not None:
        return
    records = find_monitor_records(
        read_step_records(state),
        monitor=label,
        agent=agent,
        delegation_id=read_delegation_id(state),
    )
    if any(record["samples"] for record in records):
        message = (
            f"monitor_log holds steps the monitor {label!r} decided for the agent {agent!r}, "
            f"but monitor_trajectory holds no trajectory for it: the checkpointer lost it, or "
            "the thread was checkpointed before the key existed. The protocol would count its "
            "records from nothing, so the monitor stops; start a new thread."
        )
        raise MonitorError(message)


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
