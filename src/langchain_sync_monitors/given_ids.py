"""The ids an update gives messages that had none, marked as read in each monitor's trajectory.

A trajectory reads a message without an id under the key of its type and
content. When the start or the end of a run gives that message an id
(`message_ids`), the same update marks the id as read in every label whose
trajectory read the message, so the next run does not read it again.

The end of a run always does it. The start of a run does it only after a run
that did not end, the only run that leaves a message read under its key
without an id: the input of any other run is new, and its new id must not hide
it from the trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hyoshigi import Halt
from langchain_core.messages import BaseMessage
from langgraph.types import Overwrite

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.message_ids import build_run_end_update
from langchain_sync_monitors.message_keys import build_content_key
from langchain_sync_monitors.state_keys import TRAJECTORY_KEY
from langchain_sync_monitors.task_authorship import is_run_open, read_state_messages
from langchain_sync_monitors.trajectories import (
    GivenId,
    LoadedTrajectory,
    build_trajectory_update,
    load_run_start_trajectory,
    load_trajectory,
    mark_given_ids,
)


def read_written_messages(update: AgentStateUpdate) -> list[BaseMessage]:
    """Return the history an update writes back in an `Overwrite`, or none when it writes none."""
    written = update.get("messages")
    if not isinstance(written, Overwrite):
        return []
    return [message for message in written.value if isinstance(message, BaseMessage)]


def find_given_ids(state: Mapping[str, object], *, update: AgentStateUpdate) -> list[GivenId]:
    """Return each id `update` gives a message the state holds without one.

    An update gives ids by writing the history back in an `Overwrite`, one
    message for each the state holds, in the same order. A content key
    stands for one message, the first in the history with that type and
    content, with an id or without: only its id is returned.
    """
    written = read_written_messages(update)
    held = read_state_messages(state)
    if len(written) != len(held):
        return []
    keys = [build_content_key(message) for message in held]
    return [
        (key, after.id)
        for index, (key, before, after) in enumerate(zip(keys, held, written, strict=True))
        if not before.id and after.id and keys.index(key) == index
    ]


def load_marked_trajectories(
    state: Mapping[str, object],
    *,
    given: Sequence[GivenId],
    task_author: TaskAuthor,
) -> dict[str, LoadedTrajectory]:
    """Load every label's trajectory, with each given id marked where the label read it."""
    stored = state.get(TRAJECTORY_KEY)
    labels = (
        [label for label in stored if isinstance(label, str)] if isinstance(stored, Mapping) else []
    )
    loaded = {
        label: load_trajectory(state, label=label, task_author=task_author) for label in labels
    }
    for trajectory in loaded.values():
        mark_given_ids(trajectory, given=given)
    return loaded


def build_trajectories_update(
    state: Mapping[str, object],
    *,
    given: Sequence[GivenId],
    task_author: TaskAuthor,
    own: LoadedTrajectory | None = None,
) -> AgentStateUpdate:
    """Return the update that writes the trajectories a hook changed, or none.

    The hook's `own` trajectory writes what the hook added to it, and the ids
    marked read in it. When the hook gave ids, every other label writes the
    ids it marks read.
    """
    loaded = load_marked_trajectories(state, given=given, task_author=task_author) if given else {}
    if own is not None:
        loaded[own.label] = own
    writes: dict[str, object] = {}
    for trajectory in loaded.values():
        update = build_trajectory_update(trajectory, answered_lost_writes=0)
        writes.update(update.get(TRAJECTORY_KEY, {}))
    return {TRAJECTORY_KEY: writes} if writes else {}


def build_run_start_trajectories_update(
    state: Mapping[str, object],
    *,
    update: AgentStateUpdate,
    label: str,
    task_author: TaskAuthor,
    stop: Halt | None = None,
) -> AgentStateUpdate:
    """Return the update that adds a run's input to the trajectories, with the given ids marked.

    `stop` is the halt of a held step the input stopped, recorded first.
    """
    # Only a run that did not end leaves a message read under its key without an id.
    given = find_given_ids(state, update=update) if is_run_open(state) else []
    own = load_run_start_trajectory(
        state,
        update=update,
        given=given,
        label=label,
        task_author=task_author,
        stop=stop,
    )
    return build_trajectories_update(state, given=given, task_author=task_author, own=own)


def build_run_end_trajectories_update(
    state: Mapping[str, object],
    *,
    update: AgentStateUpdate,
    task_author: TaskAuthor,
) -> AgentStateUpdate:
    """Return the update that marks the ids a run's end gives as read in the trajectories."""
    given = find_given_ids(state, update=update)
    return build_trajectories_update(state, given=given, task_author=task_author)


def build_marked_run_end_update(
    state: Mapping[str, object],
    *,
    task_author: TaskAuthor,
) -> AgentStateUpdate:
    """Return the update a run ends with: the notes tagged, the run closed, ids marked read.

    A message a trajectory read under its key, and the end gives an id,
    would otherwise be read again under that id in the next run.
    """
    update = build_run_end_update(state)
    trajectories = build_run_end_trajectories_update(state, update=update, task_author=task_author)
    return {**update, **trajectories}
