"""Each monitor's trajectory of the run, kept in the graph state.

The values below are the shapes the state key holds and the writes that
change it. `merge_trajectories` is the key's reducer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from hyoshigi import JSON


class TrajectoryState(TypedDict):
    """One monitor label's trajectory, as the state key holds it."""

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
    """What one hook adds to one label's trajectory."""

    base: TrajectoryLengths
    delegation_id: str | None
    history: list[JSON]
    ratings: list[JSON]
    records: list[JSON]
    subagents: dict[str, JSON]
    seen: list[str]
    nested: bool
    answered_lost_writes: int


def merge_trajectories(
    current: dict[str, TrajectoryState],
    writes: Sequence[dict[str, TrajectoryDelta]],
    /,
) -> dict[str, TrajectoryState]:
    """Return the trajectories unchanged."""
    del writes
    return current
