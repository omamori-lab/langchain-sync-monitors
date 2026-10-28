"""Each monitor's trajectory as the state key holds it, and the reducer that folds writes in.

The private state key `monitor_trajectory` holds each monitor's
`TrajectoryState` under its label. Each hook that adds to a trajectory
writes a `TrajectoryDelta`, the entries beyond the lengths it loaded, and
`merge_trajectories` folds the deltas in. The key is a LangGraph
`DeltaChannel`, so a checkpoint stores the writes and rebuilds the value from
them [@langgraph2026].
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TypedDict, cast

from hyoshigi import JSON, Trajectory


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
