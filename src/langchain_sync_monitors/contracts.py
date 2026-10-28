"""The options and records the middleware and its users exchange.

A step is one run of the agent's model node, as defined under "The monitored
step" in `docs/explanation/design.md`. The untrusted model proposes it, the
monitor rates it, and a control protocol decides what the agent state
receives; `monitor_log` keeps one `StepRecord` per step.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from hyoshigi import JSON


class TaskAuthor(StrEnum):
    """Who wrote the task the agent is working on.

    Inside a subagent the first human message was written by the parent agent,
    so a monitor must not read it as the user's authorisation.
    """

    USER = "user"
    PARENT_AGENT = "parent_agent"


class SubagentHalt(StrEnum):
    """What a halt inside a subagent does to the rest of the run."""

    STOP_SUBAGENT_ONLY = "stop_subagent_only"
    STOP_WHOLE_RUN = "stop_whole_run"


OutcomeName = Literal[
    "allowed",
    "resampled",
    "deferred_to_trusted",
    "steered",
    "halted",
    "approved",
    "edited",
    "rejected",
    "audited",
]
"""The outcome of a step as it is stored in `StepRecord`, a plain string."""


class Outcome(StrEnum):
    """What became of a step, as `StepRecord.outcome` names it.

    `APPROVED`, `EDITED` and `REJECTED` name a person's answer to an audit,
    and a step stopped at an audit is `HALTED`. `AUDITED` names a step held
    for a person, and appears only on the spans of the model call that held
    it, never in `monitor_log`: the step's record is written by the answer.
    """

    ALLOWED = "allowed"
    RESAMPLED = "resampled"
    DEFERRED_TO_TRUSTED = "deferred_to_trusted"
    STEERED = "steered"
    HALTED = "halted"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    AUDITED = "audited"


class SampleRecord(TypedDict):
    """One rated proposal of the agent's model, as stored in the agent state.

    `suspicion` and `reason` are the score's value and explanation, and
    `proposal` the action the monitor rated, as `Trajectory.dump` stores an
    action in a trajectory's history [@hyoshigi2026]. `executed` is true for
    the proposal that ran.
    """

    suspicion: float
    reason: str
    proposal: dict[str, JSON]
    executed: bool


class StepRecord(TypedDict):
    """One step as stored in the agent state's `monitor_log`.

    Records pass through checkpointers, so they hold plain values only: the
    outcome is a string and every sequence is a list. A subagent's step
    numbers start again with every delegation, so its records also carry
    `delegation_id`, the id of the tool call that started the subagent, and
    `(agent, delegation_id, step_number)` names one step, provided the model
    provider gives every tool call in the thread its own id, which LangChain
    does not check. The records of an agent that no monitored agent started
    have no `delegation_id`.
    """

    agent: str
    monitor: str
    step_number: int
    outcome: OutcomeName
    flagged: bool
    blocked_count: int
    samples: list[SampleRecord]
    delegation_id: NotRequired[str]


class DelegatedRun(TypedDict):
    """The run a subagent's trajectory joins, under one monitor label.

    `trajectory` is the run's root, as `Trajectory.dump()` gives it, with the
    trajectory of the agent that made the call joined in, and `path` the ids
    of the calls from the root to that agent, empty when it is the root
    [@hyoshigi2026].
    """

    trajectory: dict[str, JSON]
    path: list[str]


class Delegation(TypedDict):
    """What a monitored agent hands to a subagent that one of its tool calls starts.

    `tool_call_id` is the id of that tool call, which the subagent's records
    carry as their `delegation_id`. `delegating_agent` names the agent that
    made the call, for whoever reads the delegation; the monitor decides
    nothing by it. `runs` holds, for each monitor label whose trajectory holds
    the call, the run the subagent's trajectory joins, so a protocol counts
    the run's records. The state key that holds it is part of every monitored
    agent's input, and a value that does not fit this shape raises
    `ConfigurationError` when it is read.
    """

    tool_call_id: str
    delegating_agent: str
    runs: dict[str, DelegatedRun]
