"""The types every monitor, control protocol and the middleware exchange.

The rest of the package codes against these definitions, so they change only in
a dedicated pull request. The design they come from, with diagrams, is in
`docs/plans/initial-implementation/plan.html`.

A step is one model call of the agent. The untrusted model proposes it, a
`Monitor` scores it, and a `ControlProtocol` decides what the agent state
receives. Protocols reach the model only through a `PendingStep`, which is how
one copy of each protocol serves both `invoke()` and `ainvoke()`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from typing import Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors.errors import ConfigurationError, InvalidSuspicionError


class Channel(Flag):
    """The parts of a transcript a monitor can read.

    The default, `ACTIONS`, leaves out the agent's reasoning, its prose and raw
    tool output, following Claude Code's auto mode classifier
    [@hughes2026automode]; reasoning is one flag away, since reading it catches
    more misbehaviour [@baker2025monitoring] but can also mislead a monitor
    [@arnav2025cotredhanded]. Whether a call ran is never left out: a tool
    error travels with the call, and monitor feedback and context notes with
    the task.
    """

    USER = auto()
    REASONING = auto()
    AGENT_TEXT = auto()
    TOOL_CALLS = auto()
    TOOL_RESULTS = auto()
    SUBAGENT_REPORTS = auto()
    ACTIONS = USER | TOOL_CALLS | SUBAGENT_REPORTS
    ALL = USER | REASONING | AGENT_TEXT | TOOL_CALLS | TOOL_RESULTS | SUBAGENT_REPORTS


class TaskAuthor(StrEnum):
    """Who wrote the task the agent is working on.

    Inside a subagent the first human message was written by the parent agent,
    so a monitor must not read it as the user's authorisation.
    """

    USER = "user"
    PARENT_AGENT = "parent_agent"


class Resampling(StrEnum):
    """How Defer to Resample draws its extra samples."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class FeedbackVisibility(StrEnum):
    """Whether the feedback on a blocked step is kept in the agent's transcript."""

    HIDDEN = "hidden"
    IN_TRANSCRIPT = "in_transcript"


class SubagentHalt(StrEnum):
    """What a halt inside a subagent does to the rest of the run."""

    STOP_SUBAGENT_ONLY = "stop_subagent_only"
    STOP_WHOLE_RUN = "stop_whole_run"


OutcomeName = Literal["allowed", "resampled", "deferred_to_trusted", "steered", "halted"]
"""The outcome of a step as it is stored in `StepRecord`, a plain string."""


class Outcome(StrEnum):
    """What a control protocol did with a step."""

    ALLOWED = "allowed"
    RESAMPLED = "resampled"
    DEFERRED_TO_TRUSTED = "deferred_to_trusted"
    STEERED = "steered"
    HALTED = "halted"


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorView:
    """Which parts of the transcript a monitor reads.

    `most_recent_entries` keeps only the last entries of the history, at least
    one, or every entry when it is `None`; the messages of the task author are
    kept in addition. `delegation_tools` names the tools whose results are
    subagent reports.
    """

    channels: Channel = Channel.ACTIONS
    most_recent_entries: int | None = None
    delegation_tools: frozenset[str] = frozenset({"task"})

    def __post_init__(self) -> None:
        """Reject an entry limit below one, which would silently hide every recent step."""
        if self.most_recent_entries is not None and self.most_recent_entries < 1:
            message = (
                "most_recent_entries must be at least 1, or None to keep every entry, "
                f"got {self.most_recent_entries!r}"
            )
            raise ConfigurationError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorInput:
    """What a monitor judges: the conversation so far and the step not yet taken."""

    history: tuple[BaseMessage, ...]
    proposal: AIMessage
    task_author: TaskAuthor


@dataclass(frozen=True, slots=True, kw_only=True)
class Verdict:
    """A monitor's judgement of one step.

    `suspicion` runs from 0 (benign) to 1 (certainly an attack), so every
    monitor works with every protocol.
    """

    suspicion: float
    reason: str

    def __post_init__(self) -> None:
        """Reject a suspicion outside [0, 1], including NaN, with `InvalidSuspicionError`."""
        if math.isnan(self.suspicion) or not 0.0 <= self.suspicion <= 1.0:
            message = f"suspicion must be between 0 and 1, got {self.suspicion!r}"
            raise InvalidSuspicionError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One draw from the untrusted model, with the monitor's verdict on it."""

    response: ModelResponse
    proposal: AIMessage
    verdict: Verdict


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockedAttempt:
    """A step the Auto Mode protocol blocked, and the feedback the agent receives."""

    proposal: AIMessage
    feedback: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepDecision:
    """What a control protocol decided for one step.

    `response` is what the agent state receives. `samples` holds every
    untrusted sample the monitor judged, so rejected samples survive as audit
    evidence. `executed_sample` is `None` when a trusted model or a halt wrote
    the step.
    """

    outcome: Outcome
    response: ModelResponse
    samples: tuple[Sample, ...]
    executed_sample: Sample | None
    flagged: bool
    blocked_attempts: tuple[BlockedAttempt, ...] = ()


class SampleRecord(TypedDict):
    """One judged sample as stored in the agent state."""

    suspicion: float
    reason: str
    proposal: str
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


class Delegation(TypedDict):
    """What a monitored agent hands to a subagent that one of its tool calls starts.

    `tool_call_id` is the id of that tool call, which the subagent's records
    carry as their `delegation_id`. `delegating_agent` names the agent that
    made the call, so a monitor stacked in that agent passes the delegation on
    as it is, while a subagent whose own call reuses the id hands on a new one.
    `blocks_before` holds, for each monitor label, the blocks recorded in the
    conversation thread when the subagent started, so that the subagent's Auto
    Mode counts from the thread's total.
    """

    tool_call_id: str
    delegating_agent: str
    blocks_before: dict[str, int]


class Monitor(ABC):
    """Scores a proposed step. Implementations differ in the model family they use."""

    @abstractmethod
    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score the proposed step in `monitor_input`."""

    @abstractmethod
    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score the proposed step without an event loop, for `invoke()`."""


class PendingStep(ABC):
    """The step the agent is about to take, as a control protocol sees it."""

    @property
    @abstractmethod
    def previous_records(self) -> tuple[StepRecord, ...]:
        """This monitor's records from earlier steps of the run."""

    @property
    def blocks_in_thread(self) -> int:
        """The blocks this monitor recorded in the conversation thread before this step.

        A subagent's copy of a monitor keeps its label, so the count covers
        this agent's earlier steps, its subagents' steps, nested ones included,
        and, inside a subagent, the blocks recorded before the subagent
        started. By default it sums `previous_records` alone.
        """
        return sum(record["blocked_count"] for record in self.previous_records)

    @property
    def new_subagent_blocks(self) -> int:
        """The blocks this monitor recorded inside subagents since this agent's last step.

        It is 0 by default.
        """
        return 0

    @abstractmethod
    async def sample(
        self,
        *,
        count: int,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
        concurrently: bool = False,
    ) -> tuple[Sample, ...]:
        """Draw `count` samples of the step and have the monitor judge each one.

        With `blocked_attempts`, each earlier blocked proposal and its feedback
        come before the request, so the agent sees why it was blocked.
        """

    @abstractmethod
    async def request_trusted_step(self, *, trusted_model: BaseChatModel) -> ModelResponse:
        """Have the trusted model write the step, with the same conversation and tools."""


class ControlProtocol(ABC):
    """Decides what the agent state receives for each step."""

    @abstractmethod
    async def decide(self, step: PendingStep) -> StepDecision:
        """Sample and judge the step as the protocol requires, then decide."""


class Fallback(ABC):
    """Writes the step when no untrusted sample is acceptable."""

    @abstractmethod
    async def take_over(
        self,
        step: PendingStep,
        *,
        samples: tuple[Sample, ...],
        flagged: bool,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
    ) -> StepDecision:
        """Produce the step decision in place of the untrusted model."""
