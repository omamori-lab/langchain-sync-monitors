"""A scripted pending step and two ways to drive a protocol, for the protocol tests.

`ScriptedPendingStep` hands out samples whose verdicts the test scripts, and
records every call a protocol makes. Its methods never suspend, so a protocol
driven against it finishes with a single `send(None)`, as it must under the
middleware's synchronous `invoke()` path.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass

from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    PendingStep,
    Sample,
    StepDecision,
    StepRecord,
    Verdict,
)

TRUSTED_STEP_ID = "trusted-step"
"""The id of the message the scripted step returns for a trusted model's step."""

StepDriver = Callable[[Coroutine[object, None, StepDecision]], StepDecision]
"""Runs a protocol's or fallback's coroutine to its decision."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SampleCall:
    """One call a protocol made to `sample`, with the arguments it passed."""

    count: int
    blocked_attempts: tuple[BlockedAttempt, ...]
    concurrently: bool


class ScriptedPendingStep(PendingStep):
    """A pending step whose samples carry scripted suspicion scores.

    Drawing more samples than the test scripted fails the test, so a protocol
    that oversamples cannot pass unnoticed. Without `blocks_in_thread`, the
    thread's blocks are the ones in `previous_records`, as `PendingStep`
    counts them by default.
    """

    def __init__(
        self,
        *,
        suspicions: Sequence[float],
        previous_records: Sequence[StepRecord] = (),
        blocks_in_thread: int | None = None,
        new_subagent_blocks: int = 0,
    ) -> None:
        """Queue the scripted suspicions and start with no recorded calls."""
        self.remaining_suspicions = deque(suspicions)
        self.earlier_records = tuple(previous_records)
        self.thread_blocks = blocks_in_thread
        self.subagent_blocks = new_subagent_blocks
        self.sample_calls: list[SampleCall] = []
        self.drawn_samples: list[Sample] = []
        self.trusted_models: list[BaseChatModel] = []

    @property
    def previous_records(self) -> tuple[StepRecord, ...]:
        """The records the test gave for earlier steps of the run."""
        return self.earlier_records

    @property
    def blocks_in_thread(self) -> int:
        """The thread's blocks the test gave, or else the blocks in the earlier records."""
        if self.thread_blocks is None:
            return super().blocks_in_thread
        return self.thread_blocks

    @property
    def new_subagent_blocks(self) -> int:
        """The subagent blocks since this agent's last step that the test gave."""
        return self.subagent_blocks

    async def sample(
        self,
        *,
        count: int,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
        concurrently: bool = False,
    ) -> tuple[Sample, ...]:
        """Record the call and return the next `count` scripted samples."""
        self.sample_calls.append(
            SampleCall(count=count, blocked_attempts=blocked_attempts, concurrently=concurrently),
        )
        if count > len(self.remaining_suspicions):
            message = f"the protocol drew {count} samples, but only the scripted ones exist"
            raise AssertionError(message)
        return tuple(self.build_next_sample() for _ in range(count))

    async def request_trusted_step(self, *, trusted_model: BaseChatModel) -> ModelResponse:
        """Record the trusted model and return a step marked as the trusted model's."""
        self.trusted_models.append(trusted_model)
        return ModelResponse(result=[AIMessage(content="trusted step", id=TRUSTED_STEP_ID)])

    def build_next_sample(self) -> Sample:
        """Build a numbered sample carrying the next scripted suspicion."""
        number = len(self.drawn_samples) + 1
        proposal = AIMessage(content=f"proposal {number}", id=f"proposal-{number}")
        verdict = Verdict(suspicion=self.remaining_suspicions.popleft(), reason=f"reason {number}")
        sample = Sample(
            response=ModelResponse(result=[proposal]), proposal=proposal, verdict=verdict
        )
        self.drawn_samples.append(sample)
        return sample


def build_step_record(*, blocked_count: int) -> StepRecord:
    """Build a record of an earlier step in which Auto Mode blocked `blocked_count` samples."""
    return StepRecord(
        agent="main",
        monitor="monitor",
        step_number=1,
        outcome="steered",
        flagged=True,
        blocked_count=blocked_count,
        samples=[],
    )


def run_on_event_loop(coroutine: Coroutine[object, None, StepDecision]) -> StepDecision:
    """Run the coroutine on a fresh event loop, as `ainvoke()` does."""
    return asyncio.run(coroutine)


def run_without_event_loop(coroutine: Coroutine[object, None, StepDecision]) -> StepDecision:
    """Finish the coroutine with one `send(None)` and no event loop, as `invoke()` does.

    A coroutine that suspends has awaited something other than the pending
    step, which would hang or fail under `invoke()`, so it fails the test.
    """
    try:
        coroutine.send(None)
    except StopIteration as finished:
        decision: StepDecision = finished.value
        return decision
    coroutine.close()
    message = "the coroutine suspended, so it awaited something other than the pending step"
    raise AssertionError(message)
