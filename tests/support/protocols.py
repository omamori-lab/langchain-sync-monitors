"""Small control protocols that exercise every path of a pending step."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    ControlProtocol,
    Outcome,
    PendingStep,
    Sample,
    StepDecision,
    StepRecord,
)

SUSPICION_THRESHOLD = 0.5
HALT_MESSAGE = "[Safety monitor] Stopped by the safety monitor."
FEEDBACK_PREFIX = "[Safety monitor] Blocked: "


def build_halt_decision(
    *,
    samples: tuple[Sample, ...],
    blocked_attempts: tuple[BlockedAttempt, ...] = (),
) -> StepDecision:
    message = AIMessage(content=HALT_MESSAGE, id=f"monitor-{uuid4()}")
    return StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[message]),
        samples=samples,
        executed_sample=None,
        flagged=True,
        blocked_attempts=blocked_attempts,
    )


@dataclass(kw_only=True)
class AcceptFirst(ControlProtocol):
    """Draws one sample and runs it; flags it when the monitor is suspicious."""

    seen_previous_records: list[tuple[StepRecord, ...]] = field(default_factory=list)

    async def decide(self, step: PendingStep) -> StepDecision:
        self.seen_previous_records.append(step.previous_records)
        [sample] = await step.sample(count=1)
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=sample.verdict.suspicion >= SUSPICION_THRESHOLD,
        )


@dataclass(kw_only=True)
class ResampleThenDefer(ControlProtocol):
    """Resamples a suspicious step, runs the least suspicious acceptable sample or defers.

    With `extra_samples=0` it defers straight to the trusted model.
    """

    trusted_model: BaseChatModel
    extra_samples: int = 2
    concurrently: bool = False

    async def decide(self, step: PendingStep) -> StepDecision:
        samples = await step.sample(count=1)
        if samples[0].verdict.suspicion >= SUSPICION_THRESHOLD:
            samples += await step.sample(count=self.extra_samples, concurrently=self.concurrently)
        acceptable = [
            sample for sample in samples if sample.verdict.suspicion < SUSPICION_THRESHOLD
        ]
        if acceptable:
            chosen = min(acceptable, key=lambda sample: sample.verdict.suspicion)
            return StepDecision(
                outcome=Outcome.ALLOWED if len(samples) == 1 else Outcome.RESAMPLED,
                response=chosen.response,
                samples=samples,
                executed_sample=chosen,
                flagged=False,
            )
        response = await step.request_trusted_step(trusted_model=self.trusted_model)
        return StepDecision(
            outcome=Outcome.DEFERRED_TO_TRUSTED,
            response=response,
            samples=samples,
            executed_sample=None,
            flagged=True,
        )


@dataclass(kw_only=True)
class HaltAfterOneSample(ControlProtocol):
    """Judges one sample, then always ends the run."""

    async def decide(self, step: PendingStep) -> StepDecision:
        samples = await step.sample(count=1)
        return build_halt_decision(samples=samples)


@dataclass(kw_only=True)
class SteerWithFeedback(ControlProtocol):
    """Blocks a suspicious sample and retries with feedback, like Auto Mode; halts at the limit."""

    max_blocks: int = 3

    async def decide(self, step: PendingStep) -> StepDecision:
        samples: list[Sample] = []
        blocked: list[BlockedAttempt] = []
        while len(blocked) < self.max_blocks:
            [sample] = await step.sample(count=1, blocked_attempts=tuple(blocked))
            samples.append(sample)
            if sample.verdict.suspicion < SUSPICION_THRESHOLD:
                return StepDecision(
                    outcome=Outcome.STEERED if blocked else Outcome.ALLOWED,
                    response=sample.response,
                    samples=tuple(samples),
                    executed_sample=sample,
                    flagged=bool(blocked),
                    blocked_attempts=tuple(blocked),
                )
            feedback = FEEDBACK_PREFIX + sample.verdict.reason
            blocked.append(BlockedAttempt(proposal=sample.proposal, feedback=feedback))
        return build_halt_decision(samples=tuple(samples), blocked_attempts=tuple(blocked))


@dataclass(kw_only=True)
class AwaitsEventLoop(ControlProtocol):
    """Awaits `asyncio.sleep` after sampling: real async work that `invoke()` cannot drive."""

    delay: float = 0.0
    finished_cleanly: list[bool] = field(default_factory=list)

    async def decide(self, step: PendingStep) -> StepDecision:
        [sample] = await step.sample(count=1)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.finished_cleanly.append(True)
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=False,
        )
