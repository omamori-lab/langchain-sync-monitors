"""Defer to Resample and its special case, Defer to Trusted."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Fallback,
    Outcome,
    PendingStep,
    Resampling,
    Sample,
    StepDecision,
)
from langchain_sync_monitors.options import check_enum_option, read_count_option
from langchain_sync_monitors.protocols.fallbacks import DeferToTrustedModel, check_fallback_option
from langchain_sync_monitors.thresholds import (
    DEFAULT_AUDIT_THRESHOLD,
    DEFAULT_DEFER_THRESHOLD,
    DefaultThreshold,
    resolve_threshold,
)


class DeferToResample(ControlProtocol):
    """Replace a suspicious step with a fresh sample of it, or hand it to a fallback.

    This is the basic Defer to Resample protocol of Ctrl-Z [@bhatt2025ctrlz].
    When the first sample's suspicion is at or above `defer_threshold`, the
    protocol draws up to `max_resamples` more samples of the same step, each a
    fresh call on the same conversation. It runs the least suspicious sample
    below the threshold: the outcome is ALLOWED when that is the only sample,
    and RESAMPLED otherwise. When no sample is below the threshold, `fallback`
    writes the step; Ctrl-Z uses the trusted model, `DeferToTrustedModel`.

    `Resampling.SEQUENTIAL` draws one sample at a time and stops at the first
    acceptable one, as Ctrl-Z's appendix J words the protocol.
    `Resampling.PARALLEL` draws every resample at once, which lowers latency but
    always costs `max_resamples` extra calls. `resampling` must be a
    `Resampling` member; a plain string raises `ConfigurationError`, as do a
    `fallback` that is not a `Fallback`, such as the `DeferToTrusted` protocol
    in place of `DeferToTrustedModel`, and a `max_resamples` that is not a
    whole number of at least 0.
    Resampling needs fresh draws, so a LangChain response cache on the agent's
    model defeats it, and the pending step emits a `CachedResampleWarning`.

    The step is flagged for audit when the lowest suspicion across its samples
    is at or above `audit_threshold`, the statistic Ctrl-Z found to work well
    (section 3.2, footnote 6). A flag is a mark in `monitor_log` for a person
    to review after the fact; it never blocks the step, which has already run
    or been replaced by the time anyone reads it. For that reason an
    `audit_threshold` below `defer_threshold` is allowed: it flags steps that
    still ran, a review queue of borderline steps below the deferral line.
    Every sample stays in the step's record as evidence, and an auditor should
    read the most suspicious one first. The agent never sees the rejected
    samples, so it cannot tell that it was resampled [@bhatt2025ctrlzpost].
    `audit_threshold=None` turns this flagging off; `HaltRun` still
    flags every step it halts, so with that fallback a step that reaches it
    is flagged anyway.

    Running the least suspicious sample has a cost against prompt injection.
    An agent can write text aimed at the monitor into its step, and an
    injection that fools the monitor only some of the time gets another try
    with every sample, so resampling turns it into a best-of-n attack
    [@terekhov2025adaptive] (sections 4.2 and 6). An injection that passes one
    sample in five gets through about half the time across three samples.

    One caveat comes from how LangChain composes middleware. If a middleware
    placed inside the monitor returns commands, only the commands of the last
    model call survive. The state then stays consistent only when the last
    sample drawn is the one committed. Sequential resampling commits the last
    sample whenever it accepts one, and a fallback's trusted call is the last
    call. Parallel resampling may commit an earlier sample, and concurrent
    calls pile up their commands. Keep the monitor last in the middleware list.
    A second monitor placed inside this one is such a middleware: it returns
    its record as a command, so it loses its judgement of each rejected
    sample, and parallel resampling under `ainvoke()` leaves it one record per
    resample drawn at once, all under the same step number.
    `check_monitor_placement` warns about it unless `max_resamples=0` and the
    fallback is `HaltRun()`, which call the model at most once per step.
    """

    def __init__(
        self,
        *,
        fallback: Fallback,
        defer_threshold: float | DefaultThreshold = DEFAULT_DEFER_THRESHOLD,
        audit_threshold: float | DefaultThreshold | None = DEFAULT_AUDIT_THRESHOLD,
        max_resamples: int = 2,
        resampling: Resampling = Resampling.SEQUENTIAL,
    ) -> None:
        """Keep the configuration, warning for each threshold left at its default.

        Two resamples, three samples in all, is the number Ctrl-Z draws
        [@bhatt2025ctrlz].
        """
        check_fallback_option(fallback, parameter_name="fallback")
        check_enum_option(resampling, option_type=Resampling, parameter_name="resampling")
        self.fallback = fallback
        self.defer_threshold = resolve_threshold(
            parameter_name="defer_threshold",
            threshold=defer_threshold,
        )
        self.audit_threshold = (
            None
            if audit_threshold is None
            else resolve_threshold(parameter_name="audit_threshold", threshold=audit_threshold)
        )
        self.max_resamples = read_count_option(
            max_resamples,
            parameter_name="max_resamples",
            minimum=0,
        )
        self.resampling = resampling

    async def decide(self, step: PendingStep) -> StepDecision:
        """Sample the step, resample it if suspicious, and run the best acceptable sample."""
        samples = await self.collect_samples(step)
        flagged = self.is_flagged(samples)
        acceptable = [sample for sample in samples if not self.is_suspicious(sample)]
        if not acceptable:
            return await self.fallback.take_over(step, samples=samples, flagged=flagged)
        chosen = min(acceptable, key=lambda sample: sample.verdict.suspicion)
        return StepDecision(
            outcome=Outcome.ALLOWED if len(samples) == 1 else Outcome.RESAMPLED,
            response=chosen.response,
            samples=samples,
            executed_sample=chosen,
            flagged=flagged,
        )

    async def collect_samples(self, step: PendingStep) -> tuple[Sample, ...]:
        """Draw the first sample and, when it is suspicious, the resamples."""
        samples = await step.sample(count=1)
        if not self.is_suspicious(samples[0]) or self.max_resamples == 0:
            return samples
        if self.resampling is Resampling.PARALLEL:
            return samples + await step.sample(count=self.max_resamples, concurrently=True)
        return samples + await self.resample_sequentially(step)

    async def resample_sequentially(self, step: PendingStep) -> tuple[Sample, ...]:
        """Draw resamples one at a time, stopping at the first acceptable one."""
        resamples: list[Sample] = []
        for _ in range(self.max_resamples):
            resamples.extend(await step.sample(count=1))
            if not self.is_suspicious(resamples[-1]):
                break
        return tuple(resamples)

    def is_suspicious(self, sample: Sample) -> bool:
        """Tell whether a sample's suspicion reaches the defer threshold."""
        return sample.verdict.suspicion >= self.defer_threshold

    def is_flagged(self, samples: tuple[Sample, ...]) -> bool:
        """Tell whether even the least suspicious sample reaches the audit threshold."""
        if self.audit_threshold is None:
            return False
        return min(sample.verdict.suspicion for sample in samples) >= self.audit_threshold


class DeferToTrusted(DeferToResample):
    """Hand a suspicious step to the trusted model, without resampling.

    This is Defer to Trusted from AI Control [@greenblatt2023aicontrol]: a step
    whose suspicion is at or above `defer_threshold` is written by
    `trusted_model` instead, and a step at or above `audit_threshold` is also
    flagged. It is Defer to Resample with zero resamples. As there, a flag is a
    mark for review after the fact and never blocks the step, so an
    `audit_threshold` below `defer_threshold` flags steps that still ran.
    """

    def __init__(
        self,
        *,
        trusted_model: str | BaseChatModel,
        defer_threshold: float | DefaultThreshold = DEFAULT_DEFER_THRESHOLD,
        audit_threshold: float | DefaultThreshold | None = DEFAULT_AUDIT_THRESHOLD,
    ) -> None:
        """Build Defer to Resample with no resamples and the trusted model as fallback."""
        super().__init__(
            fallback=DeferToTrustedModel(trusted_model=trusted_model),
            defer_threshold=defer_threshold,
            audit_threshold=audit_threshold,
            max_resamples=0,
        )
