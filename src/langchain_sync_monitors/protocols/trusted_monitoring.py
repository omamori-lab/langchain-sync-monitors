"""Trusted Monitoring: score every step and flag the suspicious ones, never intervene."""

from __future__ import annotations

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Outcome,
    PendingStep,
    StepDecision,
)
from langchain_sync_monitors.thresholds import (
    DEFAULT_FLAG_THRESHOLD,
    DefaultThreshold,
    resolve_threshold,
)


class TrustedMonitoring(ControlProtocol):
    """Observe and flag, never change what the agent does.

    Each step is sampled once and always runs. It is flagged for audit when its
    suspicion reaches `flag_threshold`, the Trusted Monitoring protocol of AI
    Control [@greenblatt2023aicontrol]. Here a flag marks the step in
    `monitor_log` for a person to review; it does not stop the step.
    """

    def __init__(
        self, *, flag_threshold: float | DefaultThreshold = DEFAULT_FLAG_THRESHOLD
    ) -> None:
        """Keep the flag threshold, warning when it is the uncalibrated default."""
        self.flag_threshold = resolve_threshold(
            parameter_name="flag_threshold",
            threshold=flag_threshold,
        )

    async def decide(self, step: PendingStep) -> StepDecision:
        """Draw one sample, run it, and flag it when it looks suspicious."""
        [sample] = await step.sample(count=1)
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=sample.verdict.suspicion >= self.flag_threshold,
        )
