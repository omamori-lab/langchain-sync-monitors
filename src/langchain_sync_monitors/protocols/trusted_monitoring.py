"""Trusted Monitoring: score every step and flag the suspicious ones, never intervene."""

from __future__ import annotations

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Outcome,
    PendingStep,
    StepDecision,
)
from langchain_sync_monitors.thresholds import (
    DEFAULT_TRUSTED_MONITORING_AUDIT_THRESHOLD,
    DefaultThreshold,
    resolve_threshold,
)


class TrustedMonitoring(ControlProtocol):
    """Observe and flag, never change what the agent does.

    Each step is sampled once and always runs. It is flagged for audit when its
    suspicion reaches `audit_threshold`, as in the Trusted Monitoring protocol
    of AI Control [@greenblatt2023aicontrol]; Ctrl-Z names that line the
    auditing threshold (section 3.1.2) [@bhatt2025ctrlz]. Here a flag marks
    the step in `monitor_log` for a person to review; it does not stop the
    step.
    """

    def __init__(
        self,
        *,
        audit_threshold: float | DefaultThreshold = DEFAULT_TRUSTED_MONITORING_AUDIT_THRESHOLD,
    ) -> None:
        """Keep the audit threshold, warning when it is the uncalibrated default."""
        self.audit_threshold = resolve_threshold(
            parameter_name="audit_threshold",
            threshold=audit_threshold,
        )

    async def decide(self, step: PendingStep) -> StepDecision:
        """Draw one sample, run it, and flag it when it looks suspicious."""
        [sample] = await step.sample(count=1)
        return StepDecision(
            outcome=Outcome.ALLOWED,
            response=sample.response,
            samples=(sample,),
            executed_sample=sample,
            flagged=sample.verdict.suspicion >= self.audit_threshold,
        )
