"""Trusted Monitoring runs every step once and flags the suspicious ones."""

from __future__ import annotations

import pytest

from langchain_sync_monitors.contracts import Outcome
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import TrustedMonitoring

from .scripted_step import SampleCall, ScriptedPendingStep, StepDriver


@pytest.fixture
def protocol() -> TrustedMonitoring:
    return TrustedMonitoring(audit_threshold=0.6)


def test_benign_step_runs_unflagged(protocol: TrustedMonitoring, drive: StepDriver) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.2])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.flagged is False
    assert decision.executed_sample is step.drawn_samples[0]
    assert decision.response is step.drawn_samples[0].response
    assert step.sample_calls == [SampleCall(count=1, blocked_attempts=(), concurrently=False)]


@pytest.mark.parametrize("suspicion", [0.6, 0.95])
def test_suspicious_step_is_flagged_but_still_runs(
    protocol: TrustedMonitoring,
    drive: StepDriver,
    suspicion: float,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[suspicion])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.flagged is True
    assert decision.samples == (step.drawn_samples[0],)
    assert decision.executed_sample is step.drawn_samples[0]


def test_step_just_below_the_audit_threshold_is_not_flagged(
    protocol: TrustedMonitoring,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.59])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.flagged is False


@pytest.mark.parametrize("audit_threshold", [-0.1, 1.5])
def test_audit_threshold_outside_the_unit_interval_is_rejected(audit_threshold: float) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="audit_threshold"):
        TrustedMonitoring(audit_threshold=audit_threshold)
