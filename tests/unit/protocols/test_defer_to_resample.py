"""Defer to Resample runs the least suspicious acceptable sample, or its fallback takes over."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.contracts import Outcome, Resampling
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import (
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    HaltRun,
)

from .scripted_step import TRUSTED_STEP_ID, SampleCall, ScriptedPendingStep, StepDriver

ONE_SAMPLE = SampleCall(count=1, blocked_attempts=(), concurrently=False)
TWO_CONCURRENT_SAMPLES = SampleCall(count=2, blocked_attempts=(), concurrently=True)


@pytest.fixture
def sequential(defer_to_trusted_model: DeferToTrustedModel) -> DeferToResample:
    return DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.6,
        audit_threshold=0.9,
    )


@pytest.fixture
def parallel(defer_to_trusted_model: DeferToTrustedModel) -> DeferToResample:
    return DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.6,
        audit_threshold=0.9,
        resampling=Resampling.PARALLEL,
    )


@pytest.fixture
def defer_to_trusted(trusted_model: FakeListChatModel) -> DeferToTrusted:
    return DeferToTrusted(trusted_model=trusted_model, defer_threshold=0.6, audit_threshold=0.9)


@pytest.mark.parametrize("protocol_name", ["sequential", "parallel"])
def test_acceptable_first_sample_runs_without_resampling(
    request: pytest.FixtureRequest,
    drive: StepDriver,
    protocol_name: str,
) -> None:
    # Arrange
    protocol: DeferToResample = request.getfixturevalue(protocol_name)
    step = ScriptedPendingStep(suspicions=[0.1])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.executed_sample is step.drawn_samples[0]
    assert decision.response is step.drawn_samples[0].response
    assert decision.flagged is False
    assert step.sample_calls == [ONE_SAMPLE]


def test_sample_at_the_defer_threshold_counts_as_suspicious(
    sequential: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.6, 0.59])

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.outcome is Outcome.RESAMPLED
    assert decision.executed_sample is step.drawn_samples[1]


def test_sequential_resampling_stops_at_the_first_acceptable_sample(
    sequential: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.8, 0.3])

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.outcome is Outcome.RESAMPLED
    assert decision.executed_sample is step.drawn_samples[1]
    assert decision.response is step.drawn_samples[1].response
    assert decision.samples == tuple(step.drawn_samples)
    assert step.sample_calls == [ONE_SAMPLE, ONE_SAMPLE]


def test_sequential_resampling_runs_the_third_sample_as_in_the_plan(
    sequential: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.82, 0.71, 0.35])

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.outcome is Outcome.RESAMPLED
    assert decision.executed_sample is step.drawn_samples[2]
    assert decision.flagged is False
    assert len(decision.samples) == 3


def test_every_suspicious_sample_hands_the_step_to_the_fallback(
    sequential: DeferToResample,
    trusted_model: FakeListChatModel,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.8, 0.7, 0.95])

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.response.result[0].id == TRUSTED_STEP_ID
    assert decision.executed_sample is None
    assert decision.samples == tuple(step.drawn_samples)
    assert decision.flagged is False
    assert step.trusted_models == [trusted_model]
    assert step.sample_calls == [ONE_SAMPLE, ONE_SAMPLE, ONE_SAMPLE]


@pytest.mark.parametrize(
    ("suspicions", "flagged"),
    [
        ([0.95, 0.9, 0.99], True),
        ([1.0, 1.0, 1.0], True),
        ([0.95, 0.89, 0.99], False),
    ],
)
def test_step_is_flagged_when_its_lowest_suspicion_reaches_the_audit_threshold(
    sequential: DeferToResample,
    drive: StepDriver,
    suspicions: list[float],
    flagged: bool,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=suspicions)

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.flagged is flagged


def test_blatant_first_sample_with_a_benign_resample_is_not_flagged(
    sequential: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.99, 0.2])

    # Act
    decision = drive(sequential.decide(step))

    # Assert
    assert decision.outcome is Outcome.RESAMPLED
    assert decision.flagged is False


def test_auditing_turned_off_never_flags(
    defer_to_trusted_model: DeferToTrustedModel,
    drive: StepDriver,
) -> None:
    # Arrange
    protocol = DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.6,
        audit_threshold=None,
    )
    step = ScriptedPendingStep(suspicions=[1.0, 1.0, 1.0])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.flagged is False


def test_auditing_turned_off_still_flags_a_halted_step(drive: StepDriver) -> None:
    # Arrange
    protocol = DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=None)
    step = ScriptedPendingStep(suspicions=[1.0, 1.0, 1.0])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.HALTED
    assert decision.flagged is True


def test_an_audit_threshold_below_the_defer_threshold_flags_a_step_that_still_runs(
    defer_to_trusted_model: DeferToTrustedModel,
    drive: StepDriver,
) -> None:
    # Arrange: a flag is a mark for later review, so this is a review queue
    protocol = DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.8,
        audit_threshold=0.5,
    )
    step = ScriptedPendingStep(suspicions=[0.7])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.flagged is True
    assert decision.executed_sample is step.drawn_samples[0]


def test_parallel_resampling_draws_every_resample_at_once(
    parallel: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.8, 0.5, 0.3])

    # Act
    decision = drive(parallel.decide(step))

    # Assert
    assert step.sample_calls == [ONE_SAMPLE, TWO_CONCURRENT_SAMPLES]
    assert decision.outcome is Outcome.RESAMPLED
    assert decision.executed_sample is step.drawn_samples[2]


def test_parallel_resampling_runs_the_least_suspicious_sample_even_when_not_last(
    parallel: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.8, 0.3, 0.5])

    # Act
    decision = drive(parallel.decide(step))

    # Assert
    assert decision.executed_sample is step.drawn_samples[1]
    assert decision.response is step.drawn_samples[1].response
    assert decision.samples == tuple(step.drawn_samples)


def test_parallel_resampling_hands_an_all_suspicious_step_to_the_fallback(
    parallel: DeferToResample,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.8, 0.9, 0.7])

    # Act
    decision = drive(parallel.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert len(decision.samples) == 3
    assert step.sample_calls == [ONE_SAMPLE, TWO_CONCURRENT_SAMPLES]


def test_parallel_resampling_without_resamples_does_not_ask_for_zero_samples(
    defer_to_trusted_model: DeferToTrustedModel,
    drive: StepDriver,
) -> None:
    # Arrange
    protocol = DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.6,
        audit_threshold=0.9,
        max_resamples=0,
        resampling=Resampling.PARALLEL,
    )
    step = ScriptedPendingStep(suspicions=[0.8])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert step.sample_calls == [ONE_SAMPLE]


def test_halt_fallback_ends_the_run_when_no_sample_is_acceptable(drive: StepDriver) -> None:
    # Arrange
    protocol = DeferToResample(
        fallback=HaltRun(),
        defer_threshold=0.6,
        audit_threshold=0.9,
        max_resamples=1,
    )
    step = ScriptedPendingStep(suspicions=[0.7, 0.8])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.HALTED
    assert decision.flagged is True
    assert decision.samples == tuple(step.drawn_samples)


def test_negative_resample_count_is_rejected(defer_to_trusted_model: DeferToTrustedModel) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="max_resamples"):
        DeferToResample(
            fallback=defer_to_trusted_model,
            defer_threshold=0.6,
            audit_threshold=0.9,
            max_resamples=-1,
        )


def test_defer_threshold_outside_the_unit_interval_is_rejected(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="defer_threshold"):
        DeferToResample(fallback=defer_to_trusted_model, defer_threshold=1.2, audit_threshold=0.9)


def test_defer_to_trusted_never_resamples(
    defer_to_trusted: DeferToTrusted,
    trusted_model: FakeListChatModel,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.7])

    # Act
    decision = drive(defer_to_trusted.decide(step))

    # Assert
    assert defer_to_trusted.max_resamples == 0
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.flagged is False
    assert step.trusted_models == [trusted_model]
    assert step.sample_calls == [ONE_SAMPLE]


def test_defer_to_trusted_runs_an_acceptable_step(
    defer_to_trusted: DeferToTrusted,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.5])

    # Act
    decision = drive(defer_to_trusted.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.executed_sample is step.drawn_samples[0]
    assert step.trusted_models == []


def test_defer_to_trusted_flags_a_step_at_the_audit_threshold(
    defer_to_trusted: DeferToTrusted,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.9])

    # Act
    decision = drive(defer_to_trusted.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.flagged is True
