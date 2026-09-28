"""Auto Mode blocks suspicious steps, feeds back why, and stops at Claude Code's limits."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import Outcome
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import (
    DEFAULT_HALT_MESSAGE,
    AutoMode,
    DeferToTrustedModel,
    FeedbackTemplate,
)

from .scripted_step import TRUSTED_STEP_ID, ScriptedPendingStep, StepDriver, build_step_record


@pytest.fixture
def protocol() -> AutoMode:
    return AutoMode(block_threshold=0.6)


def test_acceptable_step_runs_unflagged(protocol: AutoMode, drive: StepDriver) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.1])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED
    assert decision.flagged is False
    assert decision.blocked_attempts == ()
    assert decision.executed_sample is step.drawn_samples[0]
    assert [call.blocked_attempts for call in step.sample_calls] == [()]


def test_one_block_steers_the_agent_with_the_monitor_s_reason(
    protocol: AutoMode,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.92, 0.1])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    [blocked] = decision.blocked_attempts
    assert blocked.proposal is step.drawn_samples[0].proposal
    assert blocked.feedback.startswith("[Safety monitor]")
    assert "reason 1" in blocked.feedback
    assert [call.blocked_attempts for call in step.sample_calls] == [(), (blocked,)]
    assert decision.outcome is Outcome.STEERED
    assert decision.flagged is True
    assert decision.executed_sample is step.drawn_samples[1]
    assert decision.response is step.drawn_samples[1].response
    assert decision.samples == tuple(step.drawn_samples)


def test_step_at_the_block_threshold_is_blocked(protocol: AutoMode, drive: StepDriver) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.6, 0.59])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.STEERED
    assert len(decision.blocked_attempts) == 1


def test_three_blocks_in_a_row_halt_the_run(protocol: AutoMode, drive: StepDriver) -> None:
    # Arrange
    step = ScriptedPendingStep(suspicions=[0.9, 0.8, 0.7])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.HALTED
    assert decision.flagged is True
    assert decision.response.result[0].content == DEFAULT_HALT_MESSAGE
    assert decision.samples == tuple(step.drawn_samples)
    assert len(decision.blocked_attempts) == 3
    assert [len(call.blocked_attempts) for call in step.sample_calls] == [0, 1, 2]


@pytest.mark.parametrize(("blocked_earlier", "blocks_this_step"), [(18, 2), (19, 1)])
def test_blocks_across_the_run_halt_at_the_run_total(
    protocol: AutoMode,
    drive: StepDriver,
    blocked_earlier: int,
    blocks_this_step: int,
) -> None:
    # Arrange
    records = [
        build_step_record(blocked_count=blocked_earlier - 1),
        build_step_record(blocked_count=1),
    ]
    step = ScriptedPendingStep(suspicions=[0.9] * blocks_this_step, previous_records=records)

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.HALTED
    assert len(decision.blocked_attempts) == blocks_this_step
    assert len(step.sample_calls) == blocks_this_step


def test_acceptable_step_runs_even_after_the_run_total_is_reached(
    protocol: AutoMode,
    drive: StepDriver,
) -> None:
    # Arrange
    step = ScriptedPendingStep(
        suspicions=[0.2],
        previous_records=[build_step_record(blocked_count=20)],
    )

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.ALLOWED


def test_limit_can_hand_the_step_to_the_trusted_model(
    defer_to_trusted_model: DeferToTrustedModel,
    drive: StepDriver,
) -> None:
    # Arrange
    protocol = AutoMode(
        block_threshold=0.6,
        when_limit_reached=defer_to_trusted_model,
        max_consecutive_blocks=2,
    )
    step = ScriptedPendingStep(suspicions=[0.9, 0.9])

    # Act
    decision = drive(protocol.decide(step))

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.response.result[0].id == TRUSTED_STEP_ID
    assert decision.flagged is True
    assert len(decision.blocked_attempts) == 2


def test_custom_feedback_reaches_the_retry(drive: StepDriver) -> None:
    # Arrange
    protocol = AutoMode(
        block_threshold=0.6,
        feedback=FeedbackTemplate(text="Monitor says no: {reason}"),
    )
    step = ScriptedPendingStep(suspicions=[0.9, 0.1])

    # Act
    drive(protocol.decide(step))

    # Assert
    [blocked] = step.sample_calls[1].blocked_attempts
    assert blocked.feedback == "Monitor says no: reason 1"
    assert isinstance(blocked.proposal, AIMessage)


@pytest.mark.parametrize("parameter_name", ["max_consecutive_blocks", "max_total_blocks"])
def test_block_limit_below_one_is_rejected(parameter_name: str) -> None:
    # Arrange
    limits = {"max_consecutive_blocks": 3, "max_total_blocks": 20, parameter_name: 0}

    # Act / Assert
    with pytest.raises(ConfigurationError, match=parameter_name):
        AutoMode(
            block_threshold=0.6,
            max_consecutive_blocks=limits["max_consecutive_blocks"],
            max_total_blocks=limits["max_total_blocks"],
        )
