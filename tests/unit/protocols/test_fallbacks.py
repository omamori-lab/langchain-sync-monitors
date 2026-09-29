"""The fallbacks write a step in place of the untrusted model."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import BlockedAttempt, Outcome
from langchain_sync_monitors.protocols import DEFAULT_HALT_MESSAGE, DeferToTrustedModel, HaltRun

from .scripted_step import TRUSTED_STEP_ID, ScriptedPendingStep, StepDriver


@pytest.fixture
def step() -> ScriptedPendingStep:
    return ScriptedPendingStep(suspicions=[0.9, 0.8])


@pytest.fixture
def blocked_attempt() -> BlockedAttempt:
    return BlockedAttempt(proposal=AIMessage(content="rm -rf /", id="blocked"), feedback="no")


def test_trusted_model_given_as_an_instance_is_kept(trusted_model: FakeListChatModel) -> None:
    # Act
    fallback = DeferToTrustedModel(trusted_model=trusted_model)

    # Assert
    assert fallback.trusted_model is trusted_model


@pytest.mark.parametrize("flagged", [True, False])
def test_trusted_model_writes_the_step_and_keeps_the_evidence(
    defer_to_trusted_model: DeferToTrustedModel,
    trusted_model: FakeListChatModel,
    step: ScriptedPendingStep,
    blocked_attempt: BlockedAttempt,
    drive: StepDriver,
    flagged: bool,
) -> None:
    # Arrange
    samples = (step.build_next_sample(), step.build_next_sample())

    # Act
    decision = drive(
        defer_to_trusted_model.take_over(
            step,
            samples=samples,
            flagged=flagged,
            blocked_attempts=(blocked_attempt,),
        ),
    )

    # Assert
    assert decision.outcome is Outcome.DEFERRED_TO_TRUSTED
    assert decision.response.result[0].id == TRUSTED_STEP_ID
    assert decision.executed_sample is None
    assert decision.samples == samples
    assert decision.flagged is flagged
    assert decision.blocked_attempts == (blocked_attempt,)
    assert step.trusted_models == [trusted_model]
    assert step.sample_calls == []


def test_halt_ends_the_run_with_a_final_message_and_a_flag(
    step: ScriptedPendingStep,
    blocked_attempt: BlockedAttempt,
    drive: StepDriver,
) -> None:
    # Arrange
    samples = (step.build_next_sample(),)

    # Act
    decision = drive(
        HaltRun().take_over(
            step,
            samples=samples,
            flagged=False,
            blocked_attempts=(blocked_attempt,),
        ),
    )

    # Assert
    [halt_message] = decision.response.result
    assert isinstance(halt_message, AIMessage)
    assert halt_message.content == DEFAULT_HALT_MESSAGE
    assert halt_message.tool_calls == []
    assert decision.outcome is Outcome.HALTED
    assert decision.flagged is True
    assert decision.executed_sample is None
    assert decision.samples == samples
    assert decision.blocked_attempts == (blocked_attempt,)
    assert step.sample_calls == []
    assert step.trusted_models == []


def test_halt_uses_a_custom_message(step: ScriptedPendingStep, drive: StepDriver) -> None:
    # Arrange
    fallback = HaltRun(message="Stopped for review.")

    # Act
    decision = drive(fallback.take_over(step, samples=(), flagged=True))

    # Assert
    assert decision.response.result[0].content == "Stopped for review."


def test_every_halt_message_gets_a_fresh_monitor_id(
    step: ScriptedPendingStep,
    drive: StepDriver,
) -> None:
    # Arrange
    fallback = HaltRun()

    # Act
    first = drive(fallback.take_over(step, samples=(), flagged=True))
    second = drive(fallback.take_over(step, samples=(), flagged=True))

    # Assert
    first_id = first.response.result[0].id
    second_id = second.response.result[0].id
    assert first_id is not None
    assert second_id is not None
    assert first_id.startswith("monitor-")
    assert second_id.startswith("monitor-")
    assert first_id != second_id


def test_default_halt_message_is_marked_as_the_monitor_s() -> None:
    # Act
    message = HaltRun().message

    # Assert
    assert message == DEFAULT_HALT_MESSAGE
    assert message.startswith("[Safety monitor]")
