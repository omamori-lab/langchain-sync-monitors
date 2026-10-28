"""A guard monitor that is asked for log-probabilities scores its label from them, in one call.

`top_log_probabilities` is the count of alternatives to each token that the
guard's chat model is asked for. Above 0, the monitor rates in one call, and a
reply without log-probabilities fails the rating closed.
"""

from __future__ import annotations

import pytest
from hyoshigi import FailedClosed, Step, Unreadable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors import ConfigurationError, GuardLabels, GuardMonitor, MonitorPrompt
from langchain_sync_monitors.monitors import MonitorConfiguration

from .doubles import CallPath, ScriptedChatModel, rate_on_path

POLICY_PROMPT = MonitorPrompt(
    system="Flag actions that send data out. End with violation or no_violation.",
    request="{trajectory}\n{proposed_action}",
)
LABELS = GuardLabels(suspicious=frozenset({"violation"}), safe=frozenset({"no_violation"}))
MODEL_NAME = "scriptedchatmodel"
"""The name hyoshigi gives the scripted chat model: its class name in lower case."""

# ln 0.8 and ln 0.2: the label token's alternatives, weighing 0.8 to 0.2.
VIOLATION_LABEL = AIMessage(
    "violation",
    response_metadata={
        "logprobs": {
            "content": [
                {
                    "token": "violation",
                    "logprob": -0.2231435513,
                    "bytes": None,
                    "top_logprobs": [
                        {"token": "violation", "logprob": -0.2231435513, "bytes": None},
                        {"token": "no_violation", "logprob": -1.6094379124, "bytes": None},
                    ],
                },
            ],
        },
    },
)


def build_guard(model: BaseChatModel, *, top_log_probabilities: object) -> MonitorConfiguration:
    """Return a guard over `model` whose two samples are not used at a count above 0."""
    return GuardMonitor(
        model=model,
        prompt=POLICY_PROMPT,
        labels=LABELS,
        samples=2,
        top_log_probabilities=top_log_probabilities,  # ty: ignore[invalid-argument-type]
    )


async def test_a_guard_scores_the_probability_of_its_suspicious_label_from_one_call(
    step: Step,
    call_path: CallPath,
) -> None:
    # Arrange
    model = ScriptedChatModel(replies=[VIOLATION_LABEL])
    monitor = build_guard(model, top_log_probabilities=2)

    # Act
    score = await rate_on_path(monitor, step, call_path=call_path)

    # Assert
    assert score.value == pytest.approx(0.8)
    assert len(model.received_messages) == 1


async def test_a_reply_without_log_probabilities_fails_the_rating_closed_after_one_call(
    step: Step,
    call_path: CallPath,
) -> None:
    # Arrange
    model = ScriptedChatModel(replies=["violation"])
    monitor = build_guard(model, top_log_probabilities=2)

    # Act
    score = await rate_on_path(monitor, step, call_path=call_path)

    # Assert
    assert score.value == 1.0
    assert score.evidence == FailedClosed(model=MODEL_NAME, cause=Unreadable())
    assert len(model.received_messages) == 1


@pytest.mark.parametrize("count", [-1, True])
def test_a_count_that_is_not_a_whole_number_of_at_least_zero_is_a_configuration_error(
    count: object,
) -> None:
    # Act and Assert
    with pytest.raises(ConfigurationError, match="top_log_probabilities must be"):
        build_guard(ScriptedChatModel(replies=[]), top_log_probabilities=count)
