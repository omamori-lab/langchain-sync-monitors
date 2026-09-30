"""Feedback templates hold exactly one field, the monitor's reason, and are checked when built."""

from __future__ import annotations

import pytest

from langchain_sync_monitors.contracts import Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import DEFAULT_FEEDBACK_TEMPLATE, FeedbackTemplate


@pytest.fixture
def verdict() -> Verdict:
    return Verdict(suspicion=0.92, reason="It sends a credential off-site.")


def test_template_renders_the_monitor_s_reason(verdict: Verdict) -> None:
    # Arrange
    template = FeedbackTemplate(text="Blocked: {reason} Try something else.")

    # Act
    feedback = template.render(verdict=verdict)

    # Assert
    assert feedback == "Blocked: It sends a credential off-site. Try something else."


def test_escaped_braces_and_braces_in_the_reason_stay_literal() -> None:
    # Arrange
    template = FeedbackTemplate(text="{{monitor}} {reason}")
    braced_verdict = Verdict(suspicion=0.9, reason="wrote {secret} to a file")

    # Act
    feedback = template.render(verdict=braced_verdict)

    # Assert
    assert feedback == "{monitor} wrote {secret} to a file"


def test_default_feedback_is_marked_and_asks_for_a_safer_route(verdict: Verdict) -> None:
    # Act
    feedback = DEFAULT_FEEDBACK_TEMPLATE.render(verdict=verdict)

    # Assert
    assert feedback.startswith("[Safety monitor]")
    assert "It sends a credential off-site." in feedback
    assert "none of your tools ran it" in feedback
    assert "safer route" in feedback
    assert "Do not retry" in feedback
    assert "approval" in feedback


@pytest.mark.parametrize(
    "text",
    [
        "Blocked, try again.",
        "Blocked: {reason} while calling {tool}.",
        "Blocked: {}",
        "Blocked: {0}",
        "Blocked: {reason!r}",
        "Blocked: {reason:>40}",
        "Blocked: {reason.upper}",
    ],
)
def test_template_without_exactly_the_reason_field_is_rejected(text: str) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"must contain \{reason\}"):
        FeedbackTemplate(text=text)


@pytest.mark.parametrize("text", ["Blocked: {reason} }", "Blocked: {reason"])
def test_malformed_template_is_rejected(text: str) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="not a valid format string"):
        FeedbackTemplate(text=text)
