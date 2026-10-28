"""Feedback templates hold exactly one field, the reason, and the halt message is the monitor's."""

from __future__ import annotations

import pytest

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.feedback import build_halt_message
from langchain_sync_monitors.protocols import (
    DEFAULT_FEEDBACK_TEMPLATE,
    DEFAULT_HALT_MESSAGE,
    FeedbackTemplate,
)

REASON = "It sends a credential off-site."


def test_template_renders_the_reason() -> None:
    # Arrange
    template = FeedbackTemplate(text="Blocked: {reason} Try something else.")

    # Act
    feedback = template.render(reason=REASON)

    # Assert
    assert feedback == "Blocked: It sends a credential off-site. Try something else."


def test_escaped_braces_and_braces_in_the_reason_stay_literal() -> None:
    # Arrange
    template = FeedbackTemplate(text="{{monitor}} {reason}")

    # Act
    feedback = template.render(reason="wrote {secret} to a file")

    # Assert
    assert feedback == "{monitor} wrote {secret} to a file"


def test_default_feedback_is_marked_and_asks_for_a_safer_route() -> None:
    # Act
    feedback = DEFAULT_FEEDBACK_TEMPLATE.render(reason=REASON)

    # Assert
    assert feedback.startswith("[Safety monitor]")
    assert REASON in feedback
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


def test_every_halt_message_gets_a_fresh_monitor_id() -> None:
    # Act
    first = build_halt_message(DEFAULT_HALT_MESSAGE)
    second = build_halt_message(DEFAULT_HALT_MESSAGE)

    # Assert
    assert (first.id or "").startswith("monitor-")
    assert (second.id or "").startswith("monitor-")
    assert first.id != second.id
    assert first.tool_calls == second.tool_calls == []


def test_default_halt_message_is_marked_as_the_monitor_s() -> None:
    # Assert
    assert DEFAULT_HALT_MESSAGE.startswith("[Safety monitor]")
