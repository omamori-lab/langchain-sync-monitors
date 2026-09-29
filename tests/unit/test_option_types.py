"""Protocols, fallbacks and the middleware check the type of every option when they are built.

A limit given as a float, or a protocol given where a fallback belongs, used to
build without error and fail only at the first suspicious step: with a
`TypeError` or an `AttributeError`, during the attack the protocol exists for.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    FeedbackTemplate,
    HaltRun,
)
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst

WRONG_COUNTS = [2.5, 2.0, "3", True, None]
WRONG_COUNT_IDS = ["float", "whole-float", "string", "bool", "none"]


def build_trusted_model() -> FakeListChatModel:
    return FakeListChatModel(responses=["unused"])


def build_defer_to_trusted() -> DeferToTrusted:
    return DeferToTrusted(
        trusted_model=build_trusted_model(),
        defer_threshold=0.6,
        audit_threshold=0.9,
    )


def build_auto_mode(**options: Any) -> AutoMode:
    return AutoMode(block_threshold=0.6, **options)


def build_defer_to_resample(**options: Any) -> DeferToResample:
    settings: dict[str, Any] = {"fallback": HaltRun(), **options}
    return DeferToResample(defer_threshold=0.6, audit_threshold=0.9, **settings)


@pytest.mark.parametrize("value", WRONG_COUNTS, ids=WRONG_COUNT_IDS)
@pytest.mark.parametrize("parameter_name", ["max_consecutive_blocks", "max_total_blocks"])
def test_auto_mode_refuses_a_block_limit_that_is_not_an_int(
    parameter_name: str,
    value: object,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=f"{parameter_name} must be an int of at least 1"):
        build_auto_mode(**{parameter_name: value})


@pytest.mark.parametrize("value", WRONG_COUNTS, ids=WRONG_COUNT_IDS)
def test_defer_to_resample_refuses_a_resample_count_that_is_not_an_int(value: object) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="max_resamples must be an int of at least 0"):
        build_defer_to_resample(max_resamples=value)


def test_the_smallest_counts_are_accepted() -> None:
    # Act
    auto_mode = build_auto_mode(max_consecutive_blocks=1, max_total_blocks=1)
    defer_to_resample = build_defer_to_resample(max_resamples=0)

    # Assert
    assert (auto_mode.max_consecutive_blocks, auto_mode.max_total_blocks) == (1, 1)
    assert defer_to_resample.max_resamples == 0


@pytest.mark.parametrize(
    "value", ["halt", None, build_trusted_model()], ids=["string", "none", "model"]
)
def test_auto_mode_refuses_a_limit_fallback_that_is_not_a_fallback(value: object) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="when_limit_reached must be a Fallback"):
        build_auto_mode(when_limit_reached=value)


def test_a_protocol_given_as_a_fallback_points_to_the_fallback_it_resembles() -> None:
    # Arrange
    protocol = build_defer_to_trusted()

    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"fallback must be a Fallback, got a DeferToTrusted\. DeferToTrusted is a control "
        r"protocol, .* use DeferToTrustedModel\(trusted_model=\.\.\.\)",
    ):
        build_defer_to_resample(fallback=protocol)


def test_a_protocol_given_as_the_limit_fallback_points_to_the_fallback_it_resembles() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"use DeferToTrustedModel\(trusted_model="):
        build_auto_mode(when_limit_reached=build_defer_to_trusted())


def test_a_fallback_given_as_the_protocol_points_to_the_protocol_it_resembles() -> None:
    # Arrange
    fallback = DeferToTrustedModel(trusted_model=build_trusted_model())

    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"protocol must be a ControlProtocol, got a DeferToTrustedModel\. "
        r"DeferToTrustedModel is a fallback, .* use DeferToTrusted\(trusted_model=\.\.\.\)",
    ):
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=fallback)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"monitor": "openrouter:some/judge"},
            "monitor must be a Monitor, got 'openrouter:some/judge'",
        ),
        ({"protocol": "auto_mode"}, "protocol must be a ControlProtocol, got 'auto_mode'"),
    ],
    ids=["monitor", "protocol"],
)
def test_the_middleware_refuses_a_monitor_or_protocol_of_the_wrong_type(
    options: dict[str, object],
    message: str,
) -> None:
    # Arrange
    arguments: dict[str, Any] = {"monitor": KeywordMonitor(), "protocol": AcceptFirst(), **options}

    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        MonitorMiddleware(**arguments)


def test_auto_mode_refuses_feedback_given_as_plain_text() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"feedback must be a FeedbackTemplate, got 'Blocked: \{reason\}'\. Wrap the text",
    ):
        build_auto_mode(feedback="Blocked: {reason}")


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: FeedbackTemplate(text=42), "text must be a str, got 42"),  # ty: ignore[invalid-argument-type]
        (lambda: HaltRun(message=None), "message must be a str, got None"),  # ty: ignore[invalid-argument-type]
    ],
    ids=["feedback-text", "halt-message"],
)
def test_a_message_that_is_not_text_is_refused(build: Any, message: str) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build()
