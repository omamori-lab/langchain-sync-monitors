"""Protocols, fallbacks and the middleware check the type of every option when they are built.

A limit given as a float, or a protocol given where a fallback belongs, used to
build without error and fail only at the first suspicious step: with a
`TypeError` or an `AttributeError`, during the attack the protocol exists for.
"""

from __future__ import annotations

import math
import numbers
from decimal import Decimal
from fractions import Fraction
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
    TrustedMonitoring,
)
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst

WRONG_COUNTS = [2.5, 2.0, "3", True, None]
WRONG_COUNT_IDS = ["float", "whole-float", "string", "bool", "none"]


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of `int`."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


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
def test_auto_mode_refuses_a_block_limit_that_is_not_a_whole_number(
    parameter_name: str,
    value: object,
) -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=f"{parameter_name} must be a whole number of at least 1",
    ):
        build_auto_mode(**{parameter_name: value})


@pytest.mark.parametrize("value", WRONG_COUNTS, ids=WRONG_COUNT_IDS)
def test_defer_to_resample_refuses_a_resample_count_that_is_not_a_whole_number(
    value: object,
) -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match="max_resamples must be a whole number of at least 0",
    ):
        build_defer_to_resample(max_resamples=value)


def test_the_smallest_counts_are_accepted() -> None:
    # Act
    auto_mode = build_auto_mode(max_consecutive_blocks=1, max_total_blocks=1)
    defer_to_resample = build_defer_to_resample(max_resamples=0)

    # Assert
    assert (auto_mode.max_consecutive_blocks, auto_mode.max_total_blocks) == (1, 1)
    assert defer_to_resample.max_resamples == 0


def test_integers_that_are_not_python_ints_are_accepted_as_ints() -> None:
    # Act
    auto_mode = build_auto_mode(
        max_consecutive_blocks=ArrayInteger(2),
        max_total_blocks=ArrayInteger(5),
    )
    defer_to_resample = build_defer_to_resample(max_resamples=ArrayInteger(3))

    # Assert
    assert (auto_mode.max_consecutive_blocks, auto_mode.max_total_blocks) == (2, 5)
    assert defer_to_resample.max_resamples == 3
    assert type(auto_mode.max_consecutive_blocks) is int
    assert type(defer_to_resample.max_resamples) is int


def test_infinity_lifts_the_total_block_limit() -> None:
    # Act
    auto_mode = build_auto_mode(max_total_blocks=math.inf)

    # Assert
    assert auto_mode.max_total_blocks == math.inf


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_consecutive_blocks": math.inf}, "max_consecutive_blocks must be a whole number"),
        ({"max_total_blocks": -math.inf}, "max_total_blocks must be a whole number"),
        ({"max_total_blocks": math.nan}, "or math.inf for no limit, got nan"),
        ({"max_total_blocks": 0}, "max_total_blocks must be a whole number of at least 1"),
    ],
    ids=["consecutive-infinity", "negative-infinity", "nan", "zero"],
)
def test_only_the_total_block_limit_accepts_infinity(
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build_auto_mode(**options)


def test_infinity_is_refused_as_a_resample_count() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError, match="max_resamples must be a whole number of at least 0, got inf"
    ):
        build_defer_to_resample(max_resamples=math.inf)


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
        match=r"fallback must be a Fallback, got an instance of DeferToTrusted\. DeferToTrusted "
        r"is a control "
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
        match=r"protocol must be a ControlProtocol, got an instance of DeferToTrustedModel\. "
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


def test_infinity_that_is_not_a_float_is_refused_as_the_total_limit() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="max_total_blocks must be a whole number"):
        build_auto_mode(max_total_blocks=Decimal("Infinity"))


def test_a_foreign_type_that_shares_a_builtin_name_is_named_with_its_module() -> None:
    # Arrange: numpy's bool is not Python's, and is named bool too
    foreign_bool = type("bool", (), {"__module__": "numpy"})()

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"got an instance of numpy\.bool$"):
        build_auto_mode(max_consecutive_blocks=foreign_bool)


def build_protocol_with_threshold(parameter_name: str, value: object) -> object:
    thresholds: dict[str, Any] = {parameter_name: value}
    if parameter_name == "flag_threshold":
        return TrustedMonitoring(**thresholds)
    if parameter_name == "block_threshold":
        return AutoMode(**thresholds)
    settings: dict[str, Any] = {"defer_threshold": 0.6, "audit_threshold": 0.9, **thresholds}
    return DeferToResample(fallback=HaltRun(), **settings)


THRESHOLD_PARAMETERS = ["flag_threshold", "block_threshold", "defer_threshold", "audit_threshold"]


@pytest.mark.parametrize("parameter_name", THRESHOLD_PARAMETERS)
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0.6", "must be a number between 0 and 1, got '0.6'"),
        (True, "must be a number between 0 and 1, got True"),
        ([0.6], r"must be a number between 0 and 1, got an instance of list"),
        (math.nan, "must be between 0 and 1, got nan"),
        (1.5, "must be between 0 and 1, got 1.5"),
        (-0.1, "must be between 0 and 1, got -0.1"),
        (Decimal("sNaN"), r"must be between 0 and 1, got Decimal\('sNaN'\)"),
    ],
    ids=["string", "bool", "list", "nan", "above-one", "below-zero", "signalling-nan"],
)
def test_a_threshold_that_is_not_a_number_from_zero_to_one_is_refused(
    parameter_name: str,
    value: object,
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=f"{parameter_name} {message}"):
        build_protocol_with_threshold(parameter_name, value)


@pytest.mark.parametrize("parameter_name", THRESHOLD_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [0, 1, 0.6, Fraction(3, 5), Decimal("0.6")],
    ids=["zero", "one", "float", "fraction", "decimal"],
)
def test_a_threshold_given_as_any_real_number_from_zero_to_one_is_kept_as_a_float(
    parameter_name: str,
    value: float | Fraction | Decimal,
) -> None:
    # Act
    protocol = build_protocol_with_threshold(parameter_name, value)

    # Assert
    threshold = getattr(protocol, parameter_name)
    assert type(threshold) is float
    assert threshold == float(value)
