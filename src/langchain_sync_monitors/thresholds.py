"""Suspicion thresholds for control protocols, with defaults that ask to be replaced.

Every protocol compares a monitor's suspicion score, from 0 to 1, against a
threshold. A good threshold depends on the monitor and the agent, so the
library's defaults are placeholders: a protocol built with one emits an
`UncalibratedThresholdWarning` that says how to choose a real value.
"""

from __future__ import annotations

import numbers
import os
import warnings
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.options import (
    check_enum_option,
    check_instance_option,
    describe_option_value,
)

LIBRARY_DIRECTORY: Final = os.path.dirname(__file__) + os.sep
"""This package's directory, so that warnings skip the library's own frames."""


def read_threshold_value(value: object, *, parameter_name: str) -> float:
    """Return a threshold as a float from 0 to 1, raising `ConfigurationError` for anything else.

    Any real number counts, a `Fraction` or numpy's floats included, and so
    does a `Decimal`. A `bool` does not, although Python counts it as a number,
    and neither does a string such as `"0.6"`. The range is checked on the
    exact value, before it becomes a float, so a value just outside [0, 1]
    that a float would round into it is refused, as is NaN, because
    suspicion scores never leave that range. `-0.0` is read as `0.0`.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real | Decimal):
        message = (
            f"{parameter_name} must be a number between 0 and 1, got {describe_option_value(value)}"
        )
        raise ConfigurationError(message)
    try:
        is_in_range = not value < 0 and value <= 1
    except ArithmeticError:
        # A Decimal NaN raises `InvalidOperation` when compared, and is out of range too.
        is_in_range = False
    if not is_in_range:
        message = f"{parameter_name} must be between 0 and 1, got {describe_option_value(value)}"
        raise ConfigurationError(message)
    # Adding 0.0 turns -0.0 into 0.0.
    return float(value) + 0.0


@dataclass(frozen=True, slots=True)
class DefaultThreshold:
    """A threshold the user did not choose.

    A protocol given one uses `value` but emits an `UncalibratedThresholdWarning`
    when it is built. Pass a plain float to choose the threshold yourself.
    `value` is read as a threshold is, and kept as a float.
    """

    value: float

    def __post_init__(self) -> None:
        """Reject a value that is no threshold, and keep it as a float."""
        value = read_threshold_value(self.value, parameter_name="value")
        # The dataclass is frozen, so the float is set past its guard.
        object.__setattr__(self, "value", value)


class UncalibratedThresholdWarning(UserWarning):
    """A protocol runs on a default threshold instead of one calibrated for its monitor."""


class ThresholdStatistic(StrEnum):
    """The suspicion statistic a protocol compares a threshold with.

    A threshold is calibrated on the statistic it is compared with, so the
    `UncalibratedThresholdWarning` for a default threshold says how to
    calibrate it on this one. `resolve_threshold` takes it as `statistic`.
    """

    SINGLE_SAMPLE_SCORE = "single_sample_score"
    """One sample's suspicion score, which most thresholds are compared with."""

    LOWEST_SAMPLE_SCORE = "lowest_sample_score"
    """The lowest suspicion score across a step's samples.

    Defer to Resample compares `audit_threshold` with it when `max_resamples`
    is above 0.
    """


DEFAULT_DEFER_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Defer to Resample treats a sample as suspicious."""

DEFAULT_BLOCK_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Auto Mode blocks a step."""

DEFAULT_FLAG_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Trusted Monitoring flags a step."""

DEFAULT_AUDIT_THRESHOLD: Final = DefaultThreshold(0.9)
"""The default lowest score across a step's samples at or above which the step is flagged."""


def build_uncalibrated_message(
    *,
    parameter_name: str,
    value: float,
    statistic: ThresholdStatistic,
) -> str:
    """Explain that a default threshold is in use and how to calibrate it from honest runs.

    Control evaluations set thresholds at a percentile of the monitor's scores
    on honest runs [@greenblatt2023aicontrol; @bhatt2025ctrlz], so the message
    points the user there.
    """
    advice = build_calibration_advice(parameter_name=parameter_name, statistic=statistic)
    return (
        f"{parameter_name} uses the uncalibrated default {value}. Suspicion scores differ "
        "between monitors and agents, so a fixed default may block honest work or miss "
        f"attacks. {advice} Pass {parameter_name} explicitly to silence this warning."
    )


def build_calibration_advice(*, parameter_name: str, statistic: ThresholdStatistic) -> str:
    """Say how to calibrate a threshold on the statistic its protocol compares it with.

    The lowest score across a step's samples is never above its first
    sample's score, so a percentile of single-sample scores would set the
    threshold too high. Ctrl-Z sets its audit threshold at a percentile of the
    honest distribution of the statistic it audits on, the 99th percentile of
    the minimum over resamples [@bhatt2025ctrlz] (figure 4, right panel;
    appendix C), so the advice for that statistic takes each step's lowest
    score and the 99th percentile.
    """
    if statistic is ThresholdStatistic.LOWEST_SAMPLE_SCORE:
        return (
            f"{parameter_name} is compared with the lowest suspicion score across a step's "
            "samples, which is never above the first sample's score, so a percentile of "
            "single-sample scores sets it too high. Run the agent on honest tasks under the "
            "protocol and settings you will deploy, take each step's lowest score from "
            "monitor_log, and set the threshold at a high percentile of them (for example the "
            '99th), as the section "Calibrate the audit threshold" of the guide "Use Defer to '
            'Resample" (docs/how-to/use-defer-to-resample.md) shows.'
        )
    return (
        "Run the agent on honest tasks, collect the suspicion scores in monitor_log, and set "
        "the threshold at a high percentile of them (for example the 98th), or wrap the "
        "monitor in CalibratedMonitor so that its scores are honest-run percentiles."
    )


def resolve_threshold(
    *,
    parameter_name: str,
    threshold: float | DefaultThreshold,
    statistic: ThresholdStatistic = ThresholdStatistic.SINGLE_SAMPLE_SCORE,
) -> float:
    """Return the threshold as a float, warning when it is the library's default.

    `statistic` is what the protocol compares the threshold with, and the
    warning says how to calibrate the threshold on it; it must be a
    `ThresholdStatistic` member, and a plain string raises `ConfigurationError`.
    The warning is attributed to the first caller outside this library, which
    is the user's constructor call however many library frames lie between.
    `read_threshold_value` says which values count.
    """
    check_instance_option(parameter_name, option_type=str, parameter_name="parameter_name")
    check_enum_option(statistic, option_type=ThresholdStatistic, parameter_name="statistic")
    value = read_threshold_value(
        threshold.value if isinstance(threshold, DefaultThreshold) else threshold,
        parameter_name=parameter_name,
    )
    if isinstance(threshold, DefaultThreshold):
        warnings.warn(
            build_uncalibrated_message(
                parameter_name=parameter_name,
                value=value,
                statistic=statistic,
            ),
            UncalibratedThresholdWarning,
            stacklevel=2,
            skip_file_prefixes=(LIBRARY_DIRECTORY,),
        )
    return value
