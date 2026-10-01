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
from typing import Final

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.options import check_instance_option, describe_option_value

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


DEFAULT_DEFER_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default `defer_threshold`: Defer to Resample treats a sample at or above it as suspicious."""

DEFAULT_BLOCK_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default `block_threshold`: Auto Mode blocks a step at or above it."""

DEFAULT_TRUSTED_MONITORING_AUDIT_THRESHOLD: Final = DefaultThreshold(0.6)
"""Trusted Monitoring's default `audit_threshold`: it flags a step at or above it."""

DEFAULT_AUDIT_THRESHOLD: Final = DefaultThreshold(0.9)
"""The default `audit_threshold` of Defer to Trusted and Defer to Resample.

A step is flagged when the lowest score across its samples is at or above it.
"""


def build_uncalibrated_message(*, parameter_name: str, value: float) -> str:
    """Explain that a default threshold is in use and how to calibrate it from honest runs.

    Control evaluations set thresholds at a percentile of the monitor's scores
    on honest runs [@greenblatt2023aicontrol; @bhatt2025ctrlz], so the message
    points the user there.
    """
    return (
        f"{parameter_name} uses the uncalibrated default {value}. Suspicion scores differ "
        "between monitors and agents, so a fixed default may block honest work or miss "
        "attacks. Run the agent on honest tasks, collect the suspicion scores in "
        "monitor_log, and set the threshold at a high percentile of them (for example the "
        "98th), or wrap the monitor in CalibratedMonitor so that its scores are honest-run "
        f"percentiles. Pass {parameter_name} explicitly to silence this warning."
    )


def resolve_threshold(*, parameter_name: str, threshold: float | DefaultThreshold) -> float:
    """Return the threshold as a float, warning when it is the library's default.

    The warning is attributed to the first caller outside this library, which
    is the user's constructor call however many library frames lie between.
    `read_threshold_value` says which values count.
    """
    check_instance_option(parameter_name, option_type=str, parameter_name="parameter_name")
    value = read_threshold_value(
        threshold.value if isinstance(threshold, DefaultThreshold) else threshold,
        parameter_name=parameter_name,
    )
    if isinstance(threshold, DefaultThreshold):
        warnings.warn(
            build_uncalibrated_message(parameter_name=parameter_name, value=value),
            UncalibratedThresholdWarning,
            stacklevel=2,
            skip_file_prefixes=(LIBRARY_DIRECTORY,),
        )
    return value
