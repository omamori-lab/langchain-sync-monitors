"""Suspicion thresholds for control protocols, with defaults that ask to be replaced.

Every protocol compares a monitor's suspicion score, from 0 to 1, against a
threshold. A good threshold depends on the monitor and the agent, so the
library's defaults are placeholders: a protocol built with one emits an
`UncalibratedThresholdWarning` that says how to choose a real value.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Final

from langchain_sync_monitors.errors import ConfigurationError

LIBRARY_DIRECTORY: Final = os.path.dirname(__file__) + os.sep
"""This package's directory, so that warnings skip the library's own frames."""


@dataclass(frozen=True, slots=True)
class DefaultThreshold:
    """A threshold the user did not choose.

    A protocol given one uses `value` but emits an `UncalibratedThresholdWarning`
    when it is built. Pass a plain float to choose the threshold yourself.
    """

    value: float


class UncalibratedThresholdWarning(UserWarning):
    """A protocol runs on a default threshold instead of one calibrated for its monitor."""


DEFAULT_DEFER_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Defer to Resample treats a sample as suspicious."""

DEFAULT_BLOCK_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Auto Mode blocks a step."""

DEFAULT_FLAG_THRESHOLD: Final = DefaultThreshold(0.6)
"""The default score at or above which Trusted Monitoring flags a step."""

DEFAULT_AUDIT_THRESHOLD: Final = DefaultThreshold(0.9)
"""The default lowest score across a step's samples at or above which the step is flagged."""


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
    A value outside [0, 1], or NaN, raises `ConfigurationError`, because
    suspicion scores never leave that range.
    """
    value = threshold.value if isinstance(threshold, DefaultThreshold) else threshold
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        message = f"{parameter_name} must be between 0 and 1, got {value!r}"
        raise ConfigurationError(message)
    if isinstance(threshold, DefaultThreshold):
        warnings.warn(
            build_uncalibrated_message(parameter_name=parameter_name, value=value),
            UncalibratedThresholdWarning,
            stacklevel=2,
            skip_file_prefixes=(LIBRARY_DIRECTORY,),
        )
    return float(value)
