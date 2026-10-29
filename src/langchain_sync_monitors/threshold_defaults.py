"""The thresholds a protocol takes for a monitor's family when it leaves them at None.

`LLMMonitor` and `DecisionMonitor` have a default defer threshold and a
default audit threshold, and `BlockAndFeedBack`'s block threshold takes the
defer one. Taking a default warns `UncalibratedThresholdWarning`, once per
process and family.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors import (
    CONSTRUCTOR_CALLER_LEVEL,
    CalibratedMonitor,
    CascadeMonitor,
    DecisionMonitor,
    GuardMonitor,
    LLMMonitor,
    MonitorConfiguration,
    MonitorTwins,
    RepeatedMonitor,
)
from langchain_sync_monitors.options import LIBRARY_DIRECTORY

DEFAULTS_URL: Final = (
    "https://omamori-lab.github.io/langchain-sync-monitors/how-to/choose-a-protocol/"
)
"""The docs page that tabulates each family's defaults and how they were measured."""


class UncalibratedThresholdWarning(UserWarning):
    """A protocol's threshold is left at None and takes its monitor family's default.

    The default was measured on honest runs of other agents, not the user's.
    The middleware emits this warning once per process and family, when it is
    built.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdDefaults:
    """A monitor family's default thresholds, and the model they were measured with."""

    family: str
    measured_on: str
    defer: float
    audit: float


# Measured by the evals that the table at DEFAULTS_URL reports.
LLM_MONITOR_DEFAULTS: Final = ThresholdDefaults(
    family="LLMMonitor",
    measured_on="xiaomi/mimo-v2.6-pro",
    defer=0.2,
    audit=0.3,
)
DECISION_MONITOR_DEFAULTS: Final = ThresholdDefaults(
    family="DecisionMonitor",
    measured_on="typesafe/jev-1.13",
    defer=0.05,
    audit=0.12,
)


@functools.cache
def warn_about_uncalibrated_thresholds(defaults: ThresholdDefaults) -> None:
    """Emit the `UncalibratedThresholdWarning` for a family, on its first call in the process.

    The cache on this function keeps the warning to one per process and family.
    It points at the line that built the middleware.
    """
    warnings.warn(
        f"A threshold left at None takes the {defaults.family} default: defer_threshold "
        f"{defaults.defer} and audit_threshold {defaults.audit}, and block_threshold the "
        f"defer one. They were measured with {defaults.measured_on} on honest runs of "
        "other agents, not yours; set the thresholds from the scores of your own agent's "
        f"honest runs. The measurements: {DEFAULTS_URL}",
        UncalibratedThresholdWarning,
        stacklevel=CONSTRUCTOR_CALLER_LEVEL,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )


def find_threshold_defaults(monitor: MonitorConfiguration) -> ThresholdDefaults | None:
    """Return the defaults of the monitor's family, or None for a family without any."""
    match monitor:
        case LLMMonitor():
            return LLM_MONITOR_DEFAULTS
        case DecisionMonitor():
            return DECISION_MONITOR_DEFAULTS
        case (
            GuardMonitor()
            | RepeatedMonitor()
            | CalibratedMonitor()
            | CascadeMonitor()
            | MonitorTwins()
        ):
            return None
        case unreachable:
            assert_never(unreachable)


def take_threshold_defaults(
    monitor: MonitorConfiguration,
    *,
    options: Sequence[str],
) -> ThresholdDefaults:
    """Return the defaults of the monitor's family for the `options` left at None, and warn.

    A monitor of a family without defaults raises `ConfigurationError` naming `options`.
    """
    defaults = find_threshold_defaults(monitor)
    if defaults is None:
        names = " and ".join(options)
        message = (
            f"{names} cannot be None with a {type(monitor).__name__}, which has no default "
            "threshold; give a number from 0 to 1, set from the scores of honest runs. Only an "
            f"LLMMonitor and a DecisionMonitor have defaults: {DEFAULTS_URL}"
        )
        raise ConfigurationError(message)
    warn_about_uncalibrated_thresholds(defaults)
    return defaults
