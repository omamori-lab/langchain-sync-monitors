"""The thresholds a protocol takes for a monitor's family when it leaves them at None.

`LLMMonitor` and `DecisionMonitor` have a default defer threshold and a
default audit threshold, and `BlockAndFeedBack`'s block threshold takes the
defer one. Taking a default warns `UncalibratedThresholdWarning`, once per
process and family.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, assert_never

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


# Measured by the evals tabulated at https://omamori-lab.github.io/langchain-sync-monitors/how-to/choose-a-protocol/
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


WARNED_FAMILIES: Final[set[str]] = set()
"""The families the process has warned about, each by name."""
WARNED_FAMILIES_LOCK: Final = threading.Lock()
"""Held while a family is checked against `WARNED_FAMILIES` and added to it."""


def warn_about_uncalibrated_thresholds(defaults: ThresholdDefaults) -> None:
    """Emit the `UncalibratedThresholdWarning` for a family, on its first call in the process.

    Later calls for the family return without a warning, from any thread. The
    warning points at the line that built the middleware.
    """
    with WARNED_FAMILIES_LOCK:
        if defaults.family in WARNED_FAMILIES:
            return
        WARNED_FAMILIES.add(defaults.family)
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


def read_threshold_defaults(
    monitor: MonitorConfiguration,
    *,
    options: Sequence[str],
) -> ThresholdDefaults:
    """Return the defaults of the monitor's family for the `options` left at None.

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
    return defaults


def raise_inverted_default(
    defaults: ThresholdDefaults,
    *,
    defer_threshold: float | None,
    audit_threshold: float | None,
) -> NoReturn:
    """Raise `ConfigurationError` for a default that is on the wrong side of the given threshold.

    Exactly one of the thresholds is None: it took the family's default, and the
    defer threshold must be at most the audit threshold.
    """
    if audit_threshold is None:
        message = (
            f"audit_threshold took the {defaults.family} default {defaults.audit}, which is "
            f"below the given defer_threshold {defer_threshold}; pass audit_threshold too"
        )
    else:
        message = (
            f"defer_threshold took the {defaults.family} default {defaults.defer}, which is "
            f"above the given audit_threshold {audit_threshold}; pass defer_threshold too"
        )
    raise ConfigurationError(message)


def warn_if_defaulted(defaults: ThresholdDefaults | None) -> None:
    """Warn that the family's defaults are in use, unless no threshold took one."""
    if defaults is not None:
        warn_about_uncalibrated_thresholds(defaults)
