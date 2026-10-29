"""The thresholds a protocol takes for a monitor's family when it leaves them at None.

`LLMMonitor` and `DecisionMonitor` have a default defer threshold and a
default audit threshold, and `BlockAndFeedBack`'s block threshold takes the
defer one. Taking a default warns `UncalibratedThresholdWarning`, once per
process and family.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Final

from langchain_sync_monitors.monitors import CONSTRUCTOR_CALLER_LEVEL
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
