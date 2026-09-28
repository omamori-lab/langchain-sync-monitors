"""Control protocols, which decide what the agent state receives for each step.

Every protocol reaches the model only through a `PendingStep` and awaits
nothing else, so one copy of each serves both `invoke()` and `ainvoke()`.
Protocols and fallbacks hold configuration only, never state from a run, so
one instance can serve parallel subagent runs.
"""

from langchain_sync_monitors.protocols.auto_mode import (
    DEFAULT_FEEDBACK_TEMPLATE,
    AutoMode,
    FeedbackTemplate,
)
from langchain_sync_monitors.protocols.defer_to_resample import DeferToResample, DeferToTrusted
from langchain_sync_monitors.protocols.fallbacks import (
    DEFAULT_HALT_MESSAGE,
    DeferToTrustedModel,
    HaltRun,
)
from langchain_sync_monitors.protocols.trusted_monitoring import TrustedMonitoring

__all__ = [
    "DEFAULT_FEEDBACK_TEMPLATE",
    "DEFAULT_HALT_MESSAGE",
    "AutoMode",
    "DeferToResample",
    "DeferToTrusted",
    "DeferToTrustedModel",
    "FeedbackTemplate",
    "HaltRun",
    "TrustedMonitoring",
]
