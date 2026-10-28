"""Monitors: the models that judge each proposed step, and wrappers around them.

Three families score a step: LLM monitors (`LLMMonitor`), guard
models (`GuardModelMonitor`) and decision models (`DecisionModelMonitor`).
The wrappers in `composition` are monitors too, so they stack around any of
them.
"""

from dataclasses import dataclass, field

import hyoshigi
from hyoshigi import DEFAULT_STATEMENT, GuardLabels, MediaTypes, MonitorPrompt, View
from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.monitors.chat import (
    ChatModelMonitor,
    LLMMonitor,
    warn_about_cached_monitor_replies,
)
from langchain_sync_monitors.monitors.composition import (
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)
from langchain_sync_monitors.monitors.decision import (
    Aggregation,
    DecisionModel,
    DecisionModelMonitor,
    TypeSafeDecisionModel,
)
from langchain_sync_monitors.monitors.decision_questions import (
    ATTACK_CRITERIA,
    IS_THE_STEP_SUSPICIOUS,
    YesNoQuestion,
)
from langchain_sync_monitors.monitors.guard import GuardModelMonitor, GuardScoring, HardLabelWarning
from langchain_sync_monitors.monitors.openrouter_decisions import OpenRouterDecisionModel

PLACEHOLDER_VERDICT = Verdict(suspicion=0.0, reason="")
"""The verdict every stub below gives, until the monitors are built on hyoshigi."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorTwins(Monitor):
    """Stub: a user's own monitor as hyoshigi's sync and async twins; it scores every step 0."""

    monitor: hyoshigi.Monitor
    async_monitor: hyoshigi.AsyncMonitor

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardMonitor(Monitor):
    """Stub: a guard model's monitor, built on hyoshigi's twins; it scores every step 0."""

    model: str | BaseChatModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    view: View = field(default_factory=View)
    monitor: hyoshigi.GuardMonitor | None = field(init=False, default=None)
    async_monitor: hyoshigi.AsyncGuardMonitor | None = field(init=False, default=None)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionMonitor(Monitor):
    """Stub: a decision model's monitor, built on hyoshigi's twins; it scores every step 0."""

    model: str
    view: View = field(default_factory=View)
    statements: tuple[str, ...] = (DEFAULT_STATEMENT,)
    aggregation: hyoshigi.Aggregation = hyoshigi.Aggregation.MAX
    perceives: MediaTypes = field(default_factory=MediaTypes)
    monitor: hyoshigi.DecisionMonitor | None = field(init=False, default=None)
    async_monitor: hyoshigi.AsyncDecisionMonitor | None = field(init=False, default=None)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Give the placeholder verdict."""
        return PLACEHOLDER_VERDICT


__all__ = [
    "ATTACK_CRITERIA",
    "IS_THE_STEP_SUSPICIOUS",
    "Aggregation",
    "CalibratedMonitor",
    "CascadeMonitor",
    "ChatModelMonitor",
    "DecisionModel",
    "DecisionModelMonitor",
    "DecisionMonitor",
    "GuardModelMonitor",
    "GuardMonitor",
    "GuardScoring",
    "HardLabelWarning",
    "LLMMonitor",
    "MonitorTwins",
    "OpenRouterDecisionModel",
    "RepeatedMonitor",
    "TypeSafeDecisionModel",
    "YesNoQuestion",
    "warn_about_cached_monitor_replies",
]
