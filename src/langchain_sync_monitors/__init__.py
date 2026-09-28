"""Control monitors for LangChain agents and Deep Agents, as middleware.

A monitor judges every step an agent proposes before anything runs, and a
control protocol decides what happens with that judgement. `MonitorMiddleware`
joins the two and plugs into `create_agent` or `create_deep_agent`;
`monitor_subagents` gives every Deep Agents subagent a monitor of its own.
"""

from langchain_sync_monitors._langchain import MonitorStepEvent
from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Channel,
    ControlProtocol,
    Fallback,
    FeedbackVisibility,
    Monitor,
    MonitorInput,
    MonitorView,
    Outcome,
    PendingStep,
    Resampling,
    Sample,
    SampleRecord,
    StepDecision,
    StepRecord,
    SubagentHalt,
    TaskAuthor,
    Verdict,
)
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import ConfigurationError, MonitorError, SynchronousRunError
from langchain_sync_monitors.middleware import (
    MonitorMiddleware,
    MonitorPlacementWarning,
    MonitorState,
    check_monitor_placement,
)
from langchain_sync_monitors.monitors import (
    ATTACK_CRITERIA,
    IS_THE_STEP_SUSPICIOUS,
    CalibratedMonitor,
    CascadeMonitor,
    ChatModelMonitor,
    Combine,
    DecisionModel,
    DecisionModelMonitor,
    GuardModelMonitor,
    GuardScoring,
    HardLabelWarning,
    LLMMonitor,
    OpenRouterDecisionModel,
    RepeatedMonitor,
    TypeSafeDecisionModel,
    YesNoQuestion,
)
from langchain_sync_monitors.prompts import DEFAULT_MONITOR_PROMPT
from langchain_sync_monitors.protocols import (
    DEFAULT_FEEDBACK_TEMPLATE,
    DEFAULT_HALT_MESSAGE,
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    FeedbackTemplate,
    HaltRun,
    TrustedMonitoring,
)
from langchain_sync_monitors.thresholds import (
    DefaultThreshold,
    UncalibratedThresholdWarning,
    resolve_threshold,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ATTACK_CRITERIA",
    "DEFAULT_FEEDBACK_TEMPLATE",
    "DEFAULT_HALT_MESSAGE",
    "DEFAULT_MONITOR_PROMPT",
    "IS_THE_STEP_SUSPICIOUS",
    "AutoMode",
    "BlockedAttempt",
    "CalibratedMonitor",
    "CascadeMonitor",
    "Channel",
    "ChatModelMonitor",
    "Combine",
    "ConfigurationError",
    "ControlProtocol",
    "DecisionModel",
    "DecisionModelMonitor",
    "DefaultThreshold",
    "DeferToResample",
    "DeferToTrusted",
    "DeferToTrustedModel",
    "Fallback",
    "FeedbackTemplate",
    "FeedbackVisibility",
    "GuardModelMonitor",
    "GuardScoring",
    "HaltRun",
    "HardLabelWarning",
    "LLMMonitor",
    "Monitor",
    "MonitorError",
    "MonitorInput",
    "MonitorMiddleware",
    "MonitorPlacementWarning",
    "MonitorState",
    "MonitorStepEvent",
    "MonitorView",
    "OpenRouterDecisionModel",
    "Outcome",
    "PendingStep",
    "RepeatedMonitor",
    "Resampling",
    "Sample",
    "SampleRecord",
    "StepDecision",
    "StepRecord",
    "SubagentHalt",
    "SynchronousRunError",
    "TaskAuthor",
    "TrustedMonitoring",
    "TypeSafeDecisionModel",
    "UncalibratedThresholdWarning",
    "Verdict",
    "YesNoQuestion",
    "__version__",
    "check_monitor_placement",
    "monitor_subagents",
    "resolve_threshold",
]
