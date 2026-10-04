"""Control monitors for LangChain agents and Deep Agents, as middleware.

A monitor judges every step an agent proposes before any of the agent's own
tools run, and a control protocol decides what happens with that judgement.
`MonitorMiddleware` joins the two and plugs into `create_agent` or
`create_deep_agent`; `monitor_subagents` gives every Deep Agents subagent a
monitor of its own. Server-side tools, which the model provider runs inside
the model call, run before any judgement; the middleware warns about the
server tools of Anthropic, OpenAI and Gemini it knows with a
`ServerToolWarning`.
"""

from langchain_sync_monitors._langchain import MonitorStepEvent, MonitorStepFailedEvent
from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Channel,
    ControlProtocol,
    Delegation,
    Fallback,
    FeedbackVisibility,
    Monitor,
    MonitorInput,
    MonitorView,
    Outcome,
    PendingStep,
    Resampling,
    Sample,
    SampleDrawing,
    SampleRecord,
    StepDecision,
    StepRecord,
    SubagentHalt,
    TaskAuthor,
    Verdict,
)
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import (
    ConfigurationError,
    InvalidSuspicionError,
    MissingExtraError,
    MonitorError,
    ServerToolWarning,
    SynchronousRunError,
)
from langchain_sync_monitors.feedback import WITHHELD_TEXT_MESSAGE
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.model_calls import CachedResampleWarning
from langchain_sync_monitors.monitor_state import MonitorState
from langchain_sync_monitors.monitors import (
    ATTACK_CRITERIA,
    IS_THE_STEP_SUSPICIOUS,
    Aggregation,
    CalibratedMonitor,
    CascadeMonitor,
    ChatModelMonitor,
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
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
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
from langchain_sync_monitors.scores import Tracer
from langchain_sync_monitors.thresholds import (
    DefaultThreshold,
    ThresholdStatistic,
    UncalibratedThresholdWarning,
    resolve_threshold,
)

__version__ = "0.1.5"

__all__ = [
    "ATTACK_CRITERIA",
    "DEFAULT_FEEDBACK_TEMPLATE",
    "DEFAULT_HALT_MESSAGE",
    "DEFAULT_MONITOR_PROMPT",
    "IS_THE_STEP_SUSPICIOUS",
    "WITHHELD_TEXT_MESSAGE",
    "Aggregation",
    "AutoMode",
    "BlockedAttempt",
    "CachedResampleWarning",
    "CalibratedMonitor",
    "CascadeMonitor",
    "Channel",
    "ChatModelMonitor",
    "ConfigurationError",
    "ControlProtocol",
    "DecisionModel",
    "DecisionModelMonitor",
    "DefaultThreshold",
    "DeferToResample",
    "DeferToTrusted",
    "DeferToTrustedModel",
    "Delegation",
    "Fallback",
    "FeedbackTemplate",
    "FeedbackVisibility",
    "GuardModelMonitor",
    "GuardScoring",
    "HaltRun",
    "HardLabelWarning",
    "InvalidSuspicionError",
    "LLMMonitor",
    "MissingExtraError",
    "Monitor",
    "MonitorError",
    "MonitorInput",
    "MonitorMiddleware",
    "MonitorPlacementWarning",
    "MonitorState",
    "MonitorStepEvent",
    "MonitorStepFailedEvent",
    "MonitorView",
    "OpenRouterDecisionModel",
    "Outcome",
    "PendingStep",
    "RepeatedMonitor",
    "Resampling",
    "Sample",
    "SampleDrawing",
    "SampleRecord",
    "ServerToolWarning",
    "StepDecision",
    "StepRecord",
    "SubagentHalt",
    "SynchronousRunError",
    "TaskAuthor",
    "ThresholdStatistic",
    "Tracer",
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
