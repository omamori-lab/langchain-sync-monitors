"""Control monitors for LangChain agents and Deep Agents, as middleware.

A monitor judges every step an agent proposes before any of the agent's own
tools run, and a control protocol decides what happens with that judgement.
`MonitorMiddleware` joins the two and plugs into `create_agent` or
`create_deep_agent`; `monitor_subagents` gives every Deep Agents subagent a
monitor of its own. Server-side tools, which the model provider runs inside
the model call, run before any judgement; the middleware warns about the
server tools of Anthropic, OpenAI and Gemini it knows with a
`ServerToolWarning`.

The monitors are hyoshigi's [@hyoshigi2026]: each monitor here is one
configuration that builds hyoshigi's sync and async twins, and the value
types its options take are hyoshigi's own, re-exported here.
"""

from hyoshigi import (
    ATTACK_CRITERIA,
    DEFAULT_PROMPT,
    DEFAULT_STATEMENT,
    GPT_OSS_SAFEGUARD,
    QWEN3GUARD,
    Aggregation,
    AuditResponse,
    Channel,
    FailureResponse,
    GuardLabels,
    HardLabelWarning,
    MediaTypes,
    MonitorPrompt,
    View,
)

from langchain_sync_monitors._langchain import MonitorStepEvent, MonitorStepFailedEvent
from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    ControlProtocol,
    DelegatedRun,
    Delegation,
    Fallback,
    FeedbackVisibility,
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
    MessageFormatError,
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
    CalibratedMonitor,
    CascadeMonitor,
    DecisionMonitor,
    GuardMonitor,
    LLMMonitor,
    MonitorTwins,
    RepeatedMonitor,
)
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
from langchain_sync_monitors.protocols import (
    DEFAULT_FEEDBACK_TEMPLATE,
    DEFAULT_HALT_MESSAGE,
    AutoMode,
    BlockAndFeedBack,
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

__version__ = "0.1.6.dev0"

__all__ = [
    "ATTACK_CRITERIA",
    "DEFAULT_FEEDBACK_TEMPLATE",
    "DEFAULT_HALT_MESSAGE",
    "DEFAULT_PROMPT",
    "DEFAULT_STATEMENT",
    "GPT_OSS_SAFEGUARD",
    "QWEN3GUARD",
    "WITHHELD_TEXT_MESSAGE",
    "Aggregation",
    "AuditResponse",
    "AutoMode",
    "BlockAndFeedBack",
    "BlockedAttempt",
    "CachedResampleWarning",
    "CalibratedMonitor",
    "CascadeMonitor",
    "Channel",
    "ConfigurationError",
    "ControlProtocol",
    "DecisionMonitor",
    "DefaultThreshold",
    "DeferToResample",
    "DeferToTrusted",
    "DeferToTrustedModel",
    "DelegatedRun",
    "Delegation",
    "FailureResponse",
    "Fallback",
    "FeedbackTemplate",
    "FeedbackVisibility",
    "GuardLabels",
    "GuardMonitor",
    "HaltRun",
    "HardLabelWarning",
    "InvalidSuspicionError",
    "LLMMonitor",
    "MediaTypes",
    "MessageFormatError",
    "MissingExtraError",
    "MonitorError",
    "MonitorMiddleware",
    "MonitorPlacementWarning",
    "MonitorPrompt",
    "MonitorState",
    "MonitorStepEvent",
    "MonitorStepFailedEvent",
    "MonitorTwins",
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
    "UncalibratedThresholdWarning",
    "Verdict",
    "View",
    "__version__",
    "check_monitor_placement",
    "monitor_subagents",
    "resolve_threshold",
]
