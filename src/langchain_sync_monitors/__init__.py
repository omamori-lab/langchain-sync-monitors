"""Control monitors for LangChain agents and Deep Agents, as middleware.

A monitor judges every step an agent proposes before any of the agent's own
tools run, and a control protocol decides what happens with that judgement.
`MonitorMiddleware` joins the two and plugs into `create_agent` or
`create_deep_agent`; `monitor_subagents` gives every Deep Agents subagent a
monitor of its own. Server-side tools, which the model provider runs inside
the model call, run before any judgement; the middleware warns about the
server tools of Anthropic, OpenAI and Gemini it knows with a
`ServerToolWarning`.

The monitors and the control protocols are hyoshigi's [@hyoshigi2026]: each
monitor and each protocol here is one configuration that builds hyoshigi's
sync and async twins, and the value types their options take are hyoshigi's
own, re-exported here.
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
    Redaction,
    Resampling,
    View,
)

from langchain_sync_monitors._langchain import MonitorStepEvent, MonitorStepFailedEvent
from langchain_sync_monitors.audits import (
    ApproveAnswer,
    AuditAnswer,
    AuditRequest,
    EditAnswer,
    EditedToolCall,
    RejectAnswer,
    StopAnswer,
)
from langchain_sync_monitors.checkpointers import MissingCheckpointerWarning
from langchain_sync_monitors.contracts import (
    DelegatedRun,
    Delegation,
    Outcome,
    SampleRecord,
    StepRecord,
    SubagentHalt,
    TaskAuthor,
)
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import (
    ConfigurationError,
    MessageFormatError,
    MissingExtraError,
    MonitorError,
    ServerToolWarning,
)
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
    BlockAndFeedBack,
    DeferToResample,
    FeedbackTemplate,
)
from langchain_sync_monitors.score_export import Tracer
from langchain_sync_monitors.threshold_defaults import UncalibratedThresholdWarning

__version__ = "0.3.0"

__all__ = [
    "ATTACK_CRITERIA",
    "DEFAULT_FEEDBACK_TEMPLATE",
    "DEFAULT_HALT_MESSAGE",
    "DEFAULT_PROMPT",
    "DEFAULT_STATEMENT",
    "GPT_OSS_SAFEGUARD",
    "QWEN3GUARD",
    "Aggregation",
    "ApproveAnswer",
    "AuditAnswer",
    "AuditRequest",
    "AuditResponse",
    "BlockAndFeedBack",
    "CachedResampleWarning",
    "CalibratedMonitor",
    "CascadeMonitor",
    "Channel",
    "ConfigurationError",
    "DecisionMonitor",
    "DeferToResample",
    "DelegatedRun",
    "Delegation",
    "EditAnswer",
    "EditedToolCall",
    "FailureResponse",
    "FeedbackTemplate",
    "GuardLabels",
    "GuardMonitor",
    "HardLabelWarning",
    "LLMMonitor",
    "MediaTypes",
    "MessageFormatError",
    "MissingCheckpointerWarning",
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
    "Redaction",
    "RejectAnswer",
    "RepeatedMonitor",
    "Resampling",
    "SampleRecord",
    "ServerToolWarning",
    "StepRecord",
    "StopAnswer",
    "SubagentHalt",
    "TaskAuthor",
    "Tracer",
    "UncalibratedThresholdWarning",
    "View",
    "__version__",
    "check_monitor_placement",
    "monitor_subagents",
]
