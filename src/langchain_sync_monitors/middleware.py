"""The middleware that puts a monitor and a control protocol around every model call.

`MonitorMiddleware` wraps each model call of an agent. The untrusted model's
proposal is judged before any tool runs, the protocol decides what the agent
state receives, and one `StepRecord` per step is appended to `monitor_log`.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, and a
middleware inside the monitor that returns state updates would have them
merged across every sample the protocol draws [@langchain2026].
`check_monitor_placement` warns about such a list.
"""

import logging
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Annotated, NotRequired, Self, override

from langchain.agents.middleware.internal_call_transformer import InternalCallTransformer
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelResponse,
    OmitFromInput,
)
from langchain_core.messages import AIMessage
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentContext,
    AgentModelRequest,
    AnyAgentMiddleware,
    AsyncModelCallHandler,
    ModelCallHandler,
    MonitorStepEvent,
    StructuredOutput,
    hide_model_calls_from_message_stream,
    read_monitor_log,
    write_stream_event,
)
from langchain_sync_monitors.contracts import (
    ControlProtocol,
    FeedbackVisibility,
    Monitor,
    Outcome,
    StepDecision,
    StepRecord,
    SubagentHalt,
    TaskAuthor,
)
from langchain_sync_monitors.feedback import (
    build_blocked_attempt_messages,
    build_monitor_message_id,
)
from langchain_sync_monitors.options import check_enum_option
from langchain_sync_monitors.pending_steps import (
    AsyncPendingStep,
    SyncPendingStep,
    run_synchronously,
)
from langchain_sync_monitors.records import build_step_record, find_monitor_records

logger = logging.getLogger(__name__)

REQUEST_ONLY_MIDDLEWARE = frozenset(
    {
        "AnthropicPromptCachingMiddleware",
        "BedrockPromptCachingMiddleware",
        "FireworksPromptCachingMiddleware",
        "MemoryMiddleware",
        "UnsupportedContentMiddleware",
        "_ToolExclusionMiddleware",
    },
)
"""Classes that only rewrite the request, so they are safe inside a monitor.

Deep Agents places these after user middleware [@deepagents2026].
"""


class MonitorState(AgentState):
    """The agent state with the log of every monitored step.

    The reducer comes last in the annotation because LangGraph reads it only
    from the last metadata position; anywhere else the log silently keeps only
    its last write [@langgraph2026]. `OmitFromInput` keeps the log out of a
    subagent's input, so a subagent never copies its parent's records back.
    """

    monitor_log: NotRequired[Annotated[list[StepRecord], OmitFromInput, operator.add]]


class MonitorPlacementWarning(UserWarning):
    """A middleware inside a monitor can return state updates for samples the monitor rejects."""


def build_subagent_halt_decision(*, subagent_names: Sequence[str]) -> StepDecision:
    """Return the decision that ends the run because a subagent was halted."""
    distinct_names = sorted(set(subagent_names))
    subject = "subagent" if len(distinct_names) == 1 else "subagents"
    message = AIMessage(
        content=(
            f"[Safety monitor] Stopped: the safety monitor halted the {subject} "
            f"{', '.join(distinct_names)}, so this agent stops too."
        ),
        id=build_monitor_message_id(),
    )
    return StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[message]),
        samples=(),
        executed_sample=None,
        flagged=True,
    )


def find_new_subagent_halts(
    records: Sequence[StepRecord],
    *,
    agent: str,
) -> list[StepRecord]:
    """Return the halts of other agents logged since this agent's last step.

    Only halts this agent has not yet answered count, so a halted run does not
    stay halted on the next turn of a checkpointed thread.
    """
    own_positions = [index for index, record in enumerate(records) if record["agent"] == agent]
    start = own_positions[-1] + 1 if own_positions else 0
    return [
        record
        for record in records[start:]
        if record["agent"] != agent and record["outcome"] == "halted"
    ]


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitorMiddleware(AgentMiddleware[MonitorState, AgentContext, StructuredOutput]):
    """Puts a monitor and a control protocol around every model call of an agent.

    The monitor judges each proposed step before any tool runs, and the
    protocol decides what the agent state receives. Only the decision is
    committed, with one record that keeps every judged sample for audit.
    Nothing the protocol calls streams to `stream_mode="messages"`: not the
    samples, not the trusted model's step, not the monitor's own calls. The
    committed step streams whole once the model node returns it.
    Place it last in a `create_agent` middleware list; `check_monitor_placement`
    checks a list.

    `label` and `agent_name` make up the name, `monitor[main]` by default,
    which LangChain requires to be unique within one agent. `task_author` says
    who wrote the task the monitor reads. `feedback_visibility` decides whether
    blocked attempts and their feedback are committed with the step or only
    shown to the retry. `when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN` halts
    this agent before its next model call once a subagent was halted. Each
    option must be a member of its enum; a plain string raises
    `ConfigurationError`.

    The instance holds configuration only. Deep Agents runs parallel subagents
    through shared middleware instances, so every piece of run state lives in
    the graph state.
    """

    monitor: Monitor
    protocol: ControlProtocol
    label: str = "monitor"
    agent_name: str = "main"
    task_author: TaskAuthor = TaskAuthor.USER
    feedback_visibility: FeedbackVisibility = FeedbackVisibility.HIDDEN
    when_subagent_halts: SubagentHalt = SubagentHalt.STOP_SUBAGENT_ONLY

    state_schema = MonitorState
    transformers = (InternalCallTransformer,)

    def __post_init__(self) -> None:
        """Reject an option that is not a member of its enum, such as a plain string."""
        check_enum_option(self.task_author, option_type=TaskAuthor, parameter_name="task_author")
        check_enum_option(
            self.feedback_visibility,
            option_type=FeedbackVisibility,
            parameter_name="feedback_visibility",
        )
        check_enum_option(
            self.when_subagent_halts,
            option_type=SubagentHalt,
            parameter_name="when_subagent_halts",
        )

    @property
    @override
    def name(self) -> str:
        """The middleware's name, unique per agent as LangChain requires."""
        return f"{self.label}[{self.agent_name}]"

    def copy_for_subagent(self, *, subagent_name: str) -> Self:
        """Return a monitor for a subagent, whose task was written by the parent agent."""
        return replace(self, agent_name=subagent_name, task_author=TaskAuthor.PARENT_AGENT)

    @override
    def wrap_model_call(
        self,
        request: AgentModelRequest,
        handler: ModelCallHandler,
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Decide this step under `invoke()`, driving the protocol without an event loop."""
        records = read_monitor_log(request.state)
        previous_records = find_monitor_records(records, monitor=self.label, agent=self.agent_name)
        decision = self.find_halt_decision(records)
        if decision is None:
            step = SyncPendingStep(
                request=request,
                handler=handler,
                monitor=self.monitor,
                task_author=self.task_author,
                previous_records=previous_records,
            )
            with hide_model_calls_from_message_stream():
                decision = run_synchronously(self.protocol.decide(step))
        return self.commit(request, decision=decision, previous_records=previous_records)

    @override
    async def awrap_model_call(
        self,
        request: AgentModelRequest,
        handler: AsyncModelCallHandler,
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Decide this step under `ainvoke()`, awaiting the protocol."""
        records = read_monitor_log(request.state)
        previous_records = find_monitor_records(records, monitor=self.label, agent=self.agent_name)
        decision = self.find_halt_decision(records)
        if decision is None:
            step = AsyncPendingStep(
                request=request,
                handler=handler,
                monitor=self.monitor,
                task_author=self.task_author,
                previous_records=previous_records,
            )
            with hide_model_calls_from_message_stream():
                decision = await self.protocol.decide(step)
        return self.commit(request, decision=decision, previous_records=previous_records)

    def find_halt_decision(self, records: Sequence[StepRecord]) -> StepDecision | None:
        """Return a halt when a subagent halted and this monitor stops the whole run, else None."""
        if self.when_subagent_halts is SubagentHalt.STOP_SUBAGENT_ONLY:
            return None
        halts = find_new_subagent_halts(records, agent=self.agent_name)
        if not halts:
            return None
        return build_subagent_halt_decision(subagent_names=[record["agent"] for record in halts])

    def commit(
        self,
        request: AgentModelRequest,
        *,
        decision: StepDecision,
        previous_records: tuple[StepRecord, ...],
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Commit the decided messages and append one record of the step to `monitor_log`.

        The record is also written to `stream_mode="custom"` as it is committed.
        With `FeedbackVisibility.IN_TRANSCRIPT`, each blocked attempt and its
        feedback come before the step's own messages.
        """
        record = build_step_record(
            decision=decision,
            agent=self.agent_name,
            monitor=self.label,
            step_number=len(previous_records) + 1,
        )
        write_stream_event(request, event=MonitorStepEvent(type="monitor_step", record=record))
        logger.debug(
            "%s committed step %d: %s", self.name, record["step_number"], record["outcome"]
        )
        messages = list(decision.response.result)
        if self.feedback_visibility is FeedbackVisibility.IN_TRANSCRIPT:
            messages = [*build_blocked_attempt_messages(decision=decision), *messages]
        response = ModelResponse(
            result=messages,
            structured_response=decision.response.structured_response,
        )
        update = {MONITOR_LOG_KEY: [record]}
        return ExtendedModelResponse(model_response=response, command=Command(update=update))


def is_model_call_wrapper(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls, as `create_agent` decides it."""
    middleware_class = type(middleware)
    return (
        middleware_class.wrap_model_call is not AgentMiddleware.wrap_model_call
        or middleware_class.awrap_model_call is not AgentMiddleware.awrap_model_call
    )


def is_unsafe_inside_monitor(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls and is not known to only rewrite the request."""
    is_request_only = type(middleware).__name__ in REQUEST_ONLY_MIDDLEWARE
    return is_model_call_wrapper(middleware) and not is_request_only


def find_middleware_inside_monitor(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware after the last monitor, which LangChain nests inside it."""
    monitor_positions = [
        index for index, item in enumerate(middleware) if isinstance(item, MonitorMiddleware)
    ]
    if not monitor_positions:
        return ()
    return middleware[monitor_positions[-1] + 1 :]


def check_monitor_placement(*, middleware: Sequence[AnyAgentMiddleware]) -> list[str]:
    """Warn about each middleware inside the last monitor that may return state updates.

    Pass the list given to `create_agent`. Only a middleware that wraps model
    calls runs inside the monitor, and a known request-only one is safe there.
    Any other can return commands, which LangChain collects per call of the
    monitor's handler, so they would pile up from every sample the protocol
    draws [@langchain2026]. Returns the names of the middleware it warned about.
    """
    misplaced = [
        item.name
        for item in find_middleware_inside_monitor(middleware)
        if is_unsafe_inside_monitor(item)
    ]
    for name in misplaced:
        warnings.warn(
            f"{name} wraps model calls inside a monitor, so any state update it returns "
            "is merged across every sample the monitor draws. Put the monitor last.",
            MonitorPlacementWarning,
            stacklevel=2,
        )
    return misplaced
