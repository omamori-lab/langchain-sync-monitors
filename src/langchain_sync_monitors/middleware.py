"""The middleware that puts a monitor and a control protocol around every model call.

`MonitorMiddleware` wraps each model call of an agent. The untrusted model's
proposal is judged before any tool runs, the protocol decides what the agent
state receives, and one `StepRecord` per step is appended to `monitor_log`.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, and a
middleware inside the monitor that returns state updates would have them
taken from a sample the protocol may not commit: LangChain keeps the updates of
the last inner call, and mixes them when samples are drawn concurrently
[@langchain2026].
`check_monitor_placement` warns about such a list, and about a middleware
outside the monitor that retries failed model calls.
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
    OmitFromOutput,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentContext,
    AgentModelRequest,
    AgentRuntime,
    AgentStateUpdate,
    AnyAgentMiddleware,
    AsyncModelCallHandler,
    AsyncToolCallHandler,
    ModelCallHandler,
    MonitorStepEvent,
    MonitorStepFailedEvent,
    StructuredOutput,
    ToolCallHandler,
    ToolCallResult,
    hide_model_calls_from_message_stream,
    read_monitor_log,
    write_stream_event,
)
from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Delegation,
    FeedbackVisibility,
    Monitor,
    SampleRecord,
    StepDecision,
    StepRecord,
    SubagentHalt,
    TaskAuthor,
)
from langchain_sync_monitors.delegation import (
    add_delegation,
    build_subagent_halt_decision,
    count_blocks_in_thread,
    count_new_subagent_blocks,
    find_new_subagent_halts,
    read_delegation_id,
)
from langchain_sync_monitors.feedback import build_blocked_attempt_messages
from langchain_sync_monitors.options import check_enum_option
from langchain_sync_monitors.pending_steps import (
    AsyncPendingStep,
    MonitoredStep,
    SyncPendingStep,
    run_synchronously,
)
from langchain_sync_monitors.records import (
    build_sample_record,
    build_step_record,
    find_monitor_records,
)

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

RETRYING_MIDDLEWARE = frozenset(
    {
        "ModelFallbackMiddleware",
        "ModelRetryMiddleware",
        "_DeepAgentsSummarizationMiddleware",
    },
)
"""Classes that call the rest of the stack again when a model call raises.

LangChain's retry and fallback middleware retry on an exception, and Deep
Agents' summarisation retries after a context overflow
[@langchain2026; @deepagents2026]. Outside a monitor, each retry runs the
whole monitored step again.
"""


class MonitorState(AgentState):
    """The agent state with the log of every monitored step.

    The reducer comes last in the annotation because LangGraph reads it only
    from the last metadata position; anywhere else the log silently keeps only
    its last write [@langgraph2026]. `OmitFromInput` keeps the log out of a
    subagent's input, so a subagent never copies its parent's records back.

    `monitor_delegation` holds the `Delegation` a subagent was started with.
    The monitor adds it to the state each tool call sees, and Deep Agents'
    `task` tool passes that state on to the subagent it starts.
    `OmitFromOutput` keeps it out of the subagent's result, so it never flows
    back into the parent, as Deep Agents does for its own forked-context flag
    [@deepagents2026].
    """

    monitor_log: NotRequired[Annotated[list[StepRecord], OmitFromInput, operator.add]]
    monitor_delegation: NotRequired[Annotated[Delegation, OmitFromOutput]]


class MonitorPlacementWarning(UserWarning):
    """A middleware placed around or inside a monitor undermines what the monitor records.

    Inside a monitor, a middleware can return state updates for samples the
    monitor rejects. Outside it, a middleware that retries failed model calls
    runs the whole step again, and the samples judged before the failure never
    reach `monitor_log`.
    """


def build_end_run_update() -> AgentStateUpdate:
    """Return the update with which an `after_model` hook ends the agent's run [@langchain2026]."""
    return {"jump_to": "end"}


def render_judged_samples(samples: Sequence[SampleRecord]) -> str:
    """Render judged samples for a log line: each one's suspicion, reason and proposal."""
    return "\n".join(
        f"- suspicion {sample['suspicion']:.3f}, reason: {sample['reason']}\n{sample['proposal']}"
        for sample in samples
    )


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

    Before each tool call, the middleware adds a `Delegation` to the state the
    tool sees: the call's id and the blocks each monitor has recorded in the
    thread. A subagent the call starts, as Deep Agents' `task` tool does,
    receives it, so the subagent's records carry the call's id as
    `delegation_id` and its Auto Mode counts from the thread's total.

    A halted step ends the run. The middleware's `after_model` hook routes the
    agent to its end, since the halt message alone does not end an agent that
    loops until it has a structured response. The hook adds one graph step per
    model call, which counts towards an explicit `recursion_limit`, and on a
    halted step it skips the `after_model` hooks that would run after it.

    If a call inside a step raises before the protocol decides, the step is
    not committed. The samples the monitor had judged are logged as a warning
    and written to `stream_mode="custom"` as a `MonitorStepFailedEvent`, and
    the exception is raised again.

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
                blocks_in_thread=count_blocks_in_thread(request.state, monitor=self.label),
                new_subagent_blocks=count_new_subagent_blocks(
                    records, agent=self.agent_name, monitor=self.label
                ),
            )
            try:
                with hide_model_calls_from_message_stream():
                    decision = run_synchronously(self.protocol.decide(step))
            except BaseException as error:
                self.report_failed_step(request, step=step, error=error)
                raise
            finally:
                step.close()
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
                blocks_in_thread=count_blocks_in_thread(request.state, monitor=self.label),
                new_subagent_blocks=count_new_subagent_blocks(
                    records, agent=self.agent_name, monitor=self.label
                ),
            )
            try:
                with hide_model_calls_from_message_stream():
                    decision = await self.protocol.decide(step)
            except BaseException as error:
                self.report_failed_step(request, step=step, error=error)
                raise
        return self.commit(request, decision=decision, previous_records=previous_records)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `invoke()`, handing any subagent it starts its delegation."""
        return handler(add_delegation(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `ainvoke()`, handing any subagent it starts its delegation."""
        return await handler(add_delegation(request))

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """End the run after a step this monitor halted, under `invoke()`."""
        return build_end_run_update() if self.has_just_halted(state) else None

    @hook_config(can_jump_to=["end"])
    @override
    async def aafter_model(
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """End the run after a step this monitor halted, under `ainvoke()`."""
        return build_end_run_update() if self.has_just_halted(state) else None

    def has_just_halted(self, state: MonitorState) -> bool:
        """Tell whether the step just committed is this monitor's halt.

        The model node's own `jump_to` would not do: a routing edge reads a
        fresh copy of the state in which only its own node's writes survive,
        and `jump_to` is cleared everywhere else [@langgraph2026]. With any
        `after_model` hook in the agent, the model node has no routing edge of
        its own, so the hook that follows it has to write `jump_to` itself.
        The last message must be the halt, a final message with no tool calls,
        so an older halt record never ends a later turn.
        """
        own_records = find_monitor_records(
            read_monitor_log(state),
            monitor=self.label,
            agent=self.agent_name,
        )
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        return (
            bool(own_records)
            and own_records[-1]["outcome"] == "halted"
            and isinstance(last_message, AIMessage)
            and not last_message.tool_calls
        )

    def find_halt_decision(self, records: Sequence[StepRecord]) -> StepDecision | None:
        """Return a halt when a subagent halted and this monitor stops the whole run, else None."""
        if self.when_subagent_halts is SubagentHalt.STOP_SUBAGENT_ONLY:
            return None
        halts = find_new_subagent_halts(records, agent=self.agent_name)
        if not halts:
            return None
        return build_subagent_halt_decision(subagent_names=[record["agent"] for record in halts])

    def report_failed_step(
        self,
        request: AgentModelRequest,
        *,
        step: MonitoredStep,
        error: BaseException,
    ) -> None:
        """Report the samples judged in a step that raised, before the error propagates.

        The step is never committed, so this is the only trace of its samples:
        a `MonitorStepFailedEvent` on `stream_mode="custom"` and, when the
        monitor had judged anything, a warning that lists each sample.
        """
        step_number = len(step.previous_records) + 1
        samples = [build_sample_record(sample, executed=False) for sample in step.judged_samples]
        event = MonitorStepFailedEvent(
            type="monitor_step_failed",
            agent=self.agent_name,
            monitor=self.label,
            step_number=step_number,
            error=f"{type(error).__name__}: {error}",
            samples=samples,
        )
        delegation_id = read_delegation_id(request.state)
        if delegation_id is not None:
            event["delegation_id"] = delegation_id
        write_stream_event(request, event=event)
        if samples:
            logger.warning(
                "%s: step %d failed with %s before it was committed, so the %d sample(s) the "
                "monitor judged are not in monitor_log:\n%s",
                self.name,
                step_number,
                event["error"],
                len(samples),
                render_judged_samples(samples),
            )

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
            delegation_id=read_delegation_id(request.state),
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


def is_retrying_middleware(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls and is known to retry them when they raise."""
    is_retrying = type(middleware).__name__ in RETRYING_MIDDLEWARE
    return is_retrying and is_model_call_wrapper(middleware)


def find_last_monitor_position(middleware: Sequence[AnyAgentMiddleware]) -> int | None:
    """Return the position of the last monitor in the list, or None when there is none."""
    monitor_positions = [
        index for index, item in enumerate(middleware) if isinstance(item, MonitorMiddleware)
    ]
    return monitor_positions[-1] if monitor_positions else None


def find_middleware_inside_monitor(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware after the last monitor, which LangChain nests inside it."""
    position = find_last_monitor_position(middleware)
    return () if position is None else middleware[position + 1 :]


def find_middleware_outside_monitor(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware before the last monitor, which LangChain wraps around it."""
    position = find_last_monitor_position(middleware)
    return () if position is None else middleware[:position]


def check_monitor_placement(*, middleware: Sequence[AnyAgentMiddleware]) -> list[str]:
    """Warn about each middleware placed where it undermines the last monitor.

    Pass the list given to `create_agent`. Two placements are warned about:

    - inside the monitor, a middleware that wraps model calls and is not known
      to only rewrite the request. It can return commands, which LangChain
      collects per call of the monitor's handler, so they would pile up from
      every sample the protocol draws [@langchain2026];
    - outside the monitor, a middleware known to retry failed model calls,
      such as `ModelRetryMiddleware`. A retry runs the whole step again with
      fresh samples, and the samples judged before the failure reach only a
      warning log line and a `MonitorStepFailedEvent`, never `monitor_log`.

    Returns the names of the middleware it warned about.
    """
    misplaced_inside = [
        item.name
        for item in find_middleware_inside_monitor(middleware)
        if is_unsafe_inside_monitor(item)
    ]
    retrying_outside = [
        item.name
        for item in find_middleware_outside_monitor(middleware)
        if is_retrying_middleware(item)
    ]
    for name in misplaced_inside:
        warnings.warn(
            f"{name} wraps model calls inside a monitor, so a state update it returns "
            "may come from a sample the protocol does not commit. Put the monitor last.",
            MonitorPlacementWarning,
            stacklevel=2,
        )
    for name in retrying_outside:
        warnings.warn(
            f"{name} retries failed model calls from outside a monitor, so a step that fails "
            "runs again from the start with fresh samples. The samples the monitor judged "
            "before the failure never reach monitor_log; only a warning log line and a "
            "monitor_step_failed event on stream_mode='custom' keep them.",
            MonitorPlacementWarning,
            stacklevel=2,
        )
    return misplaced_inside + retrying_outside
