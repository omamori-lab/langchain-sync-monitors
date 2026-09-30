"""The middleware that puts a monitor and a control protocol around every model call.

`MonitorMiddleware` wraps each model call of an agent. The untrusted model's
proposal is judged before any of the agent's own tools run, the protocol
decides what the agent state receives, and one `StepRecord` per step is
appended to `monitor_log`. Tools the model provider runs itself run inside the
model call, before the proposal is judged, and the middleware warns about the
ones it knows.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, and a
middleware inside the monitor that returns state updates would have them
taken from a sample the protocol may not commit: LangChain keeps the updates of
the last inner call, and mixes them when samples are drawn concurrently
[@langchain2026].
`check_monitor_placement` warns about such a list, about a middleware
outside the monitor that retries failed model calls, and about a middleware
that runs failed tool calls again or answers them with an error message.
`halts` has what happens after a halted step.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self, override

from langchain.agents.middleware.internal_call_transformer import InternalCallTransformer
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentContext,
    AgentModelRequest,
    AgentRuntime,
    AgentStateUpdate,
    AsyncModelCallHandler,
    AsyncToolCallHandler,
    ModelCallHandler,
    MonitorStepEvent,
    MonitorStepFailedEvent,
    StructuredOutput,
    ToolCallHandler,
    ToolCallResult,
    TracedRun,
    cast_to_tool_call_result,
    hide_model_calls_from_message_stream,
    read_monitor_log,
    write_stream_event,
)
from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Fallback,
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
    count_blocks_in_thread,
    count_new_subagent_blocks,
    read_delegation_id,
)
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.feedback import build_blocked_attempt_messages
from langchain_sync_monitors.halts import (
    build_end_run_update,
    build_halt_inputs_update,
    find_halt_decision,
    has_just_halted,
)
from langchain_sync_monitors.monitor_state import MonitorState
from langchain_sync_monitors.options import check_enum_option, check_instance_option
from langchain_sync_monitors.pending_steps import (
    AsyncPendingStep,
    MonitoredStep,
    PendingStepOptions,
    PreparedStep,
    SyncPendingStep,
    run_synchronously,
)
from langchain_sync_monitors.provider_tools import warn_about_provider_tools
from langchain_sync_monitors.records import build_sample_record, find_monitor_records
from langchain_sync_monitors.spans import (
    StepIdentity,
    build_step_span_inputs,
    open_step_span,
    open_step_span_sync,
    trace_decision,
    trace_decision_sync,
)
from langchain_sync_monitors.task_authorship import (
    TASK_MESSAGES_KEY,
    build_note_update,
    build_run_end_update,
    build_run_input_update,
    build_step_start_update,
    mark_tool_written_notes,
    read_message_ids,
)

logger = logging.getLogger(__name__)


def render_judged_samples(samples: Sequence[SampleRecord]) -> str:
    """Render judged samples for a log line: each one's suspicion, reason and proposal."""
    return "\n".join(
        f"- suspicion {sample['suspicion']:.3f}, reason: {sample['reason']}\n{sample['proposal']}"
        for sample in samples
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitorMiddleware(AgentMiddleware[MonitorState, AgentContext, StructuredOutput]):
    """Puts a monitor and a control protocol around every model call of an agent.

    The monitor judges each proposed step before any of the agent's own tools
    run, and the protocol decides what the agent state receives. Only the
    decision is committed, with one record that keeps every judged sample for
    audit. Tools the model provider runs itself, such as Anthropic's
    `web_fetch` or OpenAI's `web_search`, run inside the model call, before
    the monitor judges the step and again for every sample, so no monitor can
    stop them; the middleware emits a `ProviderToolWarning`, once per
    instance, when the model request holds such tools. `provider_tools` lists
    the ones it knows, and what it cannot see.
    Nothing the protocol calls streams to `stream_mode="messages"`: not the
    samples, not the trusted model's step, not the monitor's own calls. The
    committed step streams whole once the model node returns it.
    Place it last in a `create_agent` middleware list; `check_monitor_placement`
    checks a list.

    `label` and `agent_name` make up the name, `monitor[main]` by default,
    which LangChain requires to be unique within one agent, and a subclass
    that names itself otherwise raises `ConfigurationError`. `task_author` says
    who wrote the task the monitor reads. `feedback_visibility` decides whether
    blocked attempts and their feedback are committed with the step or only
    shown to the retry. `when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN` halts
    this agent before its next model call once a subagent was halted. Each
    option must be a member of its enum; a plain string raises
    `ConfigurationError`, as do a `monitor` that is not a `Monitor` and a
    `protocol` that is not a `ControlProtocol`, such as the fallback
    `DeferToTrustedModel` in place of the protocol `DeferToTrusted`.

    Before each tool call, the middleware adds a `Delegation` to the state the
    tool sees: the call's id, `agent_name` and the blocks each monitor has
    recorded in the thread. A subagent the call starts, as Deep Agents' `task`
    tool does, receives it, so the subagent's records carry the call's id as
    `delegation_id` and its Auto Mode counts from the thread's total.

    Only the untagged human messages a run receives as its input are read as
    the task author's. The middleware's `before_agent` hook records them in
    the graph state. Its `before_model` and `after_agent` hooks, and each
    commit, tag every other untagged human message in the state as a context
    note, and a human message a tool writes is tagged where it is written.
    After a run that stopped before reaching `after_agent`, the next run's
    new messages are notes too, since the monitor cannot tell them from what
    the stopped run left; `task_authorship` has the rule and its limits.

    A halted step ends the run. The middleware's `after_model` hook routes the
    agent to its end, since the halt message alone does not end an agent that
    loops until it has a structured response. On a halted step it skips the
    `after_model` hooks that would run after it. The hooks add two graph steps
    per agent step, however many samples or model calls it draws, and two per
    run, which count towards an explicit `recursion_limit`: N agent steps need
    4N + 2 rather than 2N, and each further monitor adds another 2N + 2.

    A halt stands until a later run brings new input. An `after_agent` hook
    can send a finished run back to the model, as Deep Agents'
    `RubricMiddleware` does when it grades the task unmet [@deepagents2026].
    While this monitor's last step is a halt and the thread has recorded no
    run input since, each further step halts again without a sample: the
    untrusted model is not called, and the step's record, flagged, holds no
    samples. A human message written between a run's start and its end never
    lifts the halt. One that a middleware listed before the monitor writes
    from its `before_agent` hook, or from its `after_agent` hook without a
    return to the model, counts as a run's input and does; `halts` has the
    rule.

    If a call inside a step raises before the protocol decides, the step is
    not committed. The samples the monitor had judged are logged as a warning
    and written to `stream_mode="custom"` as a `MonitorStepFailedEvent`, and
    the exception is raised again. LangGraph's own control flow, such as the
    `GraphInterrupt` that `interrupt()` raises, passes through unreported. Yet
    a monitor, a protocol or a fallback must not call `interrupt()`: on
    resume LangGraph runs the model node again [@langgraph2026], so the step
    is sampled and judged afresh, and the person's answer applies to samples
    they never saw.

    In LangChain tracers such as LangSmith and Langfuse, each step is a
    `monitor step` span, with the spans that `spans` describes nested in it.

    The instance holds configuration only, so it can be copied and pickled.
    Deep Agents runs parallel subagents through shared middleware instances,
    so every piece of run state lives in the graph state.
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
        """Reject an option of the wrong type, such as a plain string where an enum belongs."""
        if self.name != f"{self.label}[{self.agent_name}]":
            message = (
                f"a monitor's name must be label[agent_name], {self.label}[{self.agent_name}], "
                f"got {self.name!r}: its records and halts are kept under its label and agent, "
                "so two monitors that shared them would lift each other's halts. Give each "
                "monitor its own label instead."
            )
            raise ConfigurationError(message)
        check_instance_option(self.monitor, option_type=Monitor, parameter_name="monitor")
        protocol_hint = "Use a protocol such as AutoMode, DeferToResample or DeferToTrusted."
        if isinstance(self.protocol, Fallback):
            protocol_hint = (
                f"{type(self.protocol).__name__} is a fallback, which writes a step once a "
                "protocol rejects its samples. To defer suspicious steps to the trusted model, "
                "use DeferToTrusted(trusted_model=...)."
            )
        check_instance_option(
            self.protocol,
            option_type=ControlProtocol,
            parameter_name="protocol",
            hint=protocol_hint,
        )
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
        prepared = self.prepare_step(request)
        with open_step_span_sync(prepared.identity) as traced_step:
            # A halt found before sampling decides the step, and the protocol never runs.
            decision = prepared.halt
            if decision is None:
                step = SyncPendingStep(handler=handler, **prepared.options)
                try:
                    with hide_model_calls_from_message_stream():
                        decision = run_synchronously(self.protocol.decide(step))
                except GraphBubbleUp:
                    # LangGraph's own control flow, such as an interrupt, is not a failed step.
                    raise
                except BaseException as error:
                    self.report_failed_step(
                        request, step=step, error=error, traced_step=traced_step
                    )
                    raise
                finally:
                    # A task the protocol left on a running loop may run after the step.
                    # Once closed, the step refuses it the model.
                    step.close()
            record = prepared.identity.build_record(decision)
            trace_decision_sync(traced_step, record=record)
            return self.commit(request, decision=decision, record=record)

    @override
    async def awrap_model_call(
        self,
        request: AgentModelRequest,
        handler: AsyncModelCallHandler,
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Decide this step under `ainvoke()`, awaiting the protocol."""
        prepared = self.prepare_step(request)
        async with open_step_span(prepared.identity) as traced_step:
            # A halt found before sampling decides the step, and the protocol never runs.
            decision = prepared.halt
            if decision is None:
                step = AsyncPendingStep(handler=handler, **prepared.options)
                try:
                    with hide_model_calls_from_message_stream():
                        decision = await self.protocol.decide(step)
                except GraphBubbleUp:
                    # LangGraph's own control flow, such as an interrupt, is not a failed step.
                    raise
                except BaseException as error:
                    self.report_failed_step(
                        request, step=step, error=error, traced_step=traced_step
                    )
                    raise
            record = prepared.identity.build_record(decision)
            await trace_decision(traced_step, record=record)
            return self.commit(request, decision=decision, record=record)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `invoke()`, handing any subagent it starts its delegation.

        A new human message the tool writes is tagged as a context note, and
        no message it writes keeps the monitor's own source.
        """
        result = handler(add_delegation(request, agent=self.agent_name))
        return cast_to_tool_call_result(
            mark_tool_written_notes(
                result,
                tool_name=request.tool_call["name"],
                state=request.state,
            ),
        )

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `ainvoke()`, handing any subagent it starts its delegation.

        A new human message the tool writes is tagged as a context note, and
        no message it writes keeps the monitor's own source.
        """
        result = await handler(add_delegation(request, agent=self.agent_name))
        return cast_to_tool_call_result(
            mark_tool_written_notes(
                result,
                tool_name=request.tool_call["name"],
                state=request.state,
            ),
        )

    @override
    def before_agent(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Record the human messages this run received as its input, under `invoke()`."""
        return build_run_input_update(state)

    @override
    async def abefore_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Record the human messages this run received as its input, under `ainvoke()`."""
        return build_run_input_update(state)

    @override
    def before_model(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Record and tag the human messages so far, and open a step, under `invoke()`."""
        return build_step_start_update(state)

    @override
    async def abefore_model(  # lanorme: ignore[NAMING-011]
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Record and tag the human messages so far, and open a step, under `ainvoke()`."""
        return build_step_start_update(state)

    # Without `can_jump_to`, `create_agent` gives the hook a plain edge and ignores `jump_to`.
    @hook_config(can_jump_to=["end"])
    @override
    def after_model(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """End the run after a step this monitor halted, under `invoke()`."""
        return self.build_halt_end_update(state)

    @hook_config(can_jump_to=["end"])
    @override
    async def aafter_model(
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """End the run after a step this monitor halted, under `ainvoke()`."""
        return self.build_halt_end_update(state)

    @override
    def after_agent(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Tag the notes written since the last step, and close the run, under `invoke()`."""
        return build_run_end_update(state)

    @override
    async def aafter_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Tag the notes written since the last step, and close the run, under `ainvoke()`."""
        return build_run_end_update(state)

    def build_halt_end_update(self, state: MonitorState) -> AgentStateUpdate | None:
        """Return the update that ends the run right after this monitor's halt, else None."""
        if has_just_halted(state, monitor=self.label, agent=self.agent_name):
            return build_end_run_update()
        return None

    def prepare_step(self, request: AgentModelRequest) -> PreparedStep:
        """Read the log once for a step: its identity, a halt without a sample, its options."""
        warn_about_provider_tools(request, middleware=self, middleware_name=self.name)
        records = read_monitor_log(request.state)
        previous_records = find_monitor_records(records, monitor=self.label, agent=self.agent_name)
        identity = StepIdentity(
            monitor=self.label,
            agent=self.agent_name,
            step_number=len(previous_records) + 1,
            protocol=type(self.protocol).__name__,
            delegation_id=read_delegation_id(request.state),
        )
        halt = find_halt_decision(
            request.state,
            previous_records=previous_records,
            agent=self.agent_name,
            monitor=self.name,
            when_subagent_halts=self.when_subagent_halts,
        )
        options = PendingStepOptions(
            request=request,
            monitor=self.monitor,
            task_author=self.task_author,
            task_message_ids=read_message_ids(request.state, key=TASK_MESSAGES_KEY),
            previous_records=previous_records,
            blocks_in_thread=count_blocks_in_thread(request.state, monitor=self.label),
            new_subagent_blocks=count_new_subagent_blocks(
                records, agent=self.agent_name, monitor=self.label
            ),
        )
        return PreparedStep(identity=identity, halt=halt, options=options)

    def report_failed_step(
        self,
        request: AgentModelRequest,
        *,
        step: MonitoredStep,
        error: BaseException,
        traced_step: TracedRun,
    ) -> None:
        """Report the samples judged in a step that raised, before the error propagates.

        The step is never committed, so this is the only trace of its samples:
        a `MonitorStepFailedEvent` on `stream_mode="custom"`, the step span's
        inputs, which name the first sample judged, and, when the monitor had
        judged anything, a warning that lists each sample.
        """
        step_number = len(step.previous_records) + 1
        samples = [build_sample_record(sample, executed=False) for sample in step.judged_samples]
        traced_step.inputs_at_end = build_step_span_inputs(step_number=step_number, samples=samples)
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
        record: StepRecord,
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Commit the decided messages and append the step's record to `monitor_log`.

        The record is also written to `stream_mode="custom"` as it is committed.
        With `FeedbackVisibility.IN_TRANSCRIPT`, each blocked attempt and its
        feedback come before the step's own messages. The untagged human
        messages in the state that the monitor had not seen are recorded as
        seen, so the next run does not take them for its input.
        """
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
        update = {
            MONITOR_LOG_KEY: [record],
            # A middleware listed after the monitor runs its `before_model` hook after the
            # monitor's own, so a human message it wrote is first recorded here.
            **build_note_update(request.state),
            **build_halt_inputs_update(record, state=request.state, monitor=self.name),
        }
        return ExtendedModelResponse(model_response=response, command=Command(update=update))
