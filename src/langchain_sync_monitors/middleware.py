"""The middleware that puts a monitor and a control protocol around every model call.

`MonitorMiddleware` wraps each model call of an agent. The untrusted model's
proposal is judged before any of the agent's own tools run, the protocol
decides what the agent state receives, and one `StepRecord` per step is
appended to `monitor_log`. Server-side tools, which the model provider runs
inside the model call, run before the proposal is judged, and the middleware
warns about the ones it knows.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, and a
middleware inside the monitor that returns state updates would have them
taken from a sample the protocol may not commit: LangChain keeps the updates of
the last inner call, and mixes them when samples are drawn concurrently
[@langchain2026].
`check_monitor_placement` warns about such a list, about a middleware
outside the monitor that retries failed model calls, and about a middleware
that runs failed tool calls again or answers them with an error message. It
also warns about a second monitor inside one whose protocol can call the
model more than once in a step, which loses the records of all but its last
call or piles them up for calls drawn at once, and about a monitor inside
another that may block, since the outer monitor's trajectory then leaves the
blocked replies out.
`halts` has what happens after a halted step.
"""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Self, override

from hyoshigi import Redaction
from langchain.agents.middleware.internal_call_transformer import InternalCallTransformer
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors._langchain import (
    AgentContext,
    AgentModelRequest,
    AgentRuntime,
    AgentStateUpdate,
    AsyncModelCallHandler,
    AsyncToolCallHandler,
    ModelCallHandler,
    StructuredOutput,
    ToolCallHandler,
    ToolCallResult,
    TracedRun,
    hide_model_calls_from_message_stream,
)
from langchain_sync_monitors.audits import (
    find_held_step,
    resolve_held_step,
    stop_held_step,
    trace_answer,
    trace_answer_sync,
)
from langchain_sync_monitors.commits import build_step_response, commit_step, report_failed_step
from langchain_sync_monitors.context_values import set_context_value
from langchain_sync_monitors.contracts import StepRecord, SubagentHalt, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.given_ids import (
    build_marked_run_end_update,
    build_run_start_trajectories_update,
)
from langchain_sync_monitors.halts import build_halt_end_update
from langchain_sync_monitors.message_ids import build_run_start_update
from langchain_sync_monitors.model_calls import resolve_chat_model
from langchain_sync_monitors.monitor_state import MonitorState
from langchain_sync_monitors.monitors import (
    MonitorConfiguration,
    check_monitor_option,
    repeats_prompts_to_cached_model,
    warn_about_cached_monitor_replies,
)
from langchain_sync_monitors.options import (
    check_enum_option,
    check_instance_option,
    check_name_part_option,
    read_threshold_value,
)
from langchain_sync_monitors.protocols import (
    DEFAULT_FEEDBACK_TEMPLATE,
    FeedbackTemplate,
    ProtocolConfiguration,
    ProtocolTwins,
    check_protocol_option,
    check_trusted_model_use,
)
from langchain_sync_monitors.score_export import Tracer, check_score_export, send_step_scores
from langchain_sync_monitors.spans import (
    counting_judgements,
    open_step_span,
    open_step_span_sync,
    trace_decision,
    trace_decision_sync,
)
from langchain_sync_monitors.step_preparation import prepare_step
from langchain_sync_monitors.steps import Step, run_step, run_step_sync
from langchain_sync_monitors.task_authorship import build_step_start_update
from langchain_sync_monitors.threshold_defaults import warn_if_defaulted
from langchain_sync_monitors.tool_calls import run_tool_call, run_tool_call_sync
from langchain_sync_monitors.trajectories import check_trajectory_kept, monitor_step_open


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitorMiddleware(AgentMiddleware[MonitorState, AgentContext, StructuredOutput]):
    """Puts a monitor and a control protocol around every model call of an agent.

    The monitor rates each proposed step before any of the agent's own tools
    run, and the protocol, `DeferToResample` or `BlockAndFeedBack`, decides
    what the agent state receives, as hyoshigi's protocols decide
    [@hyoshigi2026]; `steps` has how the middleware drives them. Only the
    decision is committed, with one record that keeps every rated sample for
    audit. `monitor` is one of the library's monitors or a `MonitorTwins`:
    under `invoke()` its sync twin rates, and under `ainvoke()` its async
    twin, and so with the protocol's twins. A monitor that asks a cached
    chat model one prompt more than once warns `CachedResampleWarning` once
    per process, when the middleware is built. Server-side tools, such as
    Anthropic's `web_fetch` or OpenAI's `web_search`, run inside the model
    call, before the monitor judges the step and again for every sample, so
    no monitor can stop them; the
    middleware emits a `ServerToolWarning`, once per instance, when the model
    request holds such tools. `server_tools` lists the ones it knows, and
    what it cannot see.
    Nothing the step calls streams to `stream_mode="messages"`: not the
    samples, not the trusted model's step, not the monitor's own calls. The
    committed step streams whole once the model node returns it.
    Place it last in a `create_agent` middleware list; `check_monitor_placement`
    checks a list.

    `label` and `agent_name` make up the name, `monitor[main]` by default,
    which LangChain requires to be unique within one agent, and a subclass
    that names itself otherwise raises `ConfigurationError`. Each must be a
    non-blank string without `:` or `|`, which LangGraph refuses in the names
    of the graph nodes the hooks become. `task_author` says
    who wrote the task the monitor reads.
    `when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN` halts this agent before
    its next model call once a subagent was halted. Each option must be a
    member of its enum; a plain string raises `ConfigurationError`, as do a
    `monitor` that is not one of the monitors and a `protocol` that is not one
    of the protocols. A setting hyoshigi refuses raises `ConfigurationError`
    here, since a protocol is built with its monitor.

    `trusted_model`, a chat model or a provider string, writes the step when
    the protocol asks for a trusted step, with the same conversation and
    tools; without one, the step is audited. A `trusted_model` the protocol
    never asks for raises `ConfigurationError`. `feedback` is the message a
    blocked step sends the agent, with the monitor's reason; the blocked reply
    and the feedback are committed with the step that follows them. A step is
    flagged for a person to review when its protocol flags it, when it halts,
    and, with `flag_threshold`, a number from 0 to 1, when its lowest sample
    is at or above it. A step the protocol audits is held: the middleware's
    `after_model` hook pauses the run before any of the agent's own tools
    run, for a person's answer, and new input in place of an answer stops
    the step; `audits` has the rules.

    Before each tool call, the middleware adds a `Delegation` to the state the
    tool sees: the call's id, `agent_name`, and the run each monitor's
    trajectory of the subagent joins. A subagent the call starts, as Deep
    Agents' `task` tool does, receives it, so the subagent's records carry
    the call's id as `delegation_id`, and its trajectory takes its place in
    the run, whose records its protocol counts. A record
    is this agent's own only when it names `agent_name` and this agent's own
    delegation, so a subagent that shares the name, such as a fork, is never
    taken for this agent. The records a call returns are checked where they
    are written, and the subagent halts they hold wait in private state for
    this agent's next step; `returned_records` and
    `subagent_returns` have the rules.

    Only the untagged human messages a run receives as its input are read as
    the task author's. The middleware's `before_agent` hook records them in
    the graph state, and adds them to the monitor's trajectory as they first
    arrive, so the monitor reads each one even after summarisation or a tool
    took it out of the model request; a later rewrite of a message the
    trajectory holds does not reach the monitor. Its `before_model` and
    `after_agent` hooks, and each commit, tag every other untagged human
    message in the state as a context note, and a human message a tool writes
    is tagged where it is written. A tool's writes to the state keys only the
    monitor writes are dropped.
    After a run that stopped before reaching `after_agent`, the next run's
    new messages are notes too, since the monitor cannot tell them from what
    the stopped run left; `task_authorship` has the rule and its limits.

    Each monitor keeps its trajectory of the run under its label, in the
    state key `monitor_trajectory`, each message in it once, and each
    subagent's joined under the call that started it; `trajectories` and
    `trajectory_runs` have the rules. A stale write to it outside a stack of
    monitors, or a subagent's trajectory that cannot join, counts as lost,
    and the monitor's next step halts without a sample. A thread whose
    decided steps outlive the trajectory raises `MonitorError`.

    `redaction`, hyoshigi's `Redaction`, redacts each entry, rating and record
    before the trajectory keeps it, so what the monitor keeps and emits holds
    the redacted text; `messages` keep the original. The function must return
    its own output unchanged. The how-to Redact what the monitor keeps has the
    rest.

    A halted step ends the run. The middleware's `after_model` hook routes the
    agent to its end, since the halt message alone does not end an agent that
    loops until it has a structured response. On a halted step it skips the
    `after_model` hooks that would run after it. The hooks add two graph steps
    per agent step, however many model calls the step makes, and two per run,
    which count towards an explicit `recursion_limit`; the how-to Read the
    monitor log gives the limit a run needs.

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

    If a call inside a step raises, the step is not committed. The samples
    the monitor had rated are written to `stream_mode="custom"` as a
    `MonitorStepFailedEvent`, their suspicions are logged as a warning, and
    the exception is raised again. LangGraph's own control flow, such as the
    `GraphInterrupt` that `interrupt()` raises, passes through unreported.
    No monitor pauses inside the model node: on resume LangGraph runs the
    node again [@langgraph2026], so the step would be sampled and rated
    afresh, and the answer would apply to samples the person never saw.

    In LangChain tracers such as LangSmith and Langfuse, each step is a
    `monitor step` span, with the spans that `spans` describes nested in it.
    `export_scores`, empty by default, names the tools that also get each
    step's highest and lowest suspicion as scores; `Tracer` has the rules.

    The instance keeps no run state of its own: Deep Agents runs parallel
    subagents through shared middleware instances, so every piece of run state
    lives in the graph state, and one instance serves parallel runs. It can be
    copied or pickled only when its monitor, protocol and models can; OpenRouter
    chat models and TypeSafe's clients hold HTTP clients that cannot.
    """

    monitor: MonitorConfiguration
    protocol: ProtocolConfiguration
    trusted_model: str | BaseChatModel | None = None
    feedback: FeedbackTemplate = DEFAULT_FEEDBACK_TEMPLATE
    flag_threshold: float | None = None
    redaction: Redaction | None = None
    label: str = "monitor"
    agent_name: str = "main"
    task_author: TaskAuthor = TaskAuthor.USER
    when_subagent_halts: SubagentHalt = SubagentHalt.STOP_SUBAGENT_ONLY
    export_scores: AbstractSet[Tracer] = frozenset()
    protocols: ProtocolTwins = field(init=False, repr=False)

    # langchain declares both as writable attributes and only reads them from a middleware;
    # this frozen dataclass keeps them read-only.
    state_schema = MonitorState  # ty: ignore[invalid-attribute-override]
    transformers = (InternalCallTransformer,)  # ty: ignore[invalid-attribute-override]

    def __post_init__(self) -> None:
        """Reject an option of the wrong type, build the protocol, and finally warn for defaults."""
        check_name_part_option(self.label, parameter_name="label")
        check_name_part_option(self.agent_name, parameter_name="agent_name")
        if self.name != f"{self.label}[{self.agent_name}]":
            message = (
                f"a monitor's name must be label[agent_name], {self.label}[{self.agent_name}], "
                f"got {self.name!r}: its records and halts are kept under its label and agent, "
                "so two monitors that shared them would lift each other's halts. Give each "
                "monitor its own label instead."
            )
            raise ConfigurationError(message)
        check_monitor_option(self.monitor, parameter_name="monitor")
        if repeats_prompts_to_cached_model(self.monitor):
            warn_about_cached_monitor_replies()
        check_protocol_option(self.protocol, parameter_name="protocol")
        check_instance_option(
            self.feedback,
            option_type=FeedbackTemplate,
            parameter_name="feedback",
            hint="Wrap the text in FeedbackTemplate(text=...).",
        )
        if self.flag_threshold is not None:
            flag_threshold = read_threshold_value(
                self.flag_threshold,
                parameter_name="flag_threshold",
            )
            object.__setattr__(self, "flag_threshold", flag_threshold)
        if self.redaction is not None:
            check_instance_option(
                self.redaction,
                option_type=Redaction,
                parameter_name="redaction",
                hint="Build one with Redaction(text=..., media=...).",
            )
        protocols = self.protocol.build(self.monitor)
        if self.trusted_model is not None:
            check_trusted_model_use(protocols, protocol_name=type(self.protocol).__name__)
            object.__setattr__(self, "trusted_model", resolve_chat_model(self.trusted_model))
        object.__setattr__(self, "protocols", protocols)
        check_enum_option(self.task_author, option_type=TaskAuthor, parameter_name="task_author")
        check_enum_option(
            self.when_subagent_halts,
            option_type=SubagentHalt,
            parameter_name="when_subagent_halts",
        )
        object.__setattr__(self, "export_scores", check_score_export(self.export_scores))
        warn_if_defaulted(protocols.defaulted)

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
        """Decide this step under `invoke()`, driving the sync protocol."""
        prepared = prepare_step(request, middleware=self)
        step = prepared.step
        with open_step_span_sync(prepared.identity) as traced_step:
            # A halt found before drawing decides the step, and the protocol never runs.
            response = ModelResponse(result=[prepared.halt]) if prepared.halt else None
            if response is None:
                with report_failed_step(
                    step,
                    identity=prepared.identity,
                    traced_step=traced_step,
                    middleware_name=self.name,
                ):
                    with (
                        hide_model_calls_from_message_stream(),
                        set_context_value(monitor_step_open, value=True),
                        counting_judgements(),
                    ):
                        result = run_step_sync(
                            step,
                            handler=handler,
                            protocol=self.protocols.protocol,
                        )
                    response = build_step_response(result, step=step)
            record = prepared.identity.build_record(
                step.decisions,
                flag_threshold=self.flag_threshold,
            )
            trace_decision_sync(traced_step, record=record)
            committed = commit_step(
                request,
                response=response,
                record=record,
                middleware_name=self.name,
                trajectory=step.trajectory,
            )
            # A committed step, or a held one before its answer, sends its scores.
            self.send_scores(traced_step, step=step, record=record)
            return committed

    @override
    async def awrap_model_call(
        self,
        request: AgentModelRequest,
        handler: AsyncModelCallHandler,
    ) -> ExtendedModelResponse[StructuredOutput]:
        """Decide this step under `ainvoke()`, awaiting the async protocol."""
        prepared = prepare_step(request, middleware=self)
        step = prepared.step
        async with open_step_span(prepared.identity) as traced_step:
            # A halt found before drawing decides the step, and the protocol never runs.
            response = ModelResponse(result=[prepared.halt]) if prepared.halt else None
            if response is None:
                with report_failed_step(
                    step,
                    identity=prepared.identity,
                    traced_step=traced_step,
                    middleware_name=self.name,
                ):
                    with (
                        hide_model_calls_from_message_stream(),
                        set_context_value(monitor_step_open, value=True),
                        counting_judgements(),
                    ):
                        result = await run_step(
                            step,
                            handler=handler,
                            protocol=self.protocols.async_protocol,
                        )
                    response = build_step_response(result, step=step)
            record = prepared.identity.build_record(
                step.decisions,
                flag_threshold=self.flag_threshold,
            )
            await trace_decision(traced_step, record=record)
            committed = commit_step(
                request,
                response=response,
                record=record,
                middleware_name=self.name,
                trajectory=step.trajectory,
            )
            # A committed step, or a held one before its answer, sends its scores.
            self.send_scores(traced_step, step=step, record=record)
            return committed

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `invoke()`, handing any subagent it starts its delegation.

        What the call writes is checked, whatever the shape of the result, of
        a `Command`'s update or of a `ParentCommand` the call raises: its
        messages as `tag_tool_written_notes` says, and its records as
        `returned_records` says. `tool_calls` has the rule.
        """
        return run_tool_call_sync(request, handler=handler, middleware=self)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolCallHandler,
    ) -> ToolCallResult:
        """Run a tool call under `ainvoke()`, handing any subagent it starts its delegation.

        What the call writes is checked as under `invoke()`.
        """
        return await run_tool_call(request, handler=handler, middleware=self)

    @override
    def before_agent(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Record this run's input, and add it to the trajectory, under `invoke()`."""
        return self.build_run_start_update(state, runtime=runtime)

    @override
    async def abefore_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Record this run's input, and add it to the trajectory, under `ainvoke()`."""
        return self.build_run_start_update(state, runtime=runtime)

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
    @hook_config(can_jump_to=["end", "model"])
    @override
    def after_model(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Pause for a person's answer to a held step, or end a halted run, under `invoke()`."""
        held = find_held_step(state, middleware=self)
        if held is None:
            return build_halt_end_update(state, monitor=self.label, agent=self.agent_name)
        answered = resolve_held_step(held, state=state, runtime=runtime, middleware=self)
        trace_answer_sync(held, answered=answered, middleware=self)
        return answered.update

    @hook_config(can_jump_to=["end", "model"])
    @override
    async def aafter_model(
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Pause for a person's answer to a held step, or end a halted run, under `ainvoke()`."""
        held = find_held_step(state, middleware=self)
        if held is None:
            return build_halt_end_update(state, monitor=self.label, agent=self.agent_name)
        answered = resolve_held_step(held, state=state, runtime=runtime, middleware=self)
        await trace_answer(held, answered=answered, middleware=self)
        return answered.update

    @override
    def after_agent(self, state: MonitorState, runtime: AgentRuntime) -> AgentStateUpdate | None:
        """Tag the notes written since the last step, and close the run, under `invoke()`.

        The ids it gives messages are marked as read in the monitors' trajectories.
        """
        return build_marked_run_end_update(state, task_author=self.task_author)

    @override
    async def aafter_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: MonitorState,
        runtime: AgentRuntime,
    ) -> AgentStateUpdate | None:
        """Tag the notes written since the last step, and close the run, under `ainvoke()`.

        The ids it gives messages are marked as read in the monitors' trajectories.
        """
        return build_marked_run_end_update(state, task_author=self.task_author)

    def send_scores(self, traced_step: TracedRun, *, step: Step, record: StepRecord) -> None:
        """Send the step's scores to the tracing tools its run is traced to."""
        send_step_scores(
            traced_step,
            decisions=step.decisions,
            flagged=record["flagged"],
            label=self.label,
            flag_threshold=self.flag_threshold,
            tracers=self.export_scores,
        )

    def build_run_start_update(self, state: MonitorState, *, runtime: object) -> AgentStateUpdate:
        """Return the update a run starts with, which adds the run's input to the trajectory.

        The trajectory reads the state as the run's own update leaves it, its
        ids given and its unconfirmed input tagged, before a summariser can
        remove the input. Input to a run paused at an audit stops the held
        step first, as `audits` says. A thread whose decided steps outlive
        the trajectory raises `MonitorError`, as `trajectories` says.
        """
        check_trajectory_kept(state, label=self.label, agent=self.agent_name)
        held = find_held_step(state, middleware=self)
        if held is not None:
            return stop_held_step(held, state=state, runtime=runtime, middleware=self)
        update = build_run_start_update(state)
        trajectories = build_run_start_trajectories_update(
            state,
            update=update,
            label=self.label,
            task_author=self.task_author,
            redaction=self.redaction,
        )
        return {**update, **trajectories}
