"""The boundary with the loosely typed surfaces of LangChain and LangGraph.

LangChain types a request's runtime context, its structured response, its
state, a hook's state update and a stream writer's payload as `Any`. Those
types are named here, once, so every other module works with the library's own
precise types. This is also the only module that touches LangChain's callback
managers, through which the monitor's spans reach every tracer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NotRequired, TypedDict, cast
from uuid import UUID

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.callbacks import AsyncCallbackManager, BaseCallbackManager, CallbackManager
from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableBinding, RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config, var_child_runnable_config
from langgraph.channels import binop as langgraph_binop
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite
from pydantic import TypeAdapter, ValidationError

from langchain_sync_monitors.contracts import Delegation, SampleRecord, StepRecord
from langchain_sync_monitors.errors import ConfigurationError, MonitorError

logger = logging.getLogger(__name__)

type TraceValue = str | int | float | bool | Sequence[TraceValue] | Mapping[str, TraceValue] | None
"""A plain value a monitor span carries in its inputs, outputs or metadata."""

type TraceValues = Mapping[str, TraceValue]
"""A span's inputs, outputs or metadata: plain values by name."""

type AgentContext = Any
"""The runtime context of the agent a monitor sits in, whatever its schema."""

type StructuredOutput = Any
"""The structured response of the agent a monitor sits in, whatever its schema."""

type AgentModelRequest = ModelRequest[AgentContext]
"""A model request of any agent, whatever its context schema."""

type AgentModelResponse = ModelResponse[StructuredOutput]
"""A model response of any agent, whatever its structured response schema."""

type ModelCallHandler = Callable[[AgentModelRequest], AgentModelResponse]
"""The callback LangChain passes to `wrap_model_call` to run the rest of the stack."""

type AsyncModelCallHandler = Callable[[AgentModelRequest], Awaitable[AgentModelResponse]]
"""The callback LangChain passes to `awrap_model_call` to run the rest of the stack."""

type AnyAgentMiddleware = AgentMiddleware[Any, AgentContext, StructuredOutput]
"""A middleware of any state, context and response schema, as `create_agent` accepts."""

type SubagentMiddleware = AgentMiddleware
"""A middleware as Deep Agents types a subagent's `middleware` list, with LangChain's defaults."""

type AgentRuntime = Runtime[AgentContext]
"""The runtime LangChain passes to a middleware's node hooks, whatever the context schema."""

type AgentStateUpdate = dict[str, Any]
"""A state update a middleware's node hook returns, which LangChain types by key only."""

type ToolCallResult = ToolMessage | Command[Any]
"""What a tool call returns to the agent: a tool message, or a command with any update."""

type ToolCallResults = ToolCallResult | list[ToolCallResult]
"""What a tool call can return at run time: LangGraph's tool node also accepts a list of
tool messages and commands from one tool [@langgraph2026], which LangChain's hook types
leave out."""

type ToolCallHandler = Callable[[ToolCallRequest], ToolCallResult]
"""The callback LangChain passes to `wrap_tool_call` to run the rest of the stack."""

type AsyncToolCallHandler = Callable[[ToolCallRequest], Awaitable[ToolCallResult]]
"""The callback LangChain passes to `awrap_tool_call` to run the rest of the stack."""


def cast_to_tool_call_result(results: ToolCallResults) -> ToolCallResult:
    """Return a tool call's result as LangChain's hook types declare it, a list included."""
    return cast("ToolCallResult", results)


MESSAGES_KEY = "messages"
"""The state key that holds an agent's conversation."""

type MessageRewrite = Callable[[BaseMessage], BaseMessage]
"""Returns the message to write in place of one a command's update writes."""

type UpdateValue = Any
"""A value a state update writes under one key, which LangGraph leaves untyped."""

type UpdatePairs = Sequence[tuple[str, UpdateValue]]
"""A state update as pairs of key and value, the form in which LangGraph writes one."""

type WrittenMessages = list[BaseMessage] | Overwrite
"""What an update writes to `messages` once rewritten: messages, or an `Overwrite` of them."""


OVERWRITE_KEY = "__overwrite__"
"""The key that marks the two dictionary forms of an `Overwrite`."""


def is_update_pairs(update: object) -> bool:
    """Tell whether an update is pairs of key and value, as LangGraph tells it."""
    return isinstance(update, list | tuple) and all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) for pair in update
    )


def read_update_pairs(command: Command[Any]) -> UpdatePairs:
    """Return the pairs of key and value a command's update writes, as LangGraph reads them.

    LangGraph accepts an update as a dict, as pairs, or as an object whose
    class annotates its keys, such as a dataclass or a pydantic model, and
    reads anything else as a value for a root channel. Its own reader,
    `Command._update_as_tuples`, is the one it writes the update with
    [@langgraph2026], so the monitor reads exactly what the graph writes.
    That reader is private, so it is looked up when called. Should a release
    remove it, a dict and pairs are still read here, and any other update
    raises `MonitorError`, so no message it writes goes unread.
    """
    reader = getattr(command, "_update_as_tuples", None)
    if callable(reader):
        return reader()
    update = command.update
    if update is None:
        return []
    if isinstance(update, dict):
        return list(update.items())
    if is_update_pairs(update):
        return update
    message = (
        f"The monitor cannot read what a {type(update).__name__} update writes with this "
        "LangGraph, so it refuses the tool's command. Return the update as a dict."
    )
    raise MonitorError(message)


def read_overwrite_forms(value: UpdateValue) -> tuple[bool, UpdateValue]:
    """Tell whether a value is an `Overwrite` in a form LangGraph reads, and return its value.

    The forms are the typed `Overwrite`, `{"__overwrite__": value}`, and
    `{"type": "__overwrite__", "value": value}`, which JSON leaves of the
    typed one, as LangGraph reads them [@langgraph2026].
    """
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict) and len(value) == 1 and OVERWRITE_KEY in value:
        return True, value[OVERWRITE_KEY]
    if isinstance(value, dict) and value.get("type") == OVERWRITE_KEY and "value" in value:
        return True, value["value"]
    return False, None


def read_overwrite(value: UpdateValue) -> tuple[bool, UpdateValue]:
    """Tell whether LangGraph reads a value as an `Overwrite`, and return what it writes.

    LangGraph's own reader is private, so it is looked up when called, and
    the forms `read_overwrite_forms` knows are read if a release removes it.
    A form only a later release reads is then taken for messages, which
    `convert_to_messages` refuses, so the tool call fails closed.
    """
    reader = getattr(langgraph_binop, "_get_overwrite", None)
    return reader(value) if callable(reader) else read_overwrite_forms(value)


def rewrite_messages_value(value: UpdateValue, *, rewrite: MessageRewrite) -> WrittenMessages:
    """Return what to write to `messages` in place of one value an update writes there.

    The value is converted to messages first, as LangGraph's message reducer
    converts one message or a list, given as messages, dictionaries, tuples
    or strings, and each message is rewritten. A value LangGraph reads as an
    `Overwrite`, in any of its forms, bypasses the reducer and replaces the
    conversation [@langgraph2026], so it stays an `Overwrite`, of the
    rewritten messages.
    """
    is_overwrite, overwritten = read_overwrite(value)
    written = overwritten if is_overwrite else value
    converted = convert_to_messages(written if isinstance(written, list) else [written])
    messages = [rewrite(message) for message in converted]
    return Overwrite(messages) if is_overwrite else messages


def rewrite_update_pairs(pairs: UpdatePairs, *, rewrite: MessageRewrite) -> UpdatePairs:
    """Return the pairs with every value written to `messages` rewritten.

    A key is compared with `==`, as LangGraph finds its channel, so a key
    that only its own `__ne__` sets apart is still read as `messages`. Each
    write is converted on its own, as the message reducer converts it, so a
    message given as a dictionary, a tuple or a string is a new message in
    every write, as it is in LangGraph. A message object written more than
    once, as by a dataclass that annotates `messages` in two of its classes,
    is rewritten once, so every write holds the same copy. The reducer gives
    a message without an id its id in place and keeps one id once
    [@langgraph2026], so it keeps that copy once, as it would the original.
    """
    rewrites: dict[int, tuple[BaseMessage, BaseMessage]] = {}

    def rewrite_once(message: BaseMessage) -> BaseMessage:
        # The original is kept with its copy, so its id is not reused while the pairs are read.
        if id(message) not in rewrites:
            rewrites[id(message)] = (message, rewrite(message))
        return rewrites[id(message)][1]

    return [
        (key, rewrite_messages_value(value, rewrite=rewrite_once) if key == MESSAGES_KEY else value)
        for key, value in pairs
    ]


def rewrite_update_messages(command: Command[Any], *, rewrite: MessageRewrite) -> Command[Any]:
    """Return the command with every value its update writes to `messages` rewritten.

    A dict stays a dict, the shape LangChain's and Deep Agents' middleware
    read. Any other shape LangGraph accepts, such as pairs, a dataclass or a
    pydantic model, becomes a tuple of the pairs LangGraph reads from it,
    with the messages rewritten, so the state receives the same writes and
    none of the update's own code runs again. A command whose update writes
    no messages is returned as it is. That includes an update LangGraph reads
    as a value for a root channel, which an agent's state does not have.
    """
    pairs = read_update_pairs(command)
    if not any(key == MESSAGES_KEY for key, _ in pairs):
        return command
    return replace_update_pairs(command, pairs=rewrite_update_pairs(pairs, rewrite=rewrite))


def replace_update_pairs(command: Command[Any], *, pairs: UpdatePairs) -> Command[Any]:
    """Return the command writing `pairs`: a dict update stays a dict, any other becomes pairs."""
    if isinstance(command.update, dict):
        return replace(command, update=dict(pairs))
    return replace(command, update=tuple(pairs))


MONITOR_LOG_KEY = "monitor_log"
"""The state key that holds the step records of every monitor in the run."""

MONITOR_DELEGATION_KEY = "monitor_delegation"
"""The state key through which a monitored agent hands a subagent its `Delegation`."""

DELEGATION_ADAPTER = TypeAdapter(Delegation)
"""Validates a `Delegation` read from the state, where an agent's input can also put one."""

span_labels: ContextVar[TraceValues | None] = ContextVar("monitor_span_labels", default=None)
"""The metadata keys every monitor span opened in the current step carries, if any."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceSpan:
    """A named run the monitor opens in the agent's trace.

    `tags` and `metadata` belong to this run alone: the model calls made inside
    it keep exactly the tags and metadata LangChain gives them. `run_id` is
    generated by LangChain unless it is given.
    """

    name: str
    inputs: TraceValues = field(default_factory=dict)
    metadata: TraceValues = field(default_factory=dict)
    tags: Sequence[str] = ()
    run_id: UUID | None = None


@dataclass(slots=True, kw_only=True)
class TracedRun:
    """What a traced block reports before its span ends.

    `outputs` end the span. `inputs_at_end`, when set, replace the inputs the
    span started with, for inputs that exist only once the block has run;
    LangSmith, Langfuse and `astream_events` all take inputs at the end
    [@langchain2026].
    """

    outputs: TraceValues = field(default_factory=dict)
    inputs_at_end: TraceValues | None = None

    def build_end_arguments(self) -> dict[str, dict[str, TraceValue]]:
        """Return the keyword arguments that end the span: the late inputs, when there are any."""
        return {} if self.inputs_at_end is None else {"inputs": dict(self.inputs_at_end)}


class MonitorStepEvent(TypedDict):
    """The event a monitor writes to `stream_mode="custom"` once per committed step.

    It follows the typed events Deep Agents' `RubricMiddleware` writes to the
    same stream [@deepagents2026].
    """

    type: Literal["monitor_step"]
    record: StepRecord


class MonitorStepFailedEvent(TypedDict):
    """The event a monitor writes to `stream_mode="custom"` when a step fails uncommitted.

    A call inside the step raised before the protocol decided, so no record
    reaches `monitor_log`. The event keeps what the monitor had judged by then:
    `samples` holds each judged sample, none of them executed, and `error`
    names the exception, which the middleware raises again after the event.
    """

    type: Literal["monitor_step_failed"]
    agent: str
    monitor: str
    step_number: int
    error: str
    samples: list[SampleRecord]
    delegation_id: NotRequired[str]


type MonitorStreamEvent = MonitorStepEvent | MonitorStepFailedEvent
"""Every event a monitor writes to `stream_mode="custom"`."""


def read_monitor_log(state: Mapping[str, object]) -> list[StepRecord]:
    """Return the step records in an agent state, or an empty list when there are none."""
    records = state.get(MONITOR_LOG_KEY)
    if not isinstance(records, list):
        return []
    return cast("list[StepRecord]", records)


def read_delegation(state: object) -> Delegation | None:
    """Return the delegation a subagent was started with, or None in an agent started directly.

    A tool request's state is untyped in LangChain, and may be something other
    than a mapping, which holds no delegation. The key is part of every
    monitored agent's input, so whoever invokes the agent can set it, and the
    value is validated before it is used. A value that is not a `Delegation`
    with non-negative block counts raises `ConfigurationError`. Ignoring it
    would not do: a subagent would count from its own log alone and so reset
    the thread's total, and a negative count would lift the total altogether.
    """
    if not isinstance(state, Mapping):
        return None
    value = state.get(MONITOR_DELEGATION_KEY)
    if value is None:
        return None
    try:
        # Strict, so a count given as a bool, a float or a string is refused, not converted.
        return DELEGATION_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        message = (
            f"{MONITOR_DELEGATION_KEY} must be a Delegation with non-negative block counts. "
            "Leave it out of an agent's input: the monitor sets it for each subagent it starts."
        )
        raise ConfigurationError(message) from error


def build_tool_request_with_delegation(
    request: ToolCallRequest,
    *,
    delegation: Delegation,
) -> ToolCallRequest:
    """Return a copy of the tool request whose state holds the delegation.

    A tool reads the state from its injected runtime, not from the request, so
    both are replaced [@langgraph2026]. Deep Agents' `task` tool passes that
    state, less a few keys, to the subagent it starts, and the subagent's
    state schema keeps the key out of its output, as Deep Agents does for its
    own forked-context flag [@deepagents2026]. A request whose state is not a
    mapping, or that runs outside a graph, is returned unchanged.
    """
    runtime = cast("ToolRuntime | None", request.runtime)
    if not isinstance(request.state, Mapping) or runtime is None:
        return request
    state = {**request.state, MONITOR_DELEGATION_KEY: delegation}
    return replace(request, state=state, runtime=replace(runtime, state=state))


def read_bound_tools(model: object) -> list[object]:
    """Return the tools a chat model was bound to before the agent was built.

    `bind_tools` returns a `RunnableBinding` that keeps the tools among its
    keyword arguments [@langchaincore2026]. A model can sit inside several
    bindings, and each passes its own keyword arguments over those of the one
    inside it, so the tools of the outermost binding that sets them are the
    ones the model receives. A model with no binding has none.
    """
    while isinstance(model, RunnableBinding):
        bound: object = model.kwargs.get("tools")
        if isinstance(bound, list):
            return list(bound)
        model = model.bound
    return []


def build_request_with_messages(
    request: AgentModelRequest,
    *,
    messages: Sequence[BaseMessage],
) -> AgentModelRequest:
    """Return a copy of the request that sends these messages to the model.

    LangChain types a request's messages as its `AnyMessage` union, which every
    concrete message the library builds belongs to.
    """
    return request.override(messages=cast("list[AnyMessage]", list(messages)))


def append_subagent_middleware(
    existing: Sequence[SubagentMiddleware],
    *,
    middleware: AnyAgentMiddleware,
) -> list[SubagentMiddleware]:
    """Return a subagent's middleware list with one more middleware at the end.

    Deep Agents declares the list with LangChain's default type parameters, and
    generic middleware types do not convert to one another, although every
    middleware fits the list at run time [@deepagents2026].
    """
    return [*existing, cast("SubagentMiddleware", middleware)]


def write_stream_event(request: AgentModelRequest, *, event: MonitorStreamEvent) -> None:
    """Write the event to `stream_mode="custom"`, if the request runs inside a graph.

    A request built outside a graph has no runtime, and so no writer. A writer
    that fails must change nothing else: a committed step stays committed, and
    a failed step's own error is still raised. So the writer's error is logged
    and the event is dropped.
    """
    writer = getattr(request.runtime, "stream_writer", None)
    if writer is None:
        return
    try:
        writer(event)
    except Exception:
        logger.exception("The stream writer failed on a monitor event; the event is dropped.")


@contextmanager
def hide_model_calls_from_message_stream() -> Iterator[None]:
    """Keep every model call made inside the block out of `stream_mode="messages"`.

    LangGraph's message stream skips a model call tagged `nostream`, and still
    streams the messages a node returns once the node finishes [@langgraph2026].
    The tag goes on the config that calls inside the block inherit, so it
    reaches the agent's samples, the trusted model's step and the monitor's own
    calls alike, including calls in tasks started inside the block. A call made
    with its own `tags` replaces the inherited ones and streams again.
    """
    config = var_child_runnable_config.get() or RunnableConfig()
    hidden_config: RunnableConfig = {**config, "tags": [*config.get("tags", []), TAG_NOSTREAM]}
    token = var_child_runnable_config.set(hidden_config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


@contextmanager
def label_monitor_spans(labels: TraceValues) -> Iterator[None]:
    """Give every span a monitor opens inside the block these metadata keys as well.

    The labels name the step, so a span deep inside a monitor, such as a
    decision model's request, can be traced back to its step. They sit in a
    context variable, which the tasks started inside the block copy, and not
    in the metadata LangChain passes down, so no model call inside the block
    carries them.
    """
    token = span_labels.set(labels)
    try:
        yield
    finally:
        span_labels.reset(token)


def build_span_manager[ManagerT: (CallbackManager, AsyncCallbackManager)](
    config: RunnableConfig,
    *,
    span: TraceSpan,
    manager_class: type[ManagerT],
) -> ManagerT | None:
    """Return the callback manager that starts the span, or None when no handler listens.

    Inside a graph node, the config holds the node's own callback manager, and
    the span's manager is built from it, so the span nests under the running
    node for every handler. `CallbackManager.configure` would build it too, but
    with LangSmith tracing on it re-parents the span under the LangSmith-only
    run in which `create_agent` wraps each middleware hook, a run that Langfuse
    and every other handler never see [@langchain2026; @langfuse2026].
    Outside a graph, where the callbacks are a list or nothing, the manager is
    configured from the config as LangChain configures any run.

    The span takes the handlers and parent of that manager, and the tags and
    metadata it passes on, as LangChain's `get_child` does; a tag meant for
    the node's own next run, such as `seq:step:1`, stays behind. The node's
    manager is never changed, and the span's own tags, metadata and labels are
    not inherited by the LangChain runs inside it, model calls included.
    LangSmith's `traceable` runs are the exception: LangSmith builds one
    opened inside the span from the span's run, and copies the span's
    metadata into it [@langsmithsdk2026]. So the hook run `create_agent`
    opens for a middleware inside the monitor, or a monitor's own `traceable`
    code, carries the step's `monitor_` labels in LangSmith. Only dropping
    the labels from the spans would prevent it.
    """
    callbacks = config.get("callbacks")
    source = (
        callbacks
        if isinstance(callbacks, BaseCallbackManager)
        else CallbackManager.configure(
            inheritable_callbacks=callbacks,
            inheritable_metadata=config.get("metadata"),
        )
    )
    if not source.handlers:
        return None
    callback_manager = manager_class(
        handlers=list(source.handlers),
        inheritable_handlers=list(source.inheritable_handlers),
        parent_run_id=source.parent_run_id,
        tags=list(source.inheritable_tags),
        inheritable_tags=list(source.inheritable_tags),
        metadata=dict(source.inheritable_metadata),
        inheritable_metadata=dict(source.inheritable_metadata),
    )
    # Not inherited, so the span's own tags, labels and metadata do not spread to the runs
    # nested in it.
    callback_manager.add_tags(list(span.tags), inherit=False)
    callback_manager.add_metadata({**(span_labels.get() or {}), **span.metadata}, inherit=False)
    return callback_manager


@contextmanager
def open_traced_run_sync(span: TraceSpan) -> Iterator[TracedRun]:
    """Open the span under the running node, for `invoke()`; model calls in the block nest in it.

    With no callback handler attached nothing is opened, and the block runs as
    it would without the span. Otherwise the span starts with its inputs, and
    the calls made inside the block become its children through the config
    they inherit, the way `hide_model_calls_from_message_stream` sets it. The
    span ends with the outputs the block reports, or with the exception that
    leaves the block, cancellation included, as LangChain ends a runnable's run
    [@langchain2026]. A tracer that fails is logged by LangChain and does not
    fail the step, unless its handler sets `raise_error`, as for any LangChain
    run.
    """
    config = ensure_config()
    callback_manager = build_span_manager(config, span=span, manager_class=CallbackManager)
    traced_run = TracedRun()
    if callback_manager is None:
        yield traced_run
        return
    run_manager = callback_manager.on_chain_start(
        None, dict(span.inputs), run_id=span.run_id, name=span.name
    )
    token = var_child_runnable_config.set(patch_config(config, callbacks=run_manager.get_child()))
    try:
        yield traced_run
    except BaseException as error:
        run_manager.on_chain_error(error, **traced_run.build_end_arguments())
        raise
    else:
        run_manager.on_chain_end(dict(traced_run.outputs), **traced_run.build_end_arguments())
    finally:
        var_child_runnable_config.reset(token)


@asynccontextmanager
async def open_traced_run(span: TraceSpan) -> AsyncIterator[TracedRun]:
    """Open the span under the running node, for `ainvoke()`; model calls in the block nest in it.

    It behaves as `open_traced_run_sync`, through LangChain's async callback
    manager, whose handlers must not run on another event loop. The span ends
    even when the task is cancelled, since LangChain shields the callbacks
    that end a run [@langchain2026].
    """
    config = ensure_config()
    callback_manager = build_span_manager(config, span=span, manager_class=AsyncCallbackManager)
    traced_run = TracedRun()
    if callback_manager is None:
        yield traced_run
        return
    run_manager = await callback_manager.on_chain_start(
        None, dict(span.inputs), run_id=span.run_id, name=span.name
    )
    token = var_child_runnable_config.set(patch_config(config, callbacks=run_manager.get_child()))
    try:
        yield traced_run
    except BaseException as error:
        await run_manager.on_chain_error(error, **traced_run.build_end_arguments())
        raise
    else:
        await run_manager.on_chain_end(dict(traced_run.outputs), **traced_run.build_end_arguments())
    finally:
        var_child_runnable_config.reset(token)
