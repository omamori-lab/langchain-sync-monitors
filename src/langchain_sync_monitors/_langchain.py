"""The boundary with the loosely typed surfaces of LangChain and LangGraph.

LangChain types a request's runtime context, its structured response, its
state, a hook's state update and a stream writer's payload as `Any`. Those
types are named here, once, so every other module works with the library's own
precise types.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_sync_monitors.contracts import Delegation, SampleRecord, StepRecord

logger = logging.getLogger(__name__)

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

type ToolCallHandler = Callable[[ToolCallRequest], ToolCallResult]
"""The callback LangChain passes to `wrap_tool_call` to run the rest of the stack."""

type AsyncToolCallHandler = Callable[[ToolCallRequest], Awaitable[ToolCallResult]]
"""The callback LangChain passes to `awrap_tool_call` to run the rest of the stack."""

MONITOR_LOG_KEY = "monitor_log"
"""The state key that holds the step records of every monitor in the run."""

MONITOR_DELEGATION_KEY = "monitor_delegation"
"""The state key through which a monitored agent hands a subagent its `Delegation`."""


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
    than a mapping, which holds no delegation.
    """
    if not isinstance(state, Mapping):
        return None
    delegation = state.get(MONITOR_DELEGATION_KEY)
    if not isinstance(delegation, Mapping):
        return None
    return cast("Delegation", delegation)


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
    that fails must not undo a step that is already decided, so its error is
    logged and dropped.
    """
    writer = getattr(request.runtime, "stream_writer", None)
    if writer is None:
        return
    try:
        writer(event)
    except Exception:
        logger.exception("The stream writer failed on a monitor step event; the step stands.")


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
