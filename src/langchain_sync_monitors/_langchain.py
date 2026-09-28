"""The boundary with the loosely typed surfaces of LangChain and LangGraph.

LangChain types a request's runtime context, its structured response, its
state and a stream writer's payload as `Any`. Those types are named here, once,
so every other module works with the library's own precise types.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, TypedDict, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, BaseMessage

from langchain_sync_monitors.contracts import StepRecord

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

MONITOR_LOG_KEY = "monitor_log"
"""The state key that holds the step records of every monitor in the run."""


class MonitorStepEvent(TypedDict):
    """The event a monitor writes to `stream_mode="custom"` once per committed step.

    It follows the typed events Deep Agents' `RubricMiddleware` writes to the
    same stream [@deepagents2026].
    """

    type: Literal["monitor_step"]
    record: StepRecord


def read_monitor_log(state: Mapping[str, object]) -> list[StepRecord]:
    """Return the step records in an agent state, or an empty list when there are none."""
    records = state.get(MONITOR_LOG_KEY)
    if not isinstance(records, list):
        return []
    return cast("list[StepRecord]", records)


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


def write_stream_event(request: AgentModelRequest, *, event: MonitorStepEvent) -> None:
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
