"""Deep agents with one monitored worker subagent. Import only where `deepagents` is installed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message


def build_delegation_step(
    *,
    description: str = "Find the sources.",
    call_id: str = "call-task",
    subagent_type: str = "worker",
) -> AIMessage:
    return build_tool_call_message(
        tool_name="task",
        call_id=call_id,
        arguments={"description": description, "subagent_type": subagent_type},
    )


def build_parallel_delegation() -> AIMessage:
    """Script one step that hands `worker` and `reviewer` a task each, at once."""
    return AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="task",
                args={"description": "Find sources.", "subagent_type": "worker"},
                id="call-task-worker",
                type="tool_call",
            ),
            ToolCall(
                name="task",
                args={"description": "Check sources.", "subagent_type": "reviewer"},
                id="call-task-reviewer",
                type="tool_call",
            ),
        ],
    )


def read_monitor(spec: SubAgent) -> MonitorMiddleware:
    """Return the monitor `monitor_subagents` added to a spec, the last of its middleware."""
    monitor = spec.get("middleware", [])[-1]
    assert isinstance(monitor, MonitorMiddleware)
    return monitor


def build_deep_agent(
    *,
    main_model: ScriptedChatModel,
    worker_model: ScriptedChatModel,
    main_monitor: MonitorMiddleware,
    worker_monitor: MonitorMiddleware | None = None,
    checkpointer: InMemorySaver | None = None,
    tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph[Any, Any, Any, Any]:
    worker = SubAgent(name="worker", description="Finds sources.", model=worker_model)
    overrides = {"worker": worker_monitor} if worker_monitor else None
    subagents = monitor_subagents(middleware=main_monitor, subagents=[worker], overrides=overrides)
    return create_deep_agent(
        model=main_model,
        tools=list(tools),
        middleware=[main_monitor],
        subagents=subagents,
        checkpointer=checkpointer,
    )
