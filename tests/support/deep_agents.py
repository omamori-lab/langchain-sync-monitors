"""Deep agents with one monitored worker subagent. Import only where `deepagents` is installed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message


def build_delegation_step(*, description: str = "Find the sources.") -> AIMessage:
    return build_tool_call_message(
        tool_name="task",
        call_id="call-task",
        arguments={"description": description, "subagent_type": "worker"},
    )


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
