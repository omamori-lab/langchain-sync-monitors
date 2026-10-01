"""The server tool warning marks each middleware once, without hashing it.

A user's subclass declared as a dataclass with the default `eq=True` hashes its
fields, so a middleware whose monitor is a plain dataclass is unhashable. The
warning marks a middleware by its id instead, and forgets the id when the
instance is collected.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_sync_monitors.errors import ServerToolWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.server_tools import warned_middleware_ids
from tests.support.agents import (
    RunMode,
    Workspace,
    build_keyword_monitor,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.protocols import AcceptFirst

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}


@dataclass(frozen=True, kw_only=True)
class TeamMonitorMiddleware(MonitorMiddleware):
    """A user's subclass declared as a dataclass with the default `eq=True`."""

    team: str = "platform"


class ServerToolChatModel(ScriptedChatModel):
    """A scripted model that accepts server tool dictionaries, as provider models do."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class SecondCallToolMiddleware(AgentMiddleware[Any, Any, Any]):
    """Adds a server tool to the model request from the second model call on."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self.add_tool(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> ModelResponse[Any]:
        return await handler(self.add_tool(request))

    def add_tool(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        if len(request.messages) < 2:
            return request
        return request.override(tools=[*request.tools, WEB_SEARCH])


def read_server_tool_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught if item.category is ServerToolWarning]


@pytest.mark.parametrize("tools", [[], [WEB_SEARCH]], ids=["own-tools", "server-tool"])
def test_an_unhashable_middleware_subclass_runs_and_warns_once(
    run_mode: RunMode,
    tools: list[dict[str, Any]],
) -> None:
    # Arrange: the tests' keyword monitor is a plain dataclass, so the subclass is unhashable
    middleware = TeamMonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ServerToolChatModel(responses=[AIMessage("First."), AIMessage("Second.")])
    agent = create_agent(model, tools=tools, middleware=[middleware])

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = run_agent(agent, mode=run_mode)
        run_agent(agent, mode=run_mode)

    # Assert
    assert [record["outcome"] for record in first["monitor_log"]] == ["allowed"]
    assert len(read_server_tool_warnings(caught)) == len(tools)


def test_a_collected_middleware_s_mark_is_forgotten(run_mode: RunMode) -> None:
    # Arrange: a middleware that warned, so its id is marked
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ServerToolChatModel(responses=[AIMessage("First.")])
    agent = create_agent(model, tools=[WEB_SEARCH], middleware=[middleware])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ServerToolWarning)
        run_agent(agent, mode=run_mode)
    key = id(middleware)
    assert key in warned_middleware_ids

    # Act: a later middleware may reuse the id once this one is collected
    del agent, middleware
    gc.collect()

    # Assert
    assert key not in warned_middleware_ids


def test_a_middleware_whose_first_request_has_no_server_tool_warns_at_the_next(
    run_mode: RunMode,
) -> None:
    # Arrange: the server tool joins the request only from the second model call on
    workspace = Workspace()
    model = ServerToolChatModel(responses=[build_read_step(), AIMessage("Done.")])
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    stack: list[AgentMiddleware[Any, Any, Any]] = [SecondCallToolMiddleware(), monitor]
    agent = create_agent(model, tools=workspace.build_tools(), middleware=stack)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)

    # Assert
    assert len(read_server_tool_warnings(caught)) == 1
