"""The provider tool warning marks each middleware once, without hashing it or growing unbounded.

A user's subclass declared as a dataclass with the default `eq=True` hashes its
fields, so a middleware whose monitor is a plain dataclass is unhashable. The
warning marks a middleware by its id instead, and forgets the id when the
instance is collected.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_sync_monitors.errors import ProviderToolWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.provider_tools import (
    UNREFERENCEABLE_LIMIT,
    kept_unreferenceable,
    warn_about_provider_tools,
    warned_middleware_ids,
)
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


class Unreferenceable:
    """An object that cannot be weak-referenced, as a class with `__slots__` and no weakref slot."""

    __slots__ = ()


class ProviderToolChatModel(ScriptedChatModel):
    """A scripted model that accepts provider tool dictionaries, as provider models do."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class BindingChatModel(BaseChatModel):
    """Binds tools as provider chat models do, into a `RunnableBinding`."""

    @property
    def _llm_type(self) -> str:
        return "binding"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("Done."))])


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


def read_provider_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught if item.category is ProviderToolWarning]


@pytest.mark.parametrize("tools", [[], [WEB_SEARCH]], ids=["own-tools", "server-tool"])
def test_an_unhashable_middleware_subclass_runs_and_warns_once(
    run_mode: RunMode,
    tools: list[dict[str, Any]],
) -> None:
    # Arrange: the tests' keyword monitor is a plain dataclass, so the subclass is unhashable
    middleware = TeamMonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ProviderToolChatModel(responses=[AIMessage("First."), AIMessage("Second.")])
    agent = create_agent(model, tools=tools, middleware=[middleware])

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = run_agent(agent, mode=run_mode)
        run_agent(agent, mode=run_mode)

    # Assert
    assert [record["outcome"] for record in first["monitor_log"]] == ["allowed"]
    assert len(read_provider_warnings(caught)) == len(tools)


def test_an_instance_that_cannot_be_weak_referenced_is_marked_without_unbounded_growth() -> None:
    # Arrange
    request = ModelRequest(
        model=ProviderToolChatModel(responses=[]), messages=[], tools=[WEB_SEARCH]
    )
    first = Unreferenceable()
    others = [Unreferenceable() for _ in range(UNREFERENCEABLE_LIMIT + 5)]

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_about_provider_tools(request, middleware=first, middleware_name="first")
        warn_about_provider_tools(request, middleware=first, middleware_name="first")
        for index, other in enumerate(others):
            warn_about_provider_tools(request, middleware=other, middleware_name=f"other-{index}")

    # Assert: the first warned once, and the kept instances stay within the limit
    names = [message.split(":")[0] for message in read_provider_warnings(caught)]
    assert names.count("first") == 1
    assert len(kept_unreferenceable) <= UNREFERENCEABLE_LIMIT
    assert id(others[-1]) in warned_middleware_ids
    assert id(first) not in warned_middleware_ids, "released past the limit, its id is free"


def test_a_middleware_whose_first_request_has_no_server_tool_warns_at_the_next(
    run_mode: RunMode,
) -> None:
    # Arrange: the server tool joins the request only from the second model call on
    workspace = Workspace()
    model = ProviderToolChatModel(responses=[build_read_step(), AIMessage("Done.")])
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    stack: list[AgentMiddleware[Any, Any, Any]] = [SecondCallToolMiddleware(), monitor]
    agent = create_agent(model, tools=workspace.build_tools(), middleware=stack)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)

    # Assert
    assert len(read_provider_warnings(caught)) == 1


def test_a_server_tool_bound_on_the_model_before_the_agent_is_named(run_mode: RunMode) -> None:
    # Arrange
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    # create_agent is typed for a chat model, but takes a model with tools bound at run time
    bound_model = cast("BaseChatModel", BindingChatModel().bind_tools([WEB_SEARCH]))
    agent = create_agent(bound_model, middleware=[monitor])

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)

    # Assert
    [message] = read_provider_warnings(caught)
    assert "web_search_20250305" in message
