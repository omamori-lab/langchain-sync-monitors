"""The provider tool warning names a tool bound on the model only when it reaches the model call.

`create_agent` binds the tools of an agent that has tools of its own, or a
`response_format`, with the model's `bind_tools`, which on a model bound in
advance binds the model underneath afresh and drops the tools bound before
[@langchain2026]. Each test checks that the warning agrees with what the model
received.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_sync_monitors.errors import ProviderToolWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, Workspace, build_keyword_monitor, run_agent
from tests.support.protocols import AcceptFirst

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
LOOKUP = {"type": "function", "function": {"name": "lookup", "parameters": {}}}


class Answer(BaseModel):
    text: str


def read_tool_label(tool: object) -> str:
    """Name a tool the model call received: a dictionary by its type, anything else by name."""
    if isinstance(tool, dict):
        return str(tool.get("type") or tool.get("name"))
    return str(getattr(tool, "name", type(tool).__name__))


class RecordingBindingChatModel(BaseChatModel):
    """Binds tools as provider chat models do, and records the tools each call receives."""

    received: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-binding"

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
        self.received.append([read_tool_label(tool) for tool in kwargs.get("tools", [])])
        return ChatResult(generations=[ChatGeneration(message=AIMessage('{"text": "Done."}'))])


def run_and_read_warnings(
    model: Runnable[Any, Any],
    *,
    mode: RunMode,
    **options: Any,
) -> list[str]:
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    # create_agent is typed for a chat model, but takes a model with tools bound at run time
    agent = create_agent(cast("BaseChatModel", model), middleware=[monitor], **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=mode)
    return [str(item.message) for item in caught if item.category is ProviderToolWarning]


def has_received(model: RecordingBindingChatModel, tool_label: str) -> bool:
    return any(tool_label in labels for labels in model.received)


def test_a_server_tool_bound_on_the_model_of_an_agent_without_tools_is_named(
    run_mode: RunMode,
) -> None:
    # Arrange
    base = RecordingBindingChatModel()

    # Act
    warned = run_and_read_warnings(base.bind_tools([WEB_SEARCH]), mode=run_mode)

    # Assert
    assert has_received(base, "web_search_20250305")
    assert len(warned) == 1
    assert "web_search_20250305" in warned[0]


@pytest.mark.parametrize("shape", ["own-tools", "provider-strategy"])
def test_a_server_tool_that_create_agent_binds_away_is_not_named(
    run_mode: RunMode,
    shape: str,
) -> None:
    # Arrange: tools of the agent's own, or a response format, make create_agent bind afresh
    base = RecordingBindingChatModel()
    options: dict[str, Any] = (
        {"tools": Workspace().build_tools()}
        if shape == "own-tools"
        else {"response_format": ProviderStrategy(schema=Answer)}
    )

    # Act
    warned = run_and_read_warnings(base.bind_tools([WEB_SEARCH]), mode=run_mode, **options)

    # Assert
    assert base.received
    assert not has_received(base, "web_search_20250305")
    assert warned == []


def test_a_server_tool_under_an_outer_binding_is_named(run_mode: RunMode) -> None:
    # Arrange: an outer binding that sets no tools passes the inner binding's through
    base = RecordingBindingChatModel()
    model = RunnableBinding(bound=base.bind_tools([WEB_SEARCH]), kwargs={"temperature": 0})

    # Act
    warned = run_and_read_warnings(model, mode=run_mode)

    # Assert
    assert has_received(base, "web_search_20250305")
    assert len(warned) == 1


def test_an_outer_binding_s_tools_replace_the_inner_ones(run_mode: RunMode) -> None:
    # Arrange: each binding passes its own keyword arguments over the inner one's
    base = RecordingBindingChatModel()
    model = RunnableBinding(bound=base.bind_tools([WEB_SEARCH]), kwargs={"tools": [LOOKUP]})

    # Act
    warned = run_and_read_warnings(model, mode=run_mode)

    # Assert
    assert has_received(base, "function")
    assert not has_received(base, "web_search_20250305")
    assert warned == []
