"""The middleware warns once when the agent's model is given tools its provider runs itself.

A provider runs its server-side tools inside the model call, before the
monitor judges the step, so the monitor cannot stop them. `create_agent`
passes every tool dictionary to the provider as a built-in tool and never
runs it itself [@langchain2026].
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_sync_monitors.errors import ProviderToolWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.provider_tools import is_provider_tool
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY
from tests.support.agents import RunMode, Workspace, build_keyword_monitor, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.protocols import AcceptFirst

WEB_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3}
WEB_SEARCH = {"type": "web_search"}
GOOGLE_SEARCH = {"google_search": {}}
FUNCTION_TOOL = {
    "type": "function",
    "function": {"name": "lookup", "description": "Look a word up.", "parameters": {}},
}
ANTHROPIC_CUSTOM_TOOL = {"name": "lookup", "description": "Look up.", "input_schema": {}}
FLAT_FUNCTION_TOOL = {"name": "lookup", "description": "Look a word up.", "parameters": {}}


class ProviderToolChatModel(ScriptedChatModel):
    """A scripted model that accepts provider tool dictionaries, as provider models do."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def build_agent(
    model: ScriptedChatModel,
    *,
    provider_tools: Sequence[dict[str, Any]],
    middleware: MonitorMiddleware | None = None,
) -> Runnable[Any, Any]:
    monitor = middleware or MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AcceptFirst(),
    )
    tools: list[BaseTool | dict[str, Any]] = [*Workspace().build_tools(), *provider_tools]
    return create_agent(model, tools=tools, middleware=[monitor])


def test_provider_tools_are_named_in_one_warning(run_mode: RunMode) -> None:
    # Arrange
    model = ProviderToolChatModel(responses=[AIMessage("Done.")])
    agent = build_agent(model, provider_tools=[WEB_FETCH, WEB_SEARCH, GOOGLE_SEARCH])

    # Act
    with pytest.warns(ProviderToolWarning) as caught:
        run_agent(agent, mode=run_mode)

    # Assert
    [warning] = [item for item in caught if item.category is ProviderToolWarning]
    text = str(warning.message)
    assert "monitor[main]" in text
    assert "web_fetch_20250910, web_search, google_search" in text
    assert "before the monitor judges the step" in text
    assert not warning.filename.startswith(LIBRARY_DIRECTORY)


def test_the_warning_is_shown_once_per_middleware_instance(run_mode: RunMode) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ProviderToolChatModel(responses=[AIMessage("First."), AIMessage("Second.")])
    agent = build_agent(model, provider_tools=[WEB_SEARCH], middleware=middleware)
    subagent_copy = middleware.copy_for_subagent(subagent_name="worker")
    subagent_model = ProviderToolChatModel(responses=[AIMessage("Worker done.")])
    subagent = build_agent(subagent_model, provider_tools=[WEB_SEARCH], middleware=subagent_copy)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)
        run_agent(agent, mode=run_mode)
        run_agent(subagent, mode=run_mode)

    # Assert: one for the instance, and one for its subagent copy
    provider_warnings = [item for item in caught if item.category is ProviderToolWarning]
    assert [str(item.message).split(":")[0] for item in provider_warnings] == [
        "monitor[main]",
        "monitor[worker]",
    ]


def test_an_agent_with_only_its_own_tools_gets_no_warning(run_mode: RunMode) -> None:
    # Arrange
    model = ProviderToolChatModel(responses=[AIMessage("Done.")])
    agent = build_agent(model, provider_tools=[FUNCTION_TOOL, ANTHROPIC_CUSTOM_TOOL])

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)

    # Assert
    assert [item for item in caught if item.category is ProviderToolWarning] == []


@pytest.mark.parametrize(
    ("tool", "is_provider"),
    [
        (WEB_FETCH, True),
        (WEB_SEARCH, True),
        (GOOGLE_SEARCH, True),
        ({"type": "mcp", "server_label": "docs", "server_url": "https://mcp.example"}, True),
        (FUNCTION_TOOL, False),
        ({"type": "custom", "name": "grep", "description": "Search files."}, False),
        (ANTHROPIC_CUSTOM_TOOL, False),
        (FLAT_FUNCTION_TOOL, False),
        ({"title": "Lookup", "type": "object", "properties": {}}, False),
    ],
    ids=[
        "anthropic-web-fetch",
        "openai-web-search",
        "gemini-google-search",
        "openai-remote-mcp",
        "openai-function",
        "custom",
        "anthropic-schema",
        "flat-function-schema",
        "json-schema",
    ],
)
def test_a_tool_dictionary_is_a_provider_tool_unless_it_defines_a_function(
    tool: dict[str, Any],
    is_provider: bool,
) -> None:
    # Act
    result = is_provider_tool(tool)

    # Assert
    assert result is is_provider
