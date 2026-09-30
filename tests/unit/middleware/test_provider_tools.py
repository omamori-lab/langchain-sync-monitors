"""The middleware warns once when the agent's model is given tools its provider runs itself.

A provider runs its server-side tools inside the model call, before the
monitor judges the step, so the monitor cannot stop them. Client tools a
provider defines, such as Anthropic's text editor, come back as tool calls
that the agent runs and the monitor judges, so they are not warned about.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_sync_monitors.contracts import ControlProtocol
from langchain_sync_monitors.errors import ProviderToolWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode, DeferToResample, HaltRun
from langchain_sync_monitors.provider_tools import is_provider_tool
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY
from tests.support.agents import RunMode, Workspace, build_keyword_monitor, run_agent
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
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
ANTHROPIC_TEXT_EDITOR = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
ANTHROPIC_BASH = {"type": "bash_20250124", "name": "bash"}


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


def test_an_agent_with_only_tools_it_runs_itself_gets_no_warning(run_mode: RunMode) -> None:
    # Arrange
    model = ProviderToolChatModel(responses=[AIMessage("Done.")])
    agent = build_agent(
        model,
        provider_tools=[
            FUNCTION_TOOL,
            ANTHROPIC_CUSTOM_TOOL,
            ANTHROPIC_TEXT_EDITOR,
            ANTHROPIC_BASH,
        ],
    )

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)

    # Assert
    assert [item for item in caught if item.category is ProviderToolWarning] == []


def test_the_editor_that_langchain_anthropic_hands_the_model_gets_no_warning(
    run_mode: RunMode,
) -> None:
    # Arrange
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic.middleware import StateClaudeTextEditorMiddleware

    editor = StateClaudeTextEditorMiddleware()
    view_call = build_tool_call_message(
        tool_name=editor.tool_name,
        call_id="call-view",
        arguments={"command": "view", "path": "/notes.md"},
    )
    model = ProviderToolChatModel(responses=[view_call, AIMessage("Done.")])
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    stack: list[AgentMiddleware[Any, Any, Any]] = [editor, monitor]
    agent = create_agent(model, middleware=stack)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_agent(agent, mode=run_mode)

    # Assert: the tool node ran the editor's call, which the monitor judged first
    assert [item for item in caught if item.category is ProviderToolWarning] == []
    [first_sample] = result["monitor_log"][0]["samples"]
    assert editor.tool_name in first_sample["proposal"]
    assert any(isinstance(message, ToolMessage) for message in result["messages"])


@pytest.mark.parametrize(
    "protocol",
    [
        AcceptFirst(),
        AutoMode(block_threshold=0.6),
        DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=0.9),
    ],
    ids=["accept-first", "auto-mode", "defer-to-resample"],
)
def test_a_middleware_that_has_warned_can_still_be_copied_and_pickled(
    run_mode: RunMode,
    protocol: ControlProtocol,
) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    model = ProviderToolChatModel(responses=[AIMessage("Done.")])
    with pytest.warns(ProviderToolWarning):
        run_agent(
            build_agent(model, provider_tools=[WEB_SEARCH], middleware=middleware), mode=run_mode
        )

    # Act
    deep_copy = copy.deepcopy(middleware)
    pickled = pickle.dumps(middleware)
    unpickled = pickle.loads(pickled)  # lanorme: ignore[DESERIAL-001] bytes pickled just above

    # Assert
    for duplicate in (deep_copy, unpickled):
        assert duplicate is not middleware
        assert duplicate.name == middleware.name
        assert type(duplicate.protocol) is type(protocol)


def test_a_copy_warns_once_of_its_own_and_the_original_stays_quiet(run_mode: RunMode) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    first_model = ProviderToolChatModel(responses=[AIMessage("First.")])
    with pytest.warns(ProviderToolWarning):
        run_agent(
            build_agent(first_model, provider_tools=[WEB_SEARCH], middleware=middleware),
            mode=run_mode,
        )
    duplicate = copy.deepcopy(middleware)
    original_model = ProviderToolChatModel(responses=[AIMessage("Again.")])
    copy_model = ProviderToolChatModel(responses=[AIMessage("Copy."), AIMessage("Copy again.")])
    original = build_agent(original_model, provider_tools=[WEB_SEARCH], middleware=middleware)
    copied = build_agent(copy_model, provider_tools=[WEB_SEARCH], middleware=duplicate)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(original, mode=run_mode)
        run_agent(copied, mode=run_mode)
        run_agent(copied, mode=run_mode)

    # Assert
    assert len([item for item in caught if item.category is ProviderToolWarning]) == 1


@pytest.mark.parametrize(
    ("tool", "is_provider"),
    [
        (WEB_FETCH, True),
        ({"type": "web_search_20250305", "name": "web_search"}, True),
        ({"type": "code_execution_20250825", "name": "code_execution"}, True),
        ({"type": "mcp_toolset", "mcp_server_name": "docs"}, True),
        (WEB_SEARCH, True),
        ({"type": "web_search_preview_2025_03_11"}, True),
        ({"type": "file_search", "vector_store_ids": ["store"]}, True),
        ({"type": "code_interpreter", "container": {"type": "auto"}}, True),
        ({"type": "image_generation"}, True),
        ({"type": "mcp", "server_label": "docs", "server_url": "https://mcp.example"}, True),
        (GOOGLE_SEARCH, True),
        ({"google_search_retrieval": {}}, True),
        ({"google_maps": {}}, True),
        ({"code_execution": {}}, True),
        ({"url_context": {}}, True),
        (ANTHROPIC_TEXT_EDITOR, False),
        (ANTHROPIC_BASH, False),
        ({"type": "memory_20250818", "name": "memory"}, False),
        ({"type": "computer_20250124", "name": "computer", "display_width_px": 1024}, False),
        ({"type": "tool_search_tool_regex_20251119", "name": "tool_search"}, False),
        ({"type": "computer_use_preview", "display_width": 1024}, False),
        ({"type": "local_shell"}, False),
        ({"type": "shell"}, False),
        ({"type": "apply_patch"}, False),
        ({"type": "tool_search"}, False),
        ({"type": "computer"}, False),
        ({"computer_use": {}}, False),
        (FUNCTION_TOOL, False),
        ({"type": "custom", "name": "grep", "description": "Search files."}, False),
        (ANTHROPIC_CUSTOM_TOOL, False),
        (FLAT_FUNCTION_TOOL, False),
        ({"title": "Lookup", "type": "object", "properties": {}}, False),
    ],
    ids=[
        "anthropic-web-fetch",
        "anthropic-web-search",
        "anthropic-code-execution",
        "anthropic-mcp-connector",
        "openai-web-search",
        "openai-web-search-preview",
        "openai-file-search",
        "openai-code-interpreter",
        "openai-image-generation",
        "openai-remote-mcp",
        "gemini-google-search",
        "gemini-google-search-retrieval",
        "gemini-google-maps",
        "gemini-code-execution",
        "gemini-url-context",
        "anthropic-text-editor",
        "anthropic-bash",
        "anthropic-memory",
        "anthropic-computer-use",
        "anthropic-tool-search",
        "openai-computer-use",
        "openai-local-shell",
        "openai-shell",
        "openai-apply-patch",
        "openai-tool-search",
        "openai-computer",
        "gemini-computer-use",
        "openai-function",
        "custom",
        "anthropic-schema",
        "flat-function-schema",
        "json-schema",
    ],
)
def test_only_the_server_tools_a_provider_runs_itself_are_provider_tools(
    tool: dict[str, Any],
    is_provider: bool,
) -> None:
    # Act
    result = is_provider_tool(tool)

    # Assert
    assert result is is_provider
