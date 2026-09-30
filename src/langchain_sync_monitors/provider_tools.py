"""The tools a model provider runs itself, inside the model call.

LangChain passes every dictionary in an agent's tools to the provider as a
built-in tool and never runs it itself [@langchain2026]. Some of those the
provider runs on its own servers, inside the model call, before the monitor
judges the step and again for every sample a protocol draws, so the monitor
middleware can only warn about them:

- Anthropic's server tools web search, web fetch and code execution, and its
  MCP connector [@anthropic2026tooluse; @langchainanthropic2026];
- OpenAI's web search, file search, code interpreter, image generation and
  remote MCP [@openai2026tools; @langchaincore2026];
- Gemini's Google Search, Google Maps, code execution and URL context
  [@google2026geminitools; @langchaingooglegenai2026].

Every other tool dictionary is left out, since its calls come back to the
agent as tool calls, which the monitor judges before they run: a function's
schema, and the client tools a provider defines but the application runs,
such as Anthropic's bash, text editor, memory and computer use tools, which
langchain-anthropic's middleware hands the model as dictionaries. So are
Anthropic's tool search and advisor, which run at the provider but act on
nothing outside it. Detection reads the request's tools only: a server-side
feature switched on in the model's own settings, such as OpenRouter's web
plugin or an `:online` model [@langchainopenrouter2026], runs the same way
without a warning.
"""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Mapping, Sequence
from typing import Final

from langchain_sync_monitors.errors import ProviderToolWarning
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY

SERVER_TOOL_TYPES: Final = frozenset(
    {"web_search", "file_search", "code_interpreter", "image_generation", "mcp"},
)
"""The `type` of each tool OpenAI runs itself, as its Responses API names them."""

SERVER_TOOL_TYPE_PREFIXES: Final = ("web_search_", "web_fetch_", "code_execution_", "mcp_toolset")
"""The start of each versioned `type` a provider runs itself.

Anthropic dates its tool types, as in `web_search_20250305`, and OpenAI its
preview search, as in `web_search_preview_2025_03_11`.
"""

GEMINI_SERVER_TOOL_KEYS: Final = frozenset(
    {"google_search", "google_search_retrieval", "google_maps", "code_execution", "url_context"},
)
"""The keys by which langchain-google-genai recognises a Gemini tool that Google runs itself."""

warned_middleware: Final = weakref.WeakSet[object]()
"""The middleware instances that have shown their `ProviderToolWarning` already.

The set holds weak references, so it keeps no middleware alive, and it lives
outside the middleware, which stays configuration only and can be copied and
pickled.
"""


def is_provider_tool(tool: Mapping[str, object]) -> bool:
    """Tell whether a tool dictionary of a model request is one the model provider runs itself."""
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        return tool_type in SERVER_TOOL_TYPES or tool_type.startswith(SERVER_TOOL_TYPE_PREFIXES)
    return bool(GEMINI_SERVER_TOOL_KEYS & tool.keys())


def describe_provider_tool(tool: Mapping[str, object]) -> str:
    """Name a provider tool by its `type`, or by its keys for a Gemini tool, as `google_search`."""
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        return tool_type
    return ", ".join(sorted(GEMINI_SERVER_TOOL_KEYS & tool.keys()))


def find_provider_tools(tools: Sequence[object]) -> list[str]:
    """Name each tool of a model request that the model provider runs itself."""
    return [
        describe_provider_tool(tool)
        for tool in tools
        if isinstance(tool, Mapping) and is_provider_tool(tool)
    ]


def render_provider_tool_warning(*, middleware_name: str, provider_tools: Sequence[str]) -> str:
    """Explain that the provider runs these tools before the monitor judges the step."""
    return (
        f"{middleware_name}: the agent's model is given tools its provider runs itself: "
        f"{', '.join(provider_tools)}. The provider runs them inside the model call: before "
        "the monitor judges the step, and again for every sample the protocol draws, so the "
        "monitor cannot stop them. Give the agent tools of its own for any action that must "
        "be judged before it runs."
    )


def warn_about_provider_tools(
    tools: Sequence[object],
    *,
    middleware: object,
    middleware_name: str,
) -> None:
    """Emit a `ProviderToolWarning` when a request's tools include ones the provider runs itself.

    Each middleware instance warns once, at the first step whose request holds
    such tools. Two steps racing in parallel runs may both warn, which is
    harmless.
    """
    if middleware in warned_middleware:
        return
    provider_tools = find_provider_tools(tools)
    if not provider_tools:
        return
    warned_middleware.add(middleware)
    message = render_provider_tool_warning(
        middleware_name=middleware_name,
        provider_tools=provider_tools,
    )
    warnings.warn(
        message,
        ProviderToolWarning,
        stacklevel=2,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )
