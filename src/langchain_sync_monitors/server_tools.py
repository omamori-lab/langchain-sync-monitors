"""Server-side tools, which a model provider runs inside the model call.

LangChain passes every dictionary in an agent's tools to the provider as it
is, and never runs it itself [@langchain2026]. Some of those are server
tools: the provider runs them on its own servers, inside the model call,
before the monitor judges the step and again for every sample a protocol
draws, so the monitor middleware can only warn about them:

- Anthropic's web search, web fetch and code execution, and its MCP
  connector [@anthropic2026tooluse; @langchainanthropic2026];
- OpenAI's web search, file search, code interpreter, image generation and
  remote MCP [@openai2026tools; @langchainopenai2026];
- Gemini's Google Search, Google Maps, code execution and URL context
  [@google2026geminitools; @langchaingooglegenai2026].

Every other tool dictionary is left out, since its calls come back to the
agent as tool calls, which the monitor judges before they run: a function's
schema, and the client tools a provider defines but the application runs,
such as Anthropic's bash, text editor, memory and computer use tools, which
langchain-anthropic's middleware hands the model as dictionaries, and
OpenAI's computer use and patch tools. So are the tool search of Anthropic
and of OpenAI, and Anthropic's advisor, which run at the provider but act on
nothing outside it. The tools of an MCP server the application connects
itself are not dictionaries but the agent's own tools, so they are not
server tools either; only a provider's MCP connector is.

Detection reads the tools of the model request. It also reads the tools
bound on the model with `bind_tools` before the agent was built, but only for
an agent with no tools of its own and no `response_format`: in every other
case `create_agent` binds the model's tools afresh, which drops the ones
bound before [@langchain2026]. It cannot see a server-side feature switched on
in the model's own settings, such as OpenRouter's web plugin or an `:online`
model [@langchainopenrouter2026], nor a tool an integration has already
turned into an object of its own SDK, nor tools bound inside a wrapper such
as `with_fallbacks(...)`, nor tools queued with `bind_tools` on a
configurable model from `init_chat_model(..., configurable_fields=...)`;
those run the same way without a warning.
"""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Mapping, Sequence
from typing import Final

from langchain_sync_monitors._langchain import (
    AgentModelRequest,
    AnyAgentMiddleware,
    read_bound_tools,
)
from langchain_sync_monitors.errors import ServerToolWarning
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY

SERVER_TOOL_TYPES: Final = frozenset(
    {"web_search", "file_search", "code_interpreter", "image_generation", "mcp"},
)
"""The `type` of each OpenAI server tool, as its Responses API names them."""

SERVER_TOOL_TYPE_PREFIXES: Final = ("web_search_", "web_fetch_", "code_execution_", "mcp_toolset")
"""The start of each versioned `type` of a server tool.

Anthropic dates its tool types, as in `web_search_20250305`, and OpenAI its
preview search, as in `web_search_preview_2025_03_11`.
"""

GEMINI_SERVER_TOOL_KEYS: Final = frozenset(
    {"google_search", "google_search_retrieval", "google_maps", "code_execution", "url_context"},
)
"""The keys by which langchain-google-genai recognises a Gemini server tool."""

warned_middleware_ids: Final[set[int]] = set()
"""The ids of the middleware instances that have shown their `ServerToolWarning` already.

It holds ids, so it never hashes a middleware, which a subclass may make
unhashable, and it lives outside the middleware, which keeps no mutable state
of its own. A finaliser removes each id when its instance is collected, so a
later instance at the same address still warns.
"""


def is_server_tool(tool: Mapping[str, object]) -> bool:
    """Tell whether a tool dictionary of a model request is a server tool."""
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        return tool_type in SERVER_TOOL_TYPES or tool_type.startswith(SERVER_TOOL_TYPE_PREFIXES)
    # A Gemini server tool has no `type`: the key it sits under names it.
    return bool(GEMINI_SERVER_TOOL_KEYS & tool.keys())


def describe_server_tool(tool: Mapping[str, object]) -> str:
    """Name a server tool by its `type`, or by its keys for a Gemini tool, as `google_search`."""
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        return tool_type
    return ", ".join(sorted(GEMINI_SERVER_TOOL_KEYS & tool.keys()))


def find_server_tools(tools: Sequence[object]) -> list[str]:
    """Name each server tool of a model request."""
    return [
        describe_server_tool(tool)
        for tool in tools
        if isinstance(tool, Mapping) and is_server_tool(tool)
    ]


def render_server_tool_warning(*, middleware_name: str, server_tools: Sequence[str]) -> str:
    """Explain that the provider runs these server tools before the monitor judges the step."""
    return (
        f"{middleware_name}: the agent's model is given server tools, which its provider "
        f"runs itself: {', '.join(server_tools)}. The provider runs them inside the model "
        "call: before the monitor judges the step, and again for every sample the protocol "
        "draws, so the monitor cannot stop them. Give the agent tools of its own for any "
        "action that must be judged before it runs."
    )


def mark_warned(middleware: AnyAgentMiddleware) -> None:
    """Record that a middleware instance has shown its warning, without hashing it.

    Every LangChain middleware can be weak-referenced, since `AgentMiddleware`
    keeps a `__weakref__` slot, so a finaliser can always drop the id.
    """
    key = id(middleware)
    warned_middleware_ids.add(key)
    weakref.finalize(middleware, warned_middleware_ids.discard, key)


def read_tools_reaching_the_model(request: AgentModelRequest) -> list[object]:
    """Return the tools the model call receives: the request's, or those bound on the model.

    `create_agent` binds the request's tools, and a `response_format`'s, with
    the model's `bind_tools`, which on a model bound in advance binds the
    model underneath afresh. Only an agent with neither keeps the tools bound
    before, since then the model is bound with its settings alone
    [@langchain2026].
    """
    if request.tools or request.response_format is not None:
        return list(request.tools)
    return read_bound_tools(request.model)


def warn_about_server_tools(
    request: AgentModelRequest,
    *,
    middleware: AnyAgentMiddleware,
    middleware_name: str,
) -> None:
    """Emit a `ServerToolWarning` when a model call receives server tools.

    Each middleware instance warns once, at the first step whose model call
    receives such tools. Two steps racing in parallel runs may both warn,
    which is harmless.
    """
    if id(middleware) in warned_middleware_ids:
        return
    server_tools = find_server_tools(read_tools_reaching_the_model(request))
    if not server_tools:
        return
    mark_warned(middleware)
    message = render_server_tool_warning(
        middleware_name=middleware_name,
        server_tools=server_tools,
    )
    warnings.warn(
        message,
        ServerToolWarning,
        stacklevel=2,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )
