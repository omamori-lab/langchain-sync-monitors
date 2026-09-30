"""The tools a model provider runs itself, inside the model call.

LangChain passes every dictionary in an agent's tools to the provider as a
built-in tool and never runs it itself [@langchain2026]. Some of those the
provider runs on its own servers, inside the model call, before the monitor
judges the step and again for every sample a protocol draws, so the monitor
middleware can only warn about them:

- Anthropic's server tools web search, web fetch and code execution, and its
  MCP connector [@anthropic2026tooluse; @langchainanthropic2026];
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
nothing outside it.

Detection reads the tools of the model request and the tools bound on the
model before the agent was built, with `bind_tools`, as dictionaries. It
cannot see a server-side feature switched on in the model's own settings,
such as OpenRouter's web plugin or an `:online` model
[@langchainopenrouter2026], nor a tool an integration has already turned into
an object of its own SDK; those run the same way without a warning.
"""

from __future__ import annotations

import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Final

from langchain_sync_monitors._langchain import AgentModelRequest, read_bound_tools
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

warned_middleware_ids: Final[set[int]] = set()
"""The ids of the middleware instances that have shown their `ProviderToolWarning` already.

It holds ids, so it never hashes a middleware, which a subclass may make
unhashable, and it lives outside the middleware, which stays configuration
only and can be copied and pickled. A finaliser removes each id when its
instance is collected, so a later instance at the same address still warns.
"""

UNREFERENCEABLE_LIMIT: Final = 256
"""How many middleware instances that cannot be weak-referenced are kept alive at most."""

kept_unreferenceable: Final[OrderedDict[int, object]] = OrderedDict()
"""The warned instances that cannot be weak-referenced, kept alive while their ids count.

No finaliser can watch such an instance, so it is kept, which keeps its id its
own. Past `UNREFERENCEABLE_LIMIT` the oldest is released and its id dropped,
so it may warn again, but nothing grows without bound.
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


def mark_warned(middleware: object) -> None:
    """Record that a middleware instance has shown its warning, without hashing it."""
    key = id(middleware)
    warned_middleware_ids.add(key)
    try:
        weakref.finalize(middleware, warned_middleware_ids.discard, key)
    except TypeError:
        kept_unreferenceable[key] = middleware
        if len(kept_unreferenceable) > UNREFERENCEABLE_LIMIT:
            released_key, _ = kept_unreferenceable.popitem(last=False)
            warned_middleware_ids.discard(released_key)


def warn_about_provider_tools(
    request: AgentModelRequest,
    *,
    middleware: object,
    middleware_name: str,
) -> None:
    """Emit a `ProviderToolWarning` when a model request carries tools the provider runs itself.

    Each middleware instance warns once, at the first step whose request holds
    such tools, among its tools or those bound on its model. Two steps racing
    in parallel runs may both warn, which is harmless.
    """
    if id(middleware) in warned_middleware_ids:
        return
    provider_tools = find_provider_tools([*request.tools, *read_bound_tools(request.model)])
    if not provider_tools:
        return
    mark_warned(middleware)
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
