"""The tools a model provider runs itself, inside the model call.

LangChain passes every dictionary in an agent's tools to the provider as a
built-in tool and never runs it itself [@langchain2026]. A provider runs its
server-side tools, such as Anthropic's `web_fetch` or OpenAI's `web_search`,
inside the model call, before the monitor judges the step, so the monitor
middleware can only warn about them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FUNCTION_TOOL_TYPES = frozenset({"function", "custom"})
"""The `type` of a tool dictionary that defines a function the agent's own code answers."""

FUNCTION_SCHEMA_KEYS = frozenset({"function", "parameters", "input_schema", "properties"})
"""The keys of a tool dictionary that holds a function's schema rather than a provider tool."""


def is_provider_tool(tool: Mapping[str, object]) -> bool:
    """Tell whether a tool dictionary of a model request is one the model provider runs itself.

    LangChain passes every dictionary in an agent's tools to the provider as a
    built-in tool and never runs it itself [@langchain2026]. A dictionary that
    defines a function, by its `type` or by a schema key, is not one: its
    calls come back as tool calls, which the monitor judges.
    """
    tool_type = tool.get("type")
    return tool_type not in FUNCTION_TOOL_TYPES and not FUNCTION_SCHEMA_KEYS & tool.keys()


def describe_provider_tool(tool: Mapping[str, object]) -> str:
    """Name a provider tool by its `type`, else its `name`, else its keys, as `google_search`."""
    for key in ("type", "name"):
        value = tool.get(key)
        if isinstance(value, str):
            return value
    return ", ".join(sorted(tool))


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
        f"{', '.join(provider_tools)}. A provider runs its server-side tools, such as "
        "Anthropic's web_fetch or OpenAI's web_search, inside the model call: before the "
        "monitor judges the step, and again for every sample the protocol draws, so the "
        "monitor cannot stop them. Give the agent tools of its own for any action that must "
        "be judged before it runs."
    )
