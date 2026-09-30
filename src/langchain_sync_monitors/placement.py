"""Checks where middleware sits around a monitor in a `create_agent` list.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, so a
middleware after the monitor runs inside it, once per sample the protocol
draws, and one before it wraps the whole monitored step [@langchain2026].
Tool calls are wrapped in the same order. `check_monitor_placement` names the
middleware in a list that undermines what the monitor records.
"""

import warnings
from collections.abc import Sequence

from langchain.agents.middleware.types import AgentMiddleware

from langchain_sync_monitors._langchain import AnyAgentMiddleware
from langchain_sync_monitors.middleware import MonitorMiddleware

REQUEST_ONLY_MIDDLEWARE = frozenset(
    {
        "AnthropicPromptCachingMiddleware",
        "BedrockPromptCachingMiddleware",
        "FireworksPromptCachingMiddleware",
        "MemoryMiddleware",
        "UnsupportedContentMiddleware",
        "_ToolExclusionMiddleware",
    },
)
"""Classes that only rewrite the request, so they are safe inside a monitor.

Deep Agents places these after user middleware [@deepagents2026].
"""

RETRYING_MIDDLEWARE = frozenset(
    {
        "ModelFallbackMiddleware",
        "ModelRetryMiddleware",
        "_DeepAgentsSummarizationMiddleware",
    },
)
"""Classes that call the rest of the stack again when a model call raises.

LangChain's retry and fallback middleware retry on an exception, and Deep
Agents' summarisation retries after a context overflow
[@langchain2026; @deepagents2026]. Outside a monitor, each retry runs the
whole monitored step again. Subclasses count too.
"""

TOOL_FAILURE_HANDLING_MIDDLEWARE = frozenset({"ToolErrorMiddleware", "ToolRetryMiddleware"})
"""Classes that run a failed tool call again or answer it with an error message.

LangChain's tool retry middleware calls a tool again when it raises and, by
default, answers the call with an error message once the retries run out; its
tool error middleware answers the failures its handler chooses to
[@langchain2026]. Wherever they sit in the list, they wrap every tool call,
Deep Agents' `task` tool included, unless their `tools` argument leaves it out,
and a subagent whose run raises returns no records to its parent. Subclasses
count too.
"""


class MonitorPlacementWarning(UserWarning):
    """A middleware placed around or inside a monitor undermines what the monitor records.

    Inside a monitor, a middleware can return state updates for samples the
    monitor rejects. Outside it, a middleware that retries failed model calls
    runs the whole step again, and the samples judged before the failure never
    reach `monitor_log`. Anywhere in the list, a middleware that runs failed
    tool calls again or answers them lets a subagent's run fail without its
    blocks reaching Auto Mode's thread total.
    """


def is_model_call_wrapper(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls, as `create_agent` decides it."""
    middleware_class = type(middleware)
    return (
        middleware_class.wrap_model_call is not AgentMiddleware.wrap_model_call
        or middleware_class.awrap_model_call is not AgentMiddleware.awrap_model_call
    )


def is_unsafe_inside_monitor(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls and is not known to only rewrite the request."""
    # Only the class's own name counts, not its ancestors': a subclass may return updates the
    # original never does, so it is warned about. The retry checks read every ancestor, since a
    # subclass still retries. Both lean towards a warning.
    is_request_only = type(middleware).__name__ in REQUEST_ONLY_MIDDLEWARE
    return is_model_call_wrapper(middleware) and not is_request_only


def read_class_names(middleware: AnyAgentMiddleware) -> set[str]:
    """Return the names of a middleware's class and of every class it inherits from."""
    return {middleware_class.__name__ for middleware_class in type(middleware).__mro__}


def is_retrying_middleware(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps model calls and is, or subclasses, one that retries them."""
    is_retrying = bool(read_class_names(middleware) & RETRYING_MIDDLEWARE)
    return is_retrying and is_model_call_wrapper(middleware)


def is_tool_call_wrapper(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps tool calls, as `create_agent` decides it."""
    middleware_class = type(middleware)
    return (
        middleware_class.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or middleware_class.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    )


def is_tool_failure_handling_middleware(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware wraps tool calls and is, or subclasses, one known to retry them."""
    is_handling = bool(read_class_names(middleware) & TOOL_FAILURE_HANDLING_MIDDLEWARE)
    return is_handling and is_tool_call_wrapper(middleware)


def find_last_monitor_position(middleware: Sequence[AnyAgentMiddleware]) -> int | None:
    """Return the position of the last monitor in the list, or None when there is none."""
    monitor_positions = [
        index for index, item in enumerate(middleware) if isinstance(item, MonitorMiddleware)
    ]
    return monitor_positions[-1] if monitor_positions else None


def find_middleware_inside_monitor(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware after the last monitor, which LangChain nests inside it."""
    position = find_last_monitor_position(middleware)
    return () if position is None else middleware[position + 1 :]


def find_middleware_outside_monitor(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware before the last monitor, which LangChain wraps around it."""
    position = find_last_monitor_position(middleware)
    return () if position is None else middleware[:position]


def find_middleware_handling_tool_failures(
    middleware: Sequence[AnyAgentMiddleware],
) -> Sequence[AnyAgentMiddleware]:
    """Return the middleware known to retry or answer failed tool calls, in a list with a monitor.

    Tool calls are wrapped in list order too, so such a middleware re-runs or
    answers a failed delegation wherever it sits relative to the monitor.
    """
    if find_last_monitor_position(middleware) is None:
        return ()
    return [item for item in middleware if is_tool_failure_handling_middleware(item)]


def warn_about_placement(names: Sequence[str], *, reason: str) -> None:
    """Warn once for each named middleware, giving the reason its placement matters."""
    for name in names:
        # Level 3 skips this helper and `check_monitor_placement`, to point at their caller.
        warnings.warn(f"{name} {reason}", MonitorPlacementWarning, stacklevel=3)


def check_monitor_placement(*, middleware: Sequence[AnyAgentMiddleware]) -> list[str]:
    """Warn about each middleware placed where it undermines the last monitor.

    Pass the list given to `create_agent`. Three placements are warned about:

    - inside the monitor, a middleware that wraps model calls and is not known
      to only rewrite the request. It can return commands, which LangChain
      collects per call of the monitor's handler, so they would pile up from
      every sample the protocol draws [@langchain2026];
    - outside the monitor, a middleware known to retry failed model calls,
      such as `ModelRetryMiddleware`. A retry runs the whole step again with
      fresh samples, and the samples judged before the failure reach only a
      warning log line and a `MonitorStepFailedEvent`, never `monitor_log`;
    - anywhere in a list with a monitor, a middleware known to run failed tool
      calls again or answer them with an error message, such as
      `ToolRetryMiddleware`. A subagent's records reach its parent only in the
      result of the call that started it, so when the subagent's run raises,
      its blocks never reach Auto Mode's thread total, the run goes on, and a
      retry starts the subagent again from the same count. The check cannot
      tell whether the agent starts subagents, so it warns either way.

    A middleware whose `after_agent` hook sends a finished run back to the
    model needs no warning: after a halt, the monitor halts every further step
    until the thread records another run input, and nothing written between a
    run's start and its end becomes one. An untagged human message that a
    middleware listed before the monitor writes from its `before_agent` hook,
    or from its `after_agent` hook without a return to the model, does count
    as a run's input and lifts a halt. The check does not warn about those
    hooks, which many middleware have for sound reasons.

    Returns the names of the middleware it warned about.
    """
    misplaced_inside = [
        item.name
        for item in find_middleware_inside_monitor(middleware)
        if is_unsafe_inside_monitor(item)
    ]
    retrying_outside = [
        item.name
        for item in find_middleware_outside_monitor(middleware)
        if is_retrying_middleware(item)
    ]
    handling_tool_failures = [
        item.name for item in find_middleware_handling_tool_failures(middleware)
    ]
    warn_about_placement(
        misplaced_inside,
        reason="wraps model calls inside a monitor, so a state update it returns may come "
        "from a sample the protocol does not commit. Put the monitor last.",
    )
    warn_about_placement(
        retrying_outside,
        reason="retries failed model calls from outside a monitor, so a step that fails "
        "runs again from the start with fresh samples. The samples the monitor judged "
        "before the failure never reach monitor_log; only a warning log line and a "
        "monitor_step_failed event on stream_mode='custom' keep them.",
    )
    warn_about_placement(
        handling_tool_failures,
        reason="runs failed tool calls again or answers them with an error message. If "
        "this agent starts monitored subagents and its tools argument covers the task "
        "tool, a subagent whose run raises returns no records, so the blocks it recorded "
        "never count towards Auto Mode's total, and a retry starts it again from the "
        "same count.",
    )
    return misplaced_inside + retrying_outside + handling_tool_failures
