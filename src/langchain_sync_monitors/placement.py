"""Checks where middleware sits around a monitor in a `create_agent` list.

In a `create_agent` middleware list the monitor goes last. LangChain nests
`wrap_model_call` handlers with the first middleware outermost, so a
middleware after the monitor runs inside it, once per sample the protocol
draws, and one before it wraps the whole monitored step [@langchain2026].
Tool calls are wrapped in the same order. A second monitor in the list sits
inside the first. `check_monitor_placement` names the middleware in a list
that undermines what a monitor records.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from langchain.agents.middleware.types import AgentMiddleware

from langchain_sync_monitors._langchain import AnyAgentMiddleware
from langchain_sync_monitors.contracts import ControlProtocol, FeedbackVisibility
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.options import check_instance_option, describe_option_value
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    HaltRun,
    TrustedMonitoring,
)
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY

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


RESAMPLING_PROTOCOLS = frozenset({DeferToResample, DeferToTrusted})
"""The protocol classes that reject a sample without telling the agent why."""

LIBRARY_FALLBACKS = frozenset({DeferToTrustedModel, HaltRun})
"""The library's own fallbacks, which never add a blocked attempt to the step."""


class MonitorPlacementWarning(UserWarning):
    """A middleware placed around or inside a monitor undermines what the monitor records.

    Inside a monitor, a middleware can return state updates for samples the
    monitor rejects. Inside a monitor whose protocol can call the model more
    than once in a step, a second monitor loses the records of all but its
    last call, or piles them up for calls drawn at once. A monitor inside
    another that commits its blocked attempts with the step has the outer
    monitor judge a blocked proposal. Outside a monitor, a middleware that
    retries failed model calls runs the whole step again, and the samples
    judged before the failure never reach `monitor_log`. Anywhere in the
    list, a middleware that runs failed tool calls again or answers them lets
    a subagent's run fail without its blocks reaching Auto Mode's thread
    total.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class MisplacedMiddleware:
    """A middleware placed where it undermines a monitor, and the reason it does."""

    name: str
    reason: str


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


def is_single_call_protocol(protocol: ControlProtocol) -> bool:
    """Tell whether a protocol is known to call the rest of the stack at most once per step.

    Three settings are: `TrustedMonitoring`; `DeferToResample` with no
    resamples and `HaltRun` as its fallback; and `AutoMode` that stops at its
    first block in a step, because `max_consecutive_blocks` or
    `max_total_blocks` is 1, and then halts. Every other protocol can draw
    again, or have the trusted model write the step through the rest of the
    stack. Only the exact classes count, and the class is read before any
    setting, since a subclass may call again or never set them.
    """
    if type(protocol) is DeferToResample:
        draws_once, fallback = protocol.max_resamples == 0, protocol.fallback
    elif type(protocol) is AutoMode:
        blocks_allowed = min(protocol.max_consecutive_blocks, protocol.max_total_blocks)
        draws_once, fallback = blocks_allowed == 1, protocol.when_limit_reached
    else:
        return type(protocol) is TrustedMonitoring
    return draws_once and type(fallback) is HaltRun


def is_never_blocking_protocol(protocol: ControlProtocol) -> bool:
    """Tell whether a protocol is known never to block a sample, so no step holds a blocked attempt.

    `TrustedMonitoring` never blocks, and Defer to Resample and Defer to
    Trusted with one of the library's fallbacks reject samples without
    telling the agent. Auto Mode blocks, and any other protocol, subclasses
    included, may.
    """
    if type(protocol) is TrustedMonitoring:
        return True
    # The class is read first, so a subclass that never set a fallback cannot raise here.
    if type(protocol) in RESAMPLING_PROTOCOLS and isinstance(protocol, DeferToResample):
        return type(protocol.fallback) in LIBRARY_FALLBACKS
    return False


def is_showing_blocks(middleware: AnyAgentMiddleware) -> bool:
    """Tell whether a middleware is a monitor that may commit blocked attempts with its step."""
    return (
        isinstance(middleware, MonitorMiddleware)
        and middleware.feedback_visibility is FeedbackVisibility.IN_TRANSCRIPT
        and not is_never_blocking_protocol(middleware.protocol)
    )


def find_monitors_before(
    middleware: Sequence[AnyAgentMiddleware],
    *,
    position: int,
) -> list[MonitorMiddleware]:
    """Return the monitors listed before `position`, which LangChain wraps around the one there."""
    return [item for item in middleware[:position] if isinstance(item, MonitorMiddleware)]


def find_repeating_monitors_before(
    middleware: Sequence[AnyAgentMiddleware],
    *,
    position: int,
) -> list[MonitorMiddleware]:
    """Return the monitors around the one at `position` whose protocol may call the model again."""
    return [
        monitor
        for monitor in find_monitors_before(middleware, position=position)
        if not is_single_call_protocol(monitor.protocol)
    ]


def describe_monitors(monitors: Sequence[MonitorMiddleware]) -> str:
    """Name each monitor with its protocol's class, such as `guard[main] (DeferToResample)`."""
    return ", ".join(f"{monitor.name} ({type(monitor.protocol).__name__})" for monitor in monitors)


def build_repeated_call_reason(
    middleware: AnyAgentMiddleware,
    *,
    repeating: Sequence[MonitorMiddleware],
) -> str:
    """Say why a middleware inside a monitor that may call the model again undermines it."""
    around = describe_monitors(repeating)
    if isinstance(middleware, MonitorMiddleware):
        return (
            f"sits inside {around}, which can call the model more than once in a step. "
            f"LangChain keeps the commands of the last call only, so the samples "
            f"{middleware.name} judged in earlier calls never reach monitor_log, and calls "
            "drawn at once under ainvoke() leave one record each under the same step number, "
            "so later step numbers are skipped. As the outer monitor's protocol, use one "
            "that calls the model at most once per step, such as TrustedMonitoring, "
            "DeferToResample(max_resamples=0, fallback=HaltRun()), or AutoMode with "
            "max_consecutive_blocks=1 or max_total_blocks=1 and when_limit_reached=HaltRun(). "
            "Or judge with both monitors in one MonitorMiddleware, through a Monitor of your "
            "own that runs both and keeps the higher suspicion."
        )
    return (
        f"wraps model calls inside {around}, which can call the model more than once in a "
        "step, so a state update it returns may come from a sample the protocol does not "
        f"commit. List it before {repeating[0].name}."
    )


def find_middleware_inside_repeating_monitors(
    middleware: Sequence[AnyAgentMiddleware],
) -> list[MisplacedMiddleware]:
    """Return each middleware up to the last monitor that sits inside a monitor that may call again.

    A monitor whose protocol can call the rest of the stack more than once in a
    step undermines every middleware inside it that returns commands: LangChain
    keeps the commands of the last call only, and calls drawn at once pile
    theirs up [@langchain2026]. A monitor always returns its record as a
    command. The middleware after the last monitor are left to the check that
    warns about any middleware inside a monitor.
    """
    last_position = find_last_monitor_position(middleware)
    candidates = () if last_position is None else middleware[: last_position + 1]
    nested: list[MisplacedMiddleware] = []
    for position, item in enumerate(candidates):
        repeating = find_repeating_monitors_before(middleware, position=position)
        if repeating and is_unsafe_inside_monitor(item):
            reason = build_repeated_call_reason(item, repeating=repeating)
            nested.append(MisplacedMiddleware(name=item.name, reason=reason))
    return nested


def find_monitors_showing_blocks_to_monitors(
    middleware: Sequence[AnyAgentMiddleware],
) -> list[MisplacedMiddleware]:
    """Return each monitor inside another that commits its blocked attempts with the step.

    With `FeedbackVisibility.IN_TRANSCRIPT`, the response a monitor commits
    starts with each blocked attempt, and a monitor outside it judges the first
    AI message of that response: a proposal already blocked, never the step
    that runs. A protocol known never to block commits no blocked attempt.
    """
    nested: list[MisplacedMiddleware] = []
    for position, item in enumerate(middleware):
        around = find_monitors_before(middleware, position=position)
        if around and is_showing_blocks(item):
            reason = (
                f"commits its blocked attempts with each step from inside "
                f"{describe_monitors(around)}, so the monitor outside it judges the first "
                "blocked proposal instead of the step that runs, and its record shows that "
                "proposal. Give a monitor inside another "
                "feedback_visibility=FeedbackVisibility.HIDDEN."
            )
            nested.append(MisplacedMiddleware(name=item.name, reason=reason))
    return nested


def find_nested_placements(middleware: Sequence[AnyAgentMiddleware]) -> list[MisplacedMiddleware]:
    """Return each middleware whose place inside another monitor undermines it, with the reason."""
    return [
        *find_middleware_inside_repeating_monitors(middleware),
        *find_monitors_showing_blocks_to_monitors(middleware),
    ]


def check_middleware_list_option(middleware: object) -> None:
    """Raise `ConfigurationError` unless `middleware` is a sequence of middleware.

    It is the list given to `create_agent`. The check slices it and walks it
    more than once, so a generator, which one walk would use up, is refused.
    """
    if isinstance(middleware, str | bytes) or not isinstance(middleware, Sequence):
        message = (
            "middleware must be the list given to create_agent, "
            f"got {describe_option_value(middleware)}"
        )
        raise ConfigurationError(message)
    for position, item in enumerate(middleware):
        check_instance_option(
            item,
            option_type=AgentMiddleware,
            parameter_name=f"middleware[{position}]",
        )


def warn_about_placement(names: Sequence[str], *, reason: str) -> None:
    """Warn once for each named middleware, giving the reason its placement matters."""
    warn_about_misplaced_middleware(
        [MisplacedMiddleware(name=name, reason=reason) for name in names]
    )


def warn_about_misplaced_middleware(misplaced: Sequence[MisplacedMiddleware]) -> list[str]:
    """Warn once for each misplaced middleware, giving its own reason, and return their names.

    Both `check_monitor_placement` and `monitor_subagents` warn through here,
    so the warning skips every frame of the library to point at their caller.
    """
    for placement in misplaced:
        warnings.warn(
            f"{placement.name} {placement.reason}",
            MonitorPlacementWarning,
            stacklevel=2,
            skip_file_prefixes=(LIBRARY_DIRECTORY,),
        )
    return [placement.name for placement in misplaced]


def check_monitor_placement(*, middleware: Sequence[AnyAgentMiddleware]) -> list[str]:
    """Warn about each middleware placed where it undermines a monitor.

    Pass the list given to `create_agent`. Five placements are warned about:

    - inside the last monitor, a middleware that wraps model calls and is not
      known to only rewrite the request. It can return commands, which
      LangChain collects per call of the monitor's handler, so they would pile
      up from every sample the protocol draws [@langchain2026];
    - inside a monitor whose protocol can call the model more than once in a
      step, another monitor, or such a middleware listed before the last
      monitor. A monitor returns its record as a command, so the inner monitor
      keeps the record of the last call only, and calls drawn at once under
      `ainvoke()` leave one record each under the same step number. Of the
      library's protocols, only `TrustedMonitoring`,
      `DeferToResample(max_resamples=0, fallback=HaltRun())` and `AutoMode`
      with `max_consecutive_blocks=1` or `max_total_blocks=1` and
      `when_limit_reached=HaltRun()` call the model at most once, and only
      their exact classes count;
    - inside another monitor, a monitor with
      `FeedbackVisibility.IN_TRANSCRIPT` whose protocol may block, such as
      `AutoMode`. The monitor outside it judges the first blocked proposal
      instead of the step that runs;
    - outside the last monitor, a middleware known to retry failed model calls,
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
    hooks, which many middleware have for sound reasons. Nor does it warn
    about a monitor inside one that calls the model at most once: each
    monitor's record holds its own decision, so when the outer monitor halts
    the step, only the outer record says what ran. `monitor_subagents` runs
    the second and third checks on each subagent's middleware list.

    Returns the names of the middleware it warned about, once for each
    warning. Whatever the placement, the check only warns; a `middleware` that
    is not a sequence of middleware, such as a string or a generator, raises
    `ConfigurationError`.
    """
    check_middleware_list_option(middleware)
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
    nested = warn_about_misplaced_middleware(find_nested_placements(middleware))
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
    return misplaced_inside + nested + retrying_outside + handling_tool_failures
