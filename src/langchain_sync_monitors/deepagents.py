"""Give every Deep Agents subagent its own monitor. Needs the `deepagents` extra.

Deep Agents does not pass the main agent's middleware to its subagents
[@deepagents2026], so a monitor on the main agent alone leaves every delegated
task unmonitored. `monitor_subagents` adds one to each subagent spec, including
the built-in general-purpose one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard

from langchain_sync_monitors._langchain import append_subagent_middleware
from langchain_sync_monitors.errors import ConfigurationError, MissingExtraError
from langchain_sync_monitors.middleware import MonitorMiddleware

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent

    type SubagentSpec = SubAgent | CompiledSubAgent | AsyncSubAgent

INSTALL_HINT = (
    "monitor_subagents needs Deep Agents. "
    "Install it with: pip install 'langchain-sync-monitors[deepagents]'"
)


def read_general_purpose_subagent() -> SubAgent:
    """Return a copy of the spec Deep Agents uses for its general-purpose subagent.

    Deep Agents is imported here, on first use, so the package imports without
    the optional extra and only this call asks for it.
    """
    try:
        from deepagents.middleware.subagents import (  # lanorme: ignore[IMPORT-001]
            GENERAL_PURPOSE_SUBAGENT,
        )
    except ImportError as error:
        raise MissingExtraError(INSTALL_HINT) from error
    return GENERAL_PURPOSE_SUBAGENT.copy()


def is_declarative_subagent(spec: SubagentSpec) -> TypeGuard[SubAgent]:
    """Tell whether a spec is declarative, so Deep Agents builds it and can add middleware."""
    return "runnable" not in spec and "graph_id" not in spec


def is_forked_subagent(spec: SubagentSpec) -> bool:
    """Tell whether a spec continues the parent's conversation, `mode="fork"` in Deep Agents."""
    return spec.get("mode") == "fork"


def build_fork_message(name: str) -> str:
    """Explain why a forked subagent cannot be monitored yet."""
    return (
        f"Subagent {name!r} has mode='fork', which the monitor does not support yet (issue "
        "#35). A fork continues the parent's conversation and inherits the main agent's "
        "monitor, which reads the parent agent's task as the user's words and records the "
        "fork's steps as the main agent's, so a halt inside the fork does not stop the run. "
        "Use mode='isolated' for a monitored subagent."
    )


def build_compiled_subagent_message(name: str) -> str:
    """Explain how to monitor a compiled or remote subagent in its own graph."""
    return (
        f"Subagent {name!r} is compiled or remote, so a monitor cannot be added to it here. "
        "Add one to its own create_agent() instead, named after the subagent and reading "
        f"its task as the parent agent's: MonitorMiddleware(..., agent_name={name!r}, "
        "task_author=TaskAuthor.PARENT_AGENT). With the default agent_name='main', its "
        "records count as the main agent's own, and SubagentHalt.STOP_WHOLE_RUN misses "
        "its halts."
    )


def build_declarative_specs(subagents: Sequence[SubagentSpec]) -> list[SubAgent]:
    """Return the specs to monitor, adding the general-purpose one when it is missing.

    Forked subagents raise, since their monitors would misread who wrote the
    task. Compiled and remote subagents are built outside Deep Agents, so
    middleware cannot be added to them; they raise rather than run
    unmonitored.
    """
    specs: list[SubAgent] = []
    for spec in subagents:
        if is_forked_subagent(spec):
            raise ConfigurationError(build_fork_message(spec["name"]))
        if is_declarative_subagent(spec):
            specs.append(spec)
            continue
        raise ConfigurationError(build_compiled_subagent_message(spec["name"]))
    general_purpose = read_general_purpose_subagent()
    if all(spec["name"] != general_purpose["name"] for spec in specs):
        specs.append(general_purpose)
    return specs


def build_monitored_spec(spec: SubAgent, *, middleware: MonitorMiddleware) -> SubAgent:
    """Return a copy of the spec with the subagent's own monitor after its middleware."""
    monitored = spec.copy()
    monitor = middleware.copy_for_subagent(subagent_name=spec["name"])
    monitored["middleware"] = append_subagent_middleware(
        spec.get("middleware", []),
        middleware=monitor,
    )
    return monitored


def monitor_subagents(
    *,
    middleware: MonitorMiddleware,
    subagents: Sequence[SubagentSpec] = (),
    overrides: Mapping[str, MonitorMiddleware] | None = None,
) -> list[SubAgent]:
    """Give every subagent a monitor, including the built-in general-purpose one.

    Each subagent gets a copy of `middleware`, or of its entry in `overrides`,
    named after the subagent and told that its task comes from the parent
    agent. Pass the result as `create_deep_agent(subagents=...)`.

    A subagent with `mode="fork"` raises `ConfigurationError` (issue #35). A
    fork continues the parent's conversation and inherits the main agent's
    middleware from Deep Agents, so it runs under the main agent's monitor.
    That monitor reads the fork's task, which the parent agent wrote, as the
    user's words, and records the fork's steps under the main agent's name,
    which hides a halt inside the fork from `SubagentHalt.STOP_WHOLE_RUN`.
    This holds for a fork passed to `create_deep_agent` without this helper
    too, so do not give a monitored agent forked subagents.

    A compiled or remote subagent raises `ConfigurationError` too; monitor it
    in its own graph with `agent_name` set to its name and
    `task_author=TaskAuthor.PARENT_AGENT`.
    """
    specs = build_declarative_specs(subagents)
    chosen = overrides or {}
    unknown_names = sorted(set(chosen) - {spec["name"] for spec in specs})
    if unknown_names:
        error_message = f"overrides name subagents that do not exist: {unknown_names}"
        raise ConfigurationError(error_message)
    return [
        build_monitored_spec(spec, middleware=chosen.get(spec["name"], middleware))
        for spec in specs
    ]
