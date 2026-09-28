"""Give every Deep Agents subagent its own monitor. Needs the `deepagents` extra.

Deep Agents does not pass the main agent's middleware to its subagents
[@deepagents2026], so a monitor on the main agent alone leaves every delegated
task unmonitored. `monitor_subagents` adds one to each subagent spec, including
the built-in general-purpose one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard

from langchain_sync_monitors._langchain import AnyAgentMiddleware
from langchain_sync_monitors.errors import ConfigurationError
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
        raise ImportError(INSTALL_HINT) from error
    return GENERAL_PURPOSE_SUBAGENT.copy()


def is_declarative_subagent(spec: SubagentSpec) -> TypeGuard[SubAgent]:
    """Tell whether a spec is declarative, so Deep Agents builds it and can add middleware."""
    return "runnable" not in spec and "graph_id" not in spec


def build_declarative_specs(subagents: Sequence[SubagentSpec]) -> list[SubAgent]:
    """Return the specs to monitor, adding the general-purpose one when it is missing.

    Compiled and remote subagents are built outside Deep Agents, so middleware
    cannot be added to them; they raise rather than run unmonitored.
    """
    specs: list[SubAgent] = []
    for spec in subagents:
        if is_declarative_subagent(spec):
            specs.append(spec)
            continue
        error_message = (
            f"Subagent {spec['name']!r} is compiled or remote, so a monitor cannot be added "
            "to it here. Add a MonitorMiddleware to its own create_agent() instead."
        )
        raise ConfigurationError(error_message)
    general_purpose = read_general_purpose_subagent()
    if all(spec["name"] != general_purpose["name"] for spec in specs):
        specs.append(general_purpose)
    return specs


def build_monitored_spec(spec: SubAgent, *, middleware: MonitorMiddleware) -> SubAgent:
    """Return a copy of the spec with the subagent's own monitor after its middleware."""
    monitored = spec.copy()
    monitor: AnyAgentMiddleware = middleware.copy_for_subagent(subagent_name=spec["name"])
    monitored["middleware"] = [*spec.get("middleware", []), monitor]
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

    A subagent with `mode="fork"` also inherits the main agent's middleware
    from Deep Agents, so it runs under the main agent's monitor as well as its
    own.
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
