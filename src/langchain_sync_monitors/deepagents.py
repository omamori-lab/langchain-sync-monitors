"""Give every Deep Agents subagent its own monitor. Needs the `deepagents` extra.

Deep Agents does not pass the main agent's middleware to its subagents
[@deepagents2026], so a monitor on the main agent alone leaves every delegated
task unmonitored. `monitor_subagents` adds one to each subagent spec, including
the built-in general-purpose one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard, cast

from langchain_sync_monitors._langchain import append_subagent_middleware
from langchain_sync_monitors.errors import ConfigurationError, MissingExtraError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.options import (
    check_instance_option,
    check_name_part_option,
    describe_option_value,
)
from langchain_sync_monitors.placement import (
    find_nested_placements,
    warn_about_misplaced_middleware,
)

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent

    type SubagentSpec = SubAgent | CompiledSubAgent | AsyncSubAgent

type SkillSource = str | tuple[str, str]
"""A skill source Deep Agents reads: a path, or a `(path, label)` pair."""

INSTALL_HINT = (
    "monitor_subagents needs Deep Agents. "
    "Install it with: pip install 'langchain-sync-monitors[deepagents]'"
)


def read_general_purpose_subagent() -> SubAgent:
    """Return a copy of Deep Agents' base spec for its general-purpose subagent.

    The spec holds the subagent's name, description and prompt only. When Deep
    Agents adds the subagent itself, it also gives it the main agent's model,
    tools and skills, and applies the harness profile's settings for it.

    Deep Agents is imported here, on first use, so the package imports without
    the optional extra and only this call asks for it.
    """
    try:
        from deepagents.middleware.subagents import (  # lanorme: ignore[IMPORT-001]
            GENERAL_PURPOSE_SUBAGENT,
        )
    except ImportError as error:
        raise MissingExtraError(INSTALL_HINT) from error
    # A copy, so setting its skills leaves Deep Agents' own constant untouched.
    return GENERAL_PURPOSE_SUBAGENT.copy()


def is_declarative_subagent(spec: SubagentSpec) -> TypeGuard[SubAgent]:
    """Tell whether a spec is declarative, so Deep Agents builds it and can add middleware."""
    # A compiled subagent carries its `runnable`, and a remote one the `graph_id` it runs.
    return "runnable" not in spec and "graph_id" not in spec


def is_forked_subagent(spec: SubagentSpec) -> bool:
    """Tell whether a spec continues the parent's conversation, `mode="fork"` in Deep Agents."""
    return spec.get("mode") == "fork"


def build_fork_message(name: str) -> str:
    """Explain why a forked subagent cannot be monitored yet."""
    return (
        f"Subagent {name!r} has mode='fork', which the monitor does not support yet (issue "
        "#35). A fork continues the parent's conversation and inherits the main agent's "
        "monitor, which reads the task the parent agent wrote as the user's words. Use "
        "mode='isolated' for a monitored subagent."
    )


def build_compiled_subagent_message(name: str) -> str:
    """Explain how to monitor a compiled or remote subagent in its own graph."""
    return (
        f"Subagent {name!r} is compiled or remote, so a monitor cannot be added to it here. "
        "Add one to its own create_agent() instead, named after the subagent and reading "
        f"its task as the parent agent's: MonitorMiddleware(..., agent_name={name!r}, "
        "task_author=TaskAuthor.PARENT_AGENT), so monitor_log names its steps as its own."
    )


def build_general_purpose_skills_message(name: str) -> str:
    """Explain why `skills` cannot go with a general-purpose spec of the caller's own."""
    return (
        f"subagents already has a spec named {name!r}, so monitor_subagents adds no "
        "general-purpose subagent for skills to go to. Set 'skills' on that spec instead."
    )


def is_skill_source(value: object) -> TypeGuard[SkillSource]:
    """Tell whether a value is a skill source: a path, or a `(path, label)` pair of strings.

    Deep Agents' `SkillsMiddleware` takes both, and checks a pair this way
    [@deepagents2026].
    """
    if isinstance(value, str):
        return True
    return (
        isinstance(value, tuple)
        and len(value) == len(("path", "label"))
        and all(isinstance(part, str) for part in value)
    )


def read_skills_option(skills: Iterable[SkillSource] | None) -> list[SkillSource] | None:
    """Return the skill sources as a list, or `None`, raising `ConfigurationError` for others.

    A plain string is refused, which a list would split into one source per
    letter, and so is anything but an iterable of skill sources. A generator
    is read once.
    """
    if skills is None:
        return None
    if isinstance(skills, str):
        error_message = (
            f"skills must be a list of skill source paths, not the string {skills!r}. "
            f"Pass [{skills!r}] for a single source."
        )
        raise ConfigurationError(error_message)
    if isinstance(skills, bytes) or not isinstance(skills, Iterable):
        error_message = (
            f"skills must be a list of skill source paths, got {describe_option_value(skills)}"
        )
        raise ConfigurationError(error_message)
    sources = list(skills)
    for position, source in enumerate(sources):
        if not is_skill_source(source):
            error_message = (
                f"skills[{position}] must be a skill source path or a (path, label) pair of "
                f"strings, got {describe_option_value(source)}"
            )
            raise ConfigurationError(error_message)
    return sources


def read_subagent_specs(subagents: Iterable[SubagentSpec]) -> list[SubagentSpec]:
    """Return the subagent specs as a list, raising `ConfigurationError` unless each has a name.

    `subagents` may be any iterable of specs but a string or a single spec,
    and is read once. Each spec must be a mapping, as Deep Agents' specs are,
    whose name can name the subagent's monitor; Deep Agents checks the rest.
    """
    if isinstance(subagents, str | bytes | Mapping) or not isinstance(subagents, Iterable):
        error_message = (
            f"subagents must be a list of subagent specs, got "
            f"{describe_option_value(subagents)}. Wrap one spec in a list."
        )
        raise ConfigurationError(error_message)
    specs = list(subagents)
    for position, spec in enumerate(specs):
        check_instance_option(
            spec,
            option_type=Mapping,
            parameter_name=f"subagents[{position}]",
            hint="Pass a subagent spec, such as SubAgent(name=..., ...).",
        )
        check_name_part_option(spec.get("name"), parameter_name=f"subagents[{position}]['name']")
    return specs


def check_overrides_option(overrides: Mapping[str, MonitorMiddleware] | None) -> None:
    """Raise `ConfigurationError` unless `overrides` is `None` or maps names to monitors."""
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        error_message = (
            "overrides must map subagent names to MonitorMiddleware, "
            f"got {describe_option_value(overrides)}"
        )
        raise ConfigurationError(error_message)
    for name, override in overrides.items():
        if not isinstance(name, str):
            error_message = (
                "overrides must be keyed by subagent name, "
                f"got a key that is {describe_option_value(name)}"
            )
            raise ConfigurationError(error_message)
        check_instance_option(
            override,
            option_type=MonitorMiddleware,
            parameter_name=f"overrides[{name!r}]",
        )


def build_general_purpose_subagent(
    specs: Sequence[SubAgent],
    *,
    skills: list[SkillSource] | None,
) -> SubAgent | None:
    """Return the general-purpose spec to add, or `None` when `specs` has one.

    Deep Agents gives the main agent's skills to the general-purpose subagent
    it adds itself, but a spec passed in `subagents` gets only the skills it
    names [@deepagents2026], so they are set on the spec here. With a
    general-purpose spec of the caller's own, `skills` raises rather than go
    unused, even when it is empty.
    """
    general_purpose = read_general_purpose_subagent()
    if any(spec["name"] == general_purpose["name"] for spec in specs):
        if skills is not None:
            raise ConfigurationError(build_general_purpose_skills_message(general_purpose["name"]))
        return None
    if skills is not None:
        # Deep Agents types a spec's skills as paths, though its middleware also reads pairs.
        general_purpose["skills"] = cast("list[str]", list(skills))
    return general_purpose


def build_declarative_specs(
    subagents: Sequence[SubagentSpec],
    *,
    skills: list[SkillSource] | None,
) -> list[SubAgent]:
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
    general_purpose = build_general_purpose_subagent(specs, skills=skills)
    if general_purpose is not None:
        specs.append(general_purpose)
    return specs


def build_monitored_spec(spec: SubAgent, *, middleware: MonitorMiddleware) -> SubAgent:
    """Return a copy of the spec with the subagent's own monitor after its middleware.

    A monitor already in the spec's middleware now wraps the new one, so the
    stacked-monitor warnings of `check_monitor_placement` are raised here for
    the new list. Its other checks are left to `check_monitor_placement`,
    since they would warn about middleware the spec chose for itself.
    """
    monitored = spec.copy()
    monitor = middleware.copy_for_subagent(subagent_name=spec["name"])
    monitored["middleware"] = append_subagent_middleware(
        spec.get("middleware", []),
        middleware=monitor,
    )
    warn_about_misplaced_middleware(find_nested_placements(monitored["middleware"]))
    return monitored


def monitor_subagents(
    *,
    middleware: MonitorMiddleware,
    subagents: Sequence[SubagentSpec] = (),
    overrides: Mapping[str, MonitorMiddleware] | None = None,
    skills: Sequence[SkillSource] | None = None,
) -> list[SubAgent]:
    """Give every subagent a monitor, including the built-in general-purpose one.

    Each subagent gets a copy of `middleware`, or of its entry in `overrides`,
    named after the subagent and told that its task comes from the parent
    agent. Pass the result as `create_deep_agent(subagents=...)`.

    The general-purpose subagent is added as a spec of its own, since Deep
    Agents offers no way to add middleware to the one it builds itself and
    to no other agent. Pass the main agent's `skills`, the list given to
    `create_deep_agent(skills=...)`, so the subagent keeps them as Deep
    Agents' own would; a plain string raises `ConfigurationError`.

    A harness profile's `general_purpose_subagent` settings do not reach the
    spec: its description and prompt ignore them, though the profile's
    `base_system_prompt` and `system_prompt_suffix` still apply, and a
    profile that disables the subagent does not remove it. When the active
    profile disables it, drop the spec named `general-purpose` from the
    result; do this only then, since otherwise Deep Agents adds its own
    general-purpose subagent, unmonitored. To change the subagent, pass a spec
    named `general-purpose` in `subagents`; it is monitored in place of the
    built-in one, and `skills` then raises `ConfigurationError`, even when
    empty, since that spec takes only the skills it names.

    A subagent with `mode="fork"` raises `ConfigurationError` (issue #35). A
    fork continues the parent's conversation and inherits the main agent's
    middleware from Deep Agents, so it runs under the main agent's monitor.
    That monitor reads the fork's task, which the parent agent wrote, as the
    user's words. It records the fork's steps under the main agent's name
    but with the fork's own delegation, so they never count as the main
    agent's steps, and the main agent answers the fork's halts as
    `when_subagent_halts` says. The misread task holds for a fork passed to
    `create_deep_agent` without this helper too, so do not give a monitored
    agent forked subagents.

    A compiled or remote subagent raises `ConfigurationError` too; monitor it
    in its own graph with `agent_name` set to its name and
    `task_author=TaskAuthor.PARENT_AGENT`.

    The monitor goes after a spec's own middleware, so a monitor already
    there, one this helper added in an earlier call included, wraps it. When
    that stack loses or misjudges records, a `MonitorPlacementWarning` names
    the monitor inside, as `check_monitor_placement` would.

    Every option is checked before Deep Agents is imported: `middleware` and
    each override must be a `MonitorMiddleware`, `overrides` a mapping keyed
    by subagent name, `subagents` an iterable of specs, and `skills` one of
    skill sources, each a path or a `(path, label)` pair of strings, as Deep
    Agents' `SkillsMiddleware` takes them. A subagent's name names its
    monitor, so it must be a non-blank string without `:` or `|`. Anything
    else raises `ConfigurationError`.
    """
    check_instance_option(middleware, option_type=MonitorMiddleware, parameter_name="middleware")
    given_specs = read_subagent_specs(subagents)
    check_overrides_option(overrides)
    skill_sources = read_skills_option(skills)
    specs = build_declarative_specs(given_specs, skills=skill_sources)
    chosen = overrides or {}
    unknown_names = sorted(set(chosen) - {spec["name"] for spec in specs})
    if unknown_names:
        error_message = f"overrides name subagents that do not exist: {unknown_names}"
        raise ConfigurationError(error_message)
    return [
        build_monitored_spec(spec, middleware=chosen.get(spec["name"], middleware))
        for spec in specs
    ]
