"""monitor_subagents gives every declarative subagent its own monitor, and refuses the rest."""

from __future__ import annotations

import pytest
from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent, create_deep_agent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import read_monitor
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


@pytest.fixture
def researcher() -> SubAgent:
    return SubAgent(name="researcher", description="Finds sources.", system_prompt="Research.")


def test_every_subagent_and_the_general_purpose_one_get_a_monitor(
    middleware: MonitorMiddleware,
    researcher: SubAgent,
) -> None:
    # Act
    specs = monitor_subagents(middleware=middleware, subagents=[researcher])

    # Assert
    assert [spec["name"] for spec in specs] == ["researcher", "general-purpose"]
    monitors = [read_monitor(spec) for spec in specs]
    assert [monitor.name for monitor in monitors] == [
        "monitor[researcher]",
        "monitor[general-purpose]",
    ]
    assert all(monitor.task_author is TaskAuthor.PARENT_AGENT for monitor in monitors)
    assert all(monitor.protocol is middleware.protocol for monitor in monitors)


def test_a_given_general_purpose_spec_is_kept_and_not_duplicated(
    middleware: MonitorMiddleware,
) -> None:
    # Arrange
    general_purpose = SubAgent(name="general-purpose", description="My own helper.")

    # Act
    specs = monitor_subagents(middleware=middleware, subagents=[general_purpose])

    # Assert
    assert [spec["description"] for spec in specs] == ["My own helper."]


def test_skills_go_to_the_general_purpose_subagent_only(
    middleware: MonitorMiddleware,
    researcher: SubAgent,
) -> None:
    # Arrange
    skills = ["/skills/"]

    # Act
    [monitored_researcher, general_purpose] = monitor_subagents(
        middleware=middleware, subagents=[researcher], skills=skills
    )

    # Assert
    assert general_purpose.get("skills") == ["/skills/"]
    assert general_purpose.get("skills") is not skills
    assert "skills" not in monitored_researcher
    assert "skills" not in GENERAL_PURPOSE_SUBAGENT


def test_without_skills_the_general_purpose_subagent_names_none(
    middleware: MonitorMiddleware,
) -> None:
    # Act
    [general_purpose] = monitor_subagents(middleware=middleware)

    # Assert
    assert "skills" not in general_purpose


LABELLED_SKILLS: list[str | tuple[str, str]] = [
    "/skills/user/",
    ("/repo/.claude/skills", "Project Claude"),
]


def test_labelled_skill_sources_reach_the_general_purpose_subagent_as_given(
    middleware: MonitorMiddleware,
) -> None:
    # Act: Deep Agents' SkillsMiddleware takes a path or a (path, label) pair
    specs = monitor_subagents(middleware=middleware, skills=LABELLED_SKILLS)

    # Assert: Deep Agents, which refuses a malformed source as it builds, builds the agent
    assert specs[-1]["name"] == "general-purpose"
    assert specs[-1]["skills"] == LABELLED_SKILLS
    create_deep_agent(
        model=ScriptedChatModel(responses=[]),
        middleware=[middleware],
        subagents=specs,
        skills=LABELLED_SKILLS,  # ty: ignore[invalid-argument-type]
    )


def test_empty_skills_are_kept_on_the_general_purpose_subagent(
    middleware: MonitorMiddleware,
) -> None:
    # Act
    [general_purpose] = monitor_subagents(middleware=middleware, skills=[])

    # Assert
    assert general_purpose.get("skills") == []


@pytest.mark.parametrize("skills", [["/skills/"], []])
def test_skills_with_a_given_general_purpose_spec_are_refused(
    middleware: MonitorMiddleware,
    skills: list[str],
) -> None:
    # Arrange
    general_purpose = SubAgent(name="general-purpose", description="My own helper.")

    # Act / Assert
    with pytest.raises(ConfigurationError, match="Set 'skills' on that spec"):
        monitor_subagents(middleware=middleware, subagents=[general_purpose], skills=skills)


def test_a_plain_string_for_skills_is_refused(middleware: MonitorMiddleware) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"Pass \['/skills/'\] for a single source"):
        monitor_subagents(
            middleware=middleware,
            skills="/skills/",
        )


def test_no_subagents_still_monitors_the_general_purpose_one(middleware: MonitorMiddleware) -> None:
    # Act
    specs = monitor_subagents(middleware=middleware)

    # Assert
    assert [read_monitor(spec).name for spec in specs] == ["monitor[general-purpose]"]


def test_overrides_give_named_subagents_a_different_monitor(
    middleware: MonitorMiddleware,
    researcher: SubAgent,
) -> None:
    # Arrange
    strict = MonitorMiddleware(
        monitor=KeywordMonitor(), protocol=HaltAfterOneSample(), label="guard"
    )

    # Act
    specs = monitor_subagents(
        middleware=middleware,
        subagents=[researcher],
        overrides={"researcher": strict, "general-purpose": strict},
    )

    # Assert
    monitors = [read_monitor(spec) for spec in specs]
    assert [monitor.name for monitor in monitors] == ["guard[researcher]", "guard[general-purpose]"]
    assert all(monitor.protocol is strict.protocol for monitor in monitors)


def test_overrides_for_unknown_subagents_are_rejected(
    middleware: MonitorMiddleware,
    researcher: SubAgent,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="coder"):
        monitor_subagents(
            middleware=middleware, subagents=[researcher], overrides={"coder": middleware}
        )


def test_existing_middleware_is_kept_and_the_given_specs_are_not_changed(
    middleware: MonitorMiddleware,
) -> None:
    # Arrange
    approval = HumanInTheLoopMiddleware(interrupt_on={"http_post": True})
    spec = SubAgent(name="coder", description="Writes code.", middleware=[approval])

    # Act
    [monitored, _general_purpose] = monitor_subagents(middleware=middleware, subagents=[spec])

    # Assert
    assert monitored.get("middleware", [])[0] is approval
    assert read_monitor(monitored).name == "monitor[coder]"
    assert spec.get("middleware") == [approval]


def test_compiled_subagents_are_refused(middleware: MonitorMiddleware) -> None:
    # Arrange
    runnable = create_agent(ScriptedChatModel(responses=[]))
    compiled = CompiledSubAgent(name="prebuilt", description="Prebuilt.", runnable=runnable)

    # Act / Assert
    with pytest.raises(ConfigurationError, match="own create_agent"):
        monitor_subagents(middleware=middleware, subagents=[compiled])


def test_remote_subagents_are_refused(middleware: MonitorMiddleware) -> None:
    # Arrange
    remote = AsyncSubAgent(name="analyst", description="Remote.", graph_id="research_agent")

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"compiled or remote.*own create_agent"):
        monitor_subagents(middleware=middleware, subagents=[remote])
