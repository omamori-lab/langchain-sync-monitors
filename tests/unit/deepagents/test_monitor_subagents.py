"""monitor_subagents gives every declarative subagent its own monitor, and refuses the rest."""

from __future__ import annotations

import pytest
from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


@pytest.fixture
def researcher() -> SubAgent:
    return SubAgent(name="researcher", description="Finds sources.", system_prompt="Research.")


def read_monitor(spec: SubAgent) -> MonitorMiddleware:
    monitor = spec.get("middleware", [])[-1]
    assert isinstance(monitor, MonitorMiddleware)
    return monitor


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
    remote = AsyncSubAgent(name="remote", description="Remote.", graph_id="research_agent")

    # Act / Assert
    with pytest.raises(ConfigurationError, match="remote"):
        monitor_subagents(middleware=middleware, subagents=[remote])
