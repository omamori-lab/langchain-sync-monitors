"""The placement check warns about middleware that can return state updates inside a monitor."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langgraph.runtime import Runtime

from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst


class CommandingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Stands in for a middleware that returns state updates from wrap_model_call."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(request)


class UnsupportedContentMiddleware(CommandingMiddleware):
    """Shares its class name with Deep Agents' request-only middleware."""


@pytest.fixture
def monitor_middleware() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


def test_a_monitor_placed_last_raises_no_warning(monitor_middleware: MonitorMiddleware) -> None:
    # Arrange
    stack = [CommandingMiddleware(), monitor_middleware]

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == []


def test_a_model_call_wrapper_inside_the_monitor_is_named_in_a_warning(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    stack = [monitor_middleware, CommandingMiddleware()]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="CommandingMiddleware"):
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == ["CommandingMiddleware"]


class TeamMonitorMiddleware(MonitorMiddleware):
    """A user's own subclass of the monitor middleware."""


def test_a_model_call_wrapper_inside_a_monitor_subclass_is_named_in_a_warning() -> None:
    # Arrange
    monitor = TeamMonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())
    stack = [monitor, CommandingMiddleware()]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="CommandingMiddleware"):
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == ["CommandingMiddleware"]


def test_request_only_and_hook_only_middleware_are_safe_inside_the_monitor(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    approval = HumanInTheLoopMiddleware(interrupt_on={"http_post": True})
    stack = [monitor_middleware, UnsupportedContentMiddleware(), approval]

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == []


def test_only_middleware_after_the_last_monitor_is_checked(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    inner_monitor = monitor_middleware.copy_for_subagent(subagent_name="inner")
    stack = [monitor_middleware, CommandingMiddleware(), inner_monitor]

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == []


@pytest.mark.parametrize(
    "retrying",
    [
        ModelRetryMiddleware(),
        ModelFallbackMiddleware(ScriptedChatModel(responses=[])),
    ],
    ids=["retry", "fallback"],
)
def test_a_retrying_middleware_outside_the_monitor_is_named_in_a_warning(
    monitor_middleware: MonitorMiddleware,
    retrying: AgentMiddleware[Any, Any, Any],
) -> None:
    # Arrange
    stack = [retrying, CommandingMiddleware(), monitor_middleware]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="retries failed model calls") as caught:
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == [retrying.name]
    assert "monitor_step_failed" in str(caught[0].message)


class PatientRetryMiddleware(ModelRetryMiddleware):
    """A project's own retry policy, built on LangChain's."""


class HouseFallbackMiddleware(ModelFallbackMiddleware):
    """A project's own fallback chain, built on LangChain's."""


@pytest.mark.parametrize(
    "retrying",
    [
        PatientRetryMiddleware(),
        HouseFallbackMiddleware(ScriptedChatModel(responses=[])),
    ],
    ids=["retry", "fallback"],
)
def test_a_subclass_of_a_retrying_middleware_outside_the_monitor_is_named_in_a_warning(
    monitor_middleware: MonitorMiddleware,
    retrying: AgentMiddleware[Any, Any, Any],
) -> None:
    # Act
    with pytest.warns(MonitorPlacementWarning, match="retries failed model calls"):
        misplaced = check_monitor_placement(middleware=[retrying, monitor_middleware])

    # Assert
    assert misplaced == [retrying.name]


class ReturningToModelMiddleware(AgentMiddleware[Any, Any, Any]):
    """Stands in for a grader whose `after_agent` hook can send the run back to the model."""

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return None


@pytest.mark.parametrize("position", ["outside", "inside"])
def test_a_middleware_that_can_send_the_run_back_to_the_model_is_not_named(
    monitor_middleware: MonitorMiddleware,
    position: str,
) -> None:
    # Arrange: a halt stands against such a hook, whatever human message it adds
    returning = ReturningToModelMiddleware()
    if position == "outside":
        stack = [returning, monitor_middleware]
    else:
        stack = [monitor_middleware, returning]

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == []


def render_error_message(error: Exception, request: ToolCallRequest) -> str:
    return f"{request.tool_call['name']} failed with {type(error).__name__}."


def build_tool_retry_lookalike() -> AgentMiddleware[Any, Any, Any]:
    """Build a middleware that shares the tool retry's class name but wraps no tool call."""
    lookalike_class: type[AgentMiddleware[Any, Any, Any]] = type(
        "ToolRetryMiddleware", (AgentMiddleware,), {}
    )
    return lookalike_class()


@pytest.mark.parametrize(
    "handling",
    [
        ToolRetryMiddleware(),
        ToolErrorMiddleware(on_error=render_error_message),
    ],
    ids=["tool-retry", "tool-error"],
)
@pytest.mark.parametrize("position", ["outside", "inside"])
def test_a_middleware_that_handles_failed_tool_calls_is_named_in_a_warning(
    monitor_middleware: MonitorMiddleware,
    handling: AgentMiddleware[Any, Any, Any],
    position: str,
) -> None:
    # Arrange
    if position == "outside":
        stack = [handling, monitor_middleware]
    else:
        stack = [monitor_middleware, handling]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="failed tool calls") as caught:
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == [handling.name]
    assert "Auto Mode's total" in str(caught[0].message)


def test_a_subclass_of_a_tool_retry_middleware_is_named_in_a_warning(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    class LoggingToolRetryMiddleware(ToolRetryMiddleware):
        pass

    subclass = LoggingToolRetryMiddleware()

    # Act
    with pytest.warns(MonitorPlacementWarning, match="failed tool calls"):
        misplaced = check_monitor_placement(middleware=[subclass, monitor_middleware])

    # Assert
    assert misplaced == [subclass.name]


def test_a_placement_warning_points_at_the_code_that_checks_the_list(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        check_monitor_placement(middleware=[ToolRetryMiddleware(), monitor_middleware])

    # Assert
    assert caught[0].filename == __file__


def test_a_middleware_named_like_a_tool_retry_that_wraps_no_tool_call_is_not_named(
    monitor_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    stack = [build_tool_retry_lookalike(), monitor_middleware]

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        misplaced = check_monitor_placement(middleware=stack)

    # Assert
    assert misplaced == []


def test_a_stack_without_a_monitor_has_nothing_to_check() -> None:
    # Act
    misplaced = check_monitor_placement(
        middleware=[CommandingMiddleware(), ToolRetryMiddleware()],
    )

    # Assert
    assert misplaced == []
