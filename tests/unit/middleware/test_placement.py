"""The placement check warns about middleware that can return state updates inside a monitor."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

from langchain_sync_monitors.middleware import (
    MonitorMiddleware,
    MonitorPlacementWarning,
    check_monitor_placement,
)
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


def test_a_stack_without_a_monitor_has_nothing_to_check() -> None:
    # Act
    misplaced = check_monitor_placement(middleware=[CommandingMiddleware()])

    # Assert
    assert misplaced == []
