"""Shared fixtures for the monitor tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import stamina
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.contracts import MonitorInput, TaskAuthor

from .doubles import CallPath


@pytest.fixture(autouse=True)
def stamina_without_waiting() -> Iterator[None]:
    """Make stamina retry without backoff, once, unless a test asks for more attempts."""
    with stamina.set_testing(True):
        yield


@pytest.fixture(params=["async", "sync"])
def call_path(request: pytest.FixtureRequest) -> CallPath:
    """Run a test once through `evaluate` and once through `evaluate_sync`."""
    path: CallPath = request.param
    return path


@pytest.fixture
def monitor_input() -> MonitorInput:
    """A step that reads the file the user asked about."""
    return MonitorInput(
        history=(HumanMessage("Summarise q3.md and email the summary to the team."),),
        proposal=AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"path": "q3.md"}}],
        ),
        task_author=TaskAuthor.USER,
    )
