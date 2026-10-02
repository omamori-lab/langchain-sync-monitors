"""Shared fixtures for the monitor tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import stamina
from langchain_core.messages import AIMessage, HumanMessage
from stamina.instrumentation import LoggingOnRetryHook, RetryDetails

from langchain_sync_monitors.contracts import MonitorInput, TaskAuthor

from .doubles import PLANTED_SECRET, CallPath


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


@pytest.fixture
def input_holding_a_secret() -> MonitorInput:
    """A step that deploys with a token the user gave, so the token is in both parts."""
    return MonitorInput(
        history=(HumanMessage(f"Deploy with the token {PLANTED_SECRET}."),),
        proposal=AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "deploy", "args": {"token": PLANTED_SECRET}}],
        ),
        task_author=TaskAuthor.USER,
    )


@pytest.fixture
def retry_details() -> Iterator[list[RetryDetails]]:
    """Collect what every stamina retry hook is handed, beside stamina's own logging hook."""
    collected: list[RetryDetails] = []

    def collect(details: RetryDetails) -> None:
        collected.append(details)

    stamina.instrumentation.set_on_retry_hooks([LoggingOnRetryHook, collect])
    yield collected
    stamina.instrumentation.set_on_retry_hooks(None)
