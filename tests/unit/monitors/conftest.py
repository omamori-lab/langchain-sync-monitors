"""Shared fixtures for the monitor tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import stamina
from hyoshigi import Action, Author, Observation, Step, TextPart, ToolCall, Trajectory

from tests.support.fixtures import every_log_record as every_log_record
from tests.support.fixtures import retry_details as retry_details
from tests.support.fixtures import retry_frame_locals as retry_frame_locals
from tests.support.fixtures import retry_hook_calls as retry_hook_calls

from .doubles import PLANTED_SECRET, CallPath


@pytest.fixture(autouse=True)
def stamina_without_waiting() -> Iterator[None]:
    """Make stamina retry without backoff, once, unless a test asks for more attempts."""
    with stamina.set_testing(True):
        yield


@pytest.fixture(params=["async", "sync"])
def call_path(request: pytest.FixtureRequest) -> CallPath:
    """Run a test once through the async twin and once through the sync twin."""
    path: CallPath = request.param
    return path


def build_step(*, task: str, call: ToolCall) -> Step:
    """Return a trajectory that holds the user's task, and the action that calls `call`."""
    trajectory = Trajectory()
    trajectory.observe(Observation(author=Author.USER, parts=(TextPart(text=task),)))
    return Step(trajectory=trajectory, action=Action(tool_calls=(call,)))


@pytest.fixture
def step() -> Step:
    """A step that reads the file the user asked about."""
    return build_step(
        task="Summarise q3.md and email the summary to the team.",
        call=ToolCall(id="call-1", name="read_file", arguments='{"path": "q3.md"}'),
    )


@pytest.fixture
def step_holding_a_secret() -> Step:
    """A step that deploys with a token the user gave, so the token is in both parts."""
    return build_step(
        task=f"Deploy with the token {PLANTED_SECRET}.",
        call=ToolCall(id="call-1", name="deploy", arguments=f'{{"token": "{PLANTED_SECRET}"}}'),
    )
