"""Enum options of the public constructors accept members only, and name them when refused.

A `StrEnum` member equals its string, but the library tells options apart by
identity, so a plain string used to select another behaviour silently: for
example `when_subagent_halts="stop_subagent_only"` stopped the whole run.
"""

from __future__ import annotations

from typing import Any

import hyoshigi
import pytest

from langchain_sync_monitors import DeferToResample, Resampling
from langchain_sync_monitors.contracts import SubagentHalt, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import RUN_EVERY_STEP


@pytest.mark.parametrize(
    ("parameter_name", "value", "accepted"),
    [
        ("task_author", "user", "TaskAuthor.USER, TaskAuthor.PARENT_AGENT"),
        (
            "when_subagent_halts",
            "stop_subagent_only",
            "SubagentHalt.STOP_SUBAGENT_ONLY, SubagentHalt.STOP_WHOLE_RUN",
        ),
    ],
)
def test_the_middleware_refuses_a_plain_string_option(
    parameter_name: str,
    value: str,
    accepted: str,
) -> None:
    # Arrange
    options: dict[str, Any] = {parameter_name: value}

    # Act / Assert
    with pytest.raises(ConfigurationError, match=f"{parameter_name} must be one of {accepted}"):
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=RUN_EVERY_STEP, **options)


def test_defer_to_resample_refuses_a_plain_string_resampling() -> None:
    # Arrange
    protocol = DeferToResample(
        defer_threshold=0.6,
        audit_threshold=0.9,
        max_resamples=2,
        resampling="parallel",  # ty: ignore[invalid-argument-type]
    )

    # Act / Assert: hyoshigi refuses it when the middleware builds the protocol
    with pytest.raises(ConfigurationError, match="resampling must be a Resampling"):
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=protocol)


def test_members_are_accepted_and_survive_a_subagent_copy() -> None:
    # Arrange
    middleware = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=DeferToResample(
            defer_threshold=0.6,
            audit_threshold=0.9,
            max_resamples=2,
            resampling=Resampling("parallel"),
        ),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )

    # Act
    copy = middleware.copy_for_subagent(subagent_name="worker")

    # Assert
    assert copy.task_author is TaskAuthor.PARENT_AGENT
    rebuilt = copy.protocols.async_protocol
    assert isinstance(rebuilt, hyoshigi.AsyncDeferToResample)
    assert rebuilt.resampling is Resampling.PARALLEL
    assert copy.when_subagent_halts is SubagentHalt.STOP_WHOLE_RUN
