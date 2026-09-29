"""Enum options of the public constructors accept members only, and name them when refused.

A `StrEnum` member equals its string, but the library tells options apart by
identity, so a plain string used to select another behaviour silently: for
example `when_subagent_halts="stop_subagent_only"` stopped the whole run.
"""

from __future__ import annotations

from typing import Any

import pytest

from langchain_sync_monitors.contracts import (
    FeedbackVisibility,
    Resampling,
    SubagentHalt,
    TaskAuthor,
)
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst


@pytest.mark.parametrize(
    ("parameter_name", "value", "accepted"),
    [
        ("task_author", "user", "TaskAuthor.USER, TaskAuthor.PARENT_AGENT"),
        (
            "feedback_visibility",
            "in_transcript",
            "FeedbackVisibility.HIDDEN, FeedbackVisibility.IN_TRANSCRIPT",
        ),
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
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst(), **options)


def test_defer_to_resample_refuses_a_plain_string_resampling() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"resampling must be one of Resampling.SEQUENTIAL, Resampling.PARALLEL, "
        r"got 'parallel'. Convert a string with Resampling\(value\)",
    ):
        DeferToResample(
            fallback=HaltRun(),
            defer_threshold=0.6,
            audit_threshold=0.9,
            resampling="parallel",  # ty: ignore[invalid-argument-type]
        )


def test_members_are_accepted_and_survive_a_subagent_copy() -> None:
    # Arrange
    middleware = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=DeferToResample(
            fallback=HaltRun(),
            defer_threshold=0.6,
            audit_threshold=0.9,
            resampling=Resampling("parallel"),
        ),
        feedback_visibility=FeedbackVisibility("in_transcript"),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )

    # Act
    copy = middleware.copy_for_subagent(subagent_name="worker")

    # Assert
    assert copy.task_author is TaskAuthor.PARENT_AGENT
    assert copy.feedback_visibility is FeedbackVisibility.IN_TRANSCRIPT
    assert copy.when_subagent_halts is SubagentHalt.STOP_WHOLE_RUN
