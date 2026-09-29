"""The shared contracts validate what they must and agree with their stored forms."""

from __future__ import annotations

import math
from typing import get_args

import pytest

from langchain_sync_monitors.contracts import Channel, Outcome, OutcomeName, Verdict
from langchain_sync_monitors.errors import InvalidSuspicionError, MonitorError


@pytest.mark.parametrize("suspicion", [0.0, 0.5, 1.0])
def test_verdict_accepts_suspicion_in_the_unit_interval(suspicion: float) -> None:
    # Act
    verdict = Verdict(suspicion=suspicion, reason="in range")

    # Assert
    assert verdict.suspicion == suspicion


@pytest.mark.parametrize("suspicion", [-0.01, 1.01, math.nan, math.inf])
def test_verdict_rejects_suspicion_outside_the_unit_interval(suspicion: float) -> None:
    # Act / Assert
    with pytest.raises(ValueError, match="between 0 and 1"):
        Verdict(suspicion=suspicion, reason="out of range")


@pytest.mark.parametrize("suspicion", [math.nan, 1.5])
def test_an_invalid_suspicion_is_a_library_error(suspicion: float) -> None:
    # Act
    with pytest.raises(MonitorError) as raised:
        Verdict(suspicion=suspicion, reason="a custom monitor's bad score")

    # Assert
    assert isinstance(raised.value, InvalidSuspicionError)
    assert isinstance(raised.value, ValueError)


def test_default_view_reads_actions_but_not_reasoning() -> None:
    # Act
    channels = Channel.ACTIONS

    # Assert
    assert Channel.USER in channels
    assert Channel.TOOL_CALLS in channels
    assert Channel.SUBAGENT_REPORTS in channels
    assert Channel.REASONING not in channels
    assert Channel.AGENT_TEXT not in channels
    assert Channel.TOOL_RESULTS not in channels


def test_everything_view_reads_every_channel() -> None:
    # Arrange
    single_channels = [
        Channel.USER,
        Channel.REASONING,
        Channel.AGENT_TEXT,
        Channel.TOOL_CALLS,
        Channel.TOOL_RESULTS,
        Channel.SUBAGENT_REPORTS,
    ]

    # Act
    missing = [channel for channel in single_channels if channel not in Channel.ALL]

    # Assert
    assert missing == []


def test_outcome_values_match_the_stored_outcome_names() -> None:
    # Act
    enum_values = {outcome.value for outcome in Outcome}

    # Assert
    assert enum_values == set(get_args(OutcomeName))
