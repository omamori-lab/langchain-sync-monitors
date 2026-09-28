"""Wrappers that are monitors themselves: averaging, calibration and cascades."""

from __future__ import annotations

import pytest

from langchain_sync_monitors.contracts import MonitorInput
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors.composition import (
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)

from .doubles import ScriptedMonitor


@pytest.fixture
def three_calls() -> ScriptedMonitor:
    return ScriptedMonitor(suspicions=[0.1, 0.7, 0.4])


@pytest.fixture
def honest_scores() -> list[float]:
    return [0.1, 0.2, 0.2, 0.3]


async def test_repeated_monitor_averages_concurrent_calls(
    three_calls: ScriptedMonitor,
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    repeated = RepeatedMonitor(monitor=three_calls, repeats=3)

    # Act
    verdict = await repeated.evaluate(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.4)
    assert three_calls.calls == 3


def test_repeated_monitor_averages_sequential_calls(
    three_calls: ScriptedMonitor,
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    repeated = RepeatedMonitor(monitor=three_calls, repeats=3)

    # Act
    verdict = repeated.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.4)
    assert three_calls.calls == 3


def test_repeated_monitor_keeps_the_most_suspicious_reason(
    three_calls: ScriptedMonitor,
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    repeated = RepeatedMonitor(monitor=three_calls, repeats=3)

    # Act
    verdict = repeated.evaluate_sync(monitor_input)

    # Assert
    assert "call 2 scored 0.7" in verdict.reason
    assert "over 3 calls" in verdict.reason


def test_one_repeat_returns_the_single_score(monitor_input: MonitorInput) -> None:
    # Arrange
    repeated = RepeatedMonitor(monitor=ScriptedMonitor(suspicions=[1.0]), repeats=1)

    # Act
    verdict = repeated.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == 1.0


def test_zero_repeats_is_rejected() -> None:
    # Act and Assert
    with pytest.raises(ConfigurationError, match="repeats"):
        RepeatedMonitor(monitor=ScriptedMonitor(suspicions=[0.0]), repeats=0)


def test_a_tied_score_lands_inside_the_tie(
    honest_scores: list[float],
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    calibrated = CalibratedMonitor(
        monitor=ScriptedMonitor(suspicions=[0.2]),
        honest_scores=honest_scores,
        random_seed=7,
    )

    # Act
    percentiles = [calibrated.evaluate_sync(monitor_input).suspicion for _ in range(50)]

    # Assert
    assert all(0.25 <= percentile <= 0.75 for percentile in percentiles)
    assert len(set(percentiles)) > 1


async def test_the_same_seed_breaks_ties_the_same_way(
    honest_scores: list[float],
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    first, second = (
        CalibratedMonitor(
            monitor=ScriptedMonitor(suspicions=[0.2]),
            honest_scores=honest_scores,
            random_seed=11,
        )
        for _ in range(2)
    )

    # Act
    first_verdict = await first.evaluate(monitor_input)
    second_verdict = await second.evaluate(monitor_input)

    # Assert
    assert first_verdict.suspicion == second_verdict.suspicion


@pytest.mark.parametrize(
    ("raw_suspicion", "expected_percentile"),
    [(0.0, 0.0), (0.25, 0.75), (0.3 + 1e-9, 1.0), (1.0, 1.0)],
)
def test_untied_scores_map_to_the_share_of_honest_scores_below(
    honest_scores: list[float],
    monitor_input: MonitorInput,
    raw_suspicion: float,
    expected_percentile: float,
) -> None:
    # Arrange
    calibrated = CalibratedMonitor(
        monitor=ScriptedMonitor(suspicions=[raw_suspicion]),
        honest_scores=honest_scores,
    )

    # Act
    verdict = calibrated.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(expected_percentile)
    assert "percentile of honest runs" in verdict.reason


@pytest.mark.parametrize("honest", [[], [0.2, float("nan")], [1.5]])
def test_unusable_honest_scores_are_rejected(honest: list[float]) -> None:
    # Act and Assert
    with pytest.raises(ConfigurationError, match="honest_scores"):
        CalibratedMonitor(monitor=ScriptedMonitor(suspicions=[0.0]), honest_scores=honest)


async def test_a_calm_fast_verdict_is_kept(monitor_input: MonitorInput) -> None:
    # Arrange
    careful = ScriptedMonitor(suspicions=[0.9])
    cascade = CascadeMonitor(
        fast=ScriptedMonitor(suspicions=[0.29]),
        careful=careful,
        escalate_at=0.3,
    )

    # Act
    verdict = await cascade.evaluate(monitor_input)

    # Assert
    assert verdict.suspicion == 0.29
    assert careful.calls == 0


async def test_a_fast_verdict_at_the_threshold_escalates(monitor_input: MonitorInput) -> None:
    # Arrange
    careful = ScriptedMonitor(suspicions=[0.1])
    cascade = CascadeMonitor(
        fast=ScriptedMonitor(suspicions=[0.3]),
        careful=careful,
        escalate_at=0.3,
    )

    # Act
    verdict = await cascade.evaluate(monitor_input)

    # Assert
    assert verdict.suspicion == 0.1
    assert careful.calls == 1
    assert "fast monitor scored 0.30" in verdict.reason


def test_the_sync_cascade_escalates_the_same_way(monitor_input: MonitorInput) -> None:
    # Arrange
    fast = ScriptedMonitor(suspicions=[0.8, 0.1])
    careful = ScriptedMonitor(suspicions=[0.95])
    cascade = CascadeMonitor(fast=fast, careful=careful, escalate_at=0.5)

    # Act
    escalated = cascade.evaluate_sync(monitor_input)
    kept = cascade.evaluate_sync(monitor_input)

    # Assert
    assert (escalated.suspicion, kept.suspicion) == (0.95, 0.1)
    assert careful.calls == 1


def test_an_escalation_threshold_outside_zero_to_one_is_rejected() -> None:
    # Arrange
    monitor = ScriptedMonitor(suspicions=[0.0])

    # Act and Assert
    with pytest.raises(ConfigurationError, match="escalate_at"):
        CascadeMonitor(fast=monitor, careful=monitor, escalate_at=1.2)
