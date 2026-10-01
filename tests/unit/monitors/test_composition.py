"""Wrappers that are monitors themselves: averaging, calibration and cascades."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import pytest
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import MonitorInput
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors.composition import (
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)

from .doubles import CallPath, ScriptedMonitor, evaluate_on_path


def build_step_with_text(monitor_input: MonitorInput, *, text: str) -> MonitorInput:
    return dataclasses.replace(monitor_input, proposal=AIMessage(content=text))


@pytest.fixture
def three_calls() -> ScriptedMonitor:
    return ScriptedMonitor(suspicions=[0.1, 0.7, 0.4])


@pytest.fixture
def honest_scores() -> list[float]:
    return [0.1, 0.2, 0.2, 0.3]


async def test_repeated_monitor_averages_its_calls(
    three_calls: ScriptedMonitor,
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: that the async path's calls overlap is checked in test_calibration.py.
    repeated = RepeatedMonitor(monitor=three_calls, repeats=3)

    # Act
    verdict = await evaluate_on_path(repeated, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.4)
    assert three_calls.calls == 3


def test_equal_means_compare_equal_after_rounding(
    monitor_input: MonitorInput,
) -> None:
    # Arrange: both steps average to 0.1, but their float sums differ in the last bits.
    first = RepeatedMonitor(monitor=ScriptedMonitor(suspicions=[0.1, 0.0, 0.2]), repeats=3)
    second = RepeatedMonitor(monitor=ScriptedMonitor(suspicions=[0.0, 0.0, 0.3]), repeats=3)

    # Act
    first_mean = first.evaluate_sync(monitor_input).suspicion
    second_mean = second.evaluate_sync(monitor_input).suspicion

    # Assert
    assert first_mean == second_mean


def test_repeated_monitor_keeps_the_most_suspicious_reason(
    three_calls: ScriptedMonitor,
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    repeated = RepeatedMonitor(monitor=three_calls, repeats=3)

    # Act
    verdict = repeated.evaluate_sync(monitor_input)

    # Assert
    assert verdict.reason == "call 2 scored 0.7"


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


@pytest.mark.parametrize("random_seed", [None, 7], ids=["unseeded", "seeded"])
def test_tied_scores_spread_at_random_over_the_slots_of_the_tie(
    honest_scores: list[float],
    monitor_input: MonitorInput,
    random_seed: int | None,
) -> None:
    # Arrange
    calibrated = CalibratedMonitor(
        monitor=ScriptedMonitor(suspicions=[0.2]),
        honest_scores=honest_scores,
        random_seed=random_seed,
    )
    steps = [build_step_with_text(monitor_input, text=f"step {index}") for index in range(50)]

    # Act
    percentiles = [calibrated.evaluate_sync(step).suspicion for step in steps]

    # Assert: one honest score below, two tied, so slots 1 to 3 of 5, the first and last of
    # which 50 uniform draws each miss with a chance of (2/3)^50, about 2e-9
    assert all(1 / 5 <= percentile < 4 / 5 for percentile in percentiles)
    assert len(set(percentiles)) > 1
    assert min(percentiles) < 2 / 5
    assert max(percentiles) >= 3 / 5


def test_different_seeds_break_the_same_ties_differently(
    honest_scores: list[float],
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    first, second = (
        CalibratedMonitor(
            monitor=ScriptedMonitor(suspicions=[0.2]),
            honest_scores=honest_scores,
            random_seed=seed,
        )
        for seed in (11, 12)
    )
    steps = [build_step_with_text(monitor_input, text=f"step {index}") for index in range(5)]

    # Act
    pairs = [
        (first.evaluate_sync(step).suspicion, second.evaluate_sync(step).suspicion)
        for step in steps
    ]

    # Assert
    assert all(
        first_percentile != second_percentile for first_percentile, second_percentile in pairs
    )


@pytest.mark.parametrize(
    ("raw_suspicion", "honest_scores_below"),
    [(0.0, 0), (0.25, 3), (0.3 + 1e-9, 4), (1.0, 4)],
)
def test_an_untied_score_lands_in_its_own_rank_slot(
    honest_scores: list[float],
    monitor_input: MonitorInput,
    raw_suspicion: float,
    honest_scores_below: int,
) -> None:
    # Arrange
    calibrated = CalibratedMonitor(
        monitor=ScriptedMonitor(suspicions=[raw_suspicion]),
        honest_scores=honest_scores,
    )
    slot_count = len(honest_scores) + 1

    # Act
    verdict = calibrated.evaluate_sync(monitor_input)

    # Assert
    assert honest_scores_below / slot_count <= verdict.suspicion
    assert verdict.suspicion < (honest_scores_below + 1) / slot_count
    assert verdict.reason == f"call 1 scored {raw_suspicion}"


async def test_honest_scores_given_as_a_generator_are_all_ranked_among(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: a generator is consumed by one pass, and ranking needs two.
    calibrated = CalibratedMonitor(
        monitor=ScriptedMonitor(suspicions=[0.95]),
        honest_scores=(0.1 for _ in range(500)),
    )

    # Act
    verdicts = [
        await evaluate_on_path(calibrated, monitor_input, call_path=call_path) for _ in range(20)
    ]

    # Assert: above all 500 honest scores, the step lands in the top slot.
    assert min(verdict.suspicion for verdict in verdicts) >= 500 / 501


@pytest.mark.parametrize(
    "honest",
    [[], (score for score in ()), [0.2, float("nan")], [1.5]],
    ids=["empty list", "empty generator", "not a number", "above one"],
)
def test_unusable_honest_scores_are_rejected(honest: Iterable[float]) -> None:
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
    assert verdict.reason == "call 1 scored 0.1"


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
