"""What a calibrated threshold means, and how the wrappers treat seeds, failures and reasons."""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors.contracts import Monitor, MonitorInput, TaskAuthor, Verdict
from langchain_sync_monitors.monitors.composition import (
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)

from .doubles import CallPath, evaluate_on_path

AUDIT_THRESHOLD = 0.99
TRIAL_COUNT = 40_000
REASON_WITHOUT_NUMBERS = "It posts the API key to an unknown host."
RECORDED_SEEDED_DRAW = 0.1886877280579693
"""What seed 3 drew for the step in `test_a_seeded_draw_keeps_its_recorded_value`."""


@dataclass(kw_only=True)
class ScoreInProposal(Monitor):
    """Scores a step with the number written in its proposal, such as `score=0.5`."""

    reason: str = "fixed score"

    def score(self, monitor_input: MonitorInput) -> Verdict:
        value = float(monitor_input.proposal.text.split("=")[1].split()[0])
        return Verdict(suspicion=value, reason=self.reason)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return self.score(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return self.score(monitor_input)


@dataclass(kw_only=True)
class SlowOrFailingMonitor(Monitor):
    """Its second call fails at once; the others finish after `delay` seconds."""

    delay: float
    started: int = 0
    finished: list[int] = field(default_factory=list)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        self.started += 1
        call_number = self.started
        if call_number == 2:
            message = "judge timed out"
            raise TimeoutError(message)
        await asyncio.sleep(self.delay)
        self.finished.append(call_number)
        return Verdict(suspicion=0.1, reason="fine")

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return Verdict(suspicion=0.1, reason="fine")


def build_scored_step(
    score: float,
    *,
    label: str = "",
    task: str = "Summarise the report.",
    task_author: TaskAuthor = TaskAuthor.USER,
) -> MonitorInput:
    return MonitorInput(
        history=(HumanMessage(task),),
        proposal=AIMessage(f"score={score} {label}"),
        task_author=task_author,
    )


def measure_honest_flag_rate(*, calibration_size: int) -> float:
    """Draw exchangeable honest scores and count how often a new one reaches the threshold.

    Each trial draws `calibration_size + 1` scores from one distribution: the
    first ones are the honest scores, the last one is the new honest step.
    """
    generator = random.Random(calibration_size)
    flagged = 0
    for trial in range(TRIAL_COUNT):
        scores = [generator.random() for _ in range(calibration_size + 1)]
        calibrated = CalibratedMonitor(
            monitor=ScoreInProposal(),
            honest_scores=scores[:-1],
            random_seed=1,
        )
        verdict = calibrated.evaluate_sync(build_scored_step(scores[-1], label=f"trial {trial}"))
        flagged += verdict.suspicion >= AUDIT_THRESHOLD
    return flagged / TRIAL_COUNT


@pytest.mark.parametrize("calibration_size", [100, 150])
def test_a_new_honest_step_is_flagged_at_exactly_the_audit_budget(calibration_size: int) -> None:
    """The new score's rank among n + 1 exchangeable scores is uniform on the n + 1 slots.

    Spreading the step uniformly over its slot makes the calibrated suspicion
    uniform on [0, 1], so a threshold of 0.99 flags exactly 1% of new honest
    steps. The earlier `below / n` flagged (n - ceil(0.99 n) + 1) / (n + 1):
    2/101, about 1.98%, at n = 100, and 2/151, about 1.32%, at n = 150.
    """
    # Act
    rate = measure_honest_flag_rate(calibration_size=calibration_size)

    # Assert: 40,000 trials give a standard error of 0.0005
    assert rate == pytest.approx(1 - AUDIT_THRESHOLD, abs=0.0018)


def test_every_rank_slot_is_reached_and_none_reaches_one() -> None:
    # Arrange: each of 21 distinct scores in turn plays the new step
    scores = [index / 20 for index in range(21)]
    slot_count = len(scores)

    # Act
    percentiles = []
    for rank, score in enumerate(scores):
        honest = scores[:rank] + scores[rank + 1 :]
        calibrated = CalibratedMonitor(monitor=ScoreInProposal(), honest_scores=honest)
        percentiles.append((rank, calibrated.evaluate_sync(build_scored_step(score)).suspicion))

    # Assert
    assert all(rank / slot_count <= value < (rank + 1) / slot_count for rank, value in percentiles)


def test_a_seeded_step_gets_the_same_suspicion_whatever_ran_before() -> None:
    # Arrange
    honest = [0.5] * 20
    alone = CalibratedMonitor(monitor=ScoreInProposal(), honest_scores=honest, random_seed=42)
    after_other_work = CalibratedMonitor(
        monitor=ScoreInProposal(), honest_scores=honest, random_seed=42
    )
    after_other_work.evaluate_sync(build_scored_step(0.5, label="another run's step"))

    # Act
    first = alone.evaluate_sync(build_scored_step(0.5)).suspicion
    second = after_other_work.evaluate_sync(build_scored_step(0.5)).suspicion

    # Assert
    assert first == second


async def test_parallel_runs_do_not_move_a_seeded_suspicion() -> None:
    # Arrange
    honest = [0.5] * 20
    shared = CalibratedMonitor(monitor=ScoreInProposal(), honest_scores=honest, random_seed=7)
    alone = await CalibratedMonitor(
        monitor=ScoreInProposal(), honest_scores=honest, random_seed=7
    ).evaluate(build_scored_step(0.5))

    # Act
    together = await asyncio.gather(
        shared.evaluate(build_scored_step(0.5, label="the other run")),
        shared.evaluate(build_scored_step(0.5)),
    )

    # Assert
    assert together[1].suspicion == alone.suspicion


def test_different_steps_with_one_seed_break_ties_differently() -> None:
    # Arrange
    calibrated = CalibratedMonitor(
        monitor=ScoreInProposal(), honest_scores=[0.5] * 20, random_seed=3
    )

    # Act
    percentiles = {
        calibrated.evaluate_sync(build_scored_step(0.5, label=f"step {index}")).suspicion
        for index in range(20)
    }

    # Assert
    assert len(percentiles) == 20


@pytest.mark.parametrize(
    "other_step",
    [
        build_scored_step(0.5, task="Summarise the other report."),
        build_scored_step(0.5, task_author=TaskAuthor.PARENT_AGENT),
    ],
    ids=["history", "task-author"],
)
async def test_steps_that_propose_the_same_thing_after_a_different_history_break_ties_differently(
    other_step: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: the tie-break reads the whole step, not its proposal alone
    calibrated = CalibratedMonitor(
        monitor=ScoreInProposal(), honest_scores=[0.5] * 20, random_seed=3
    )

    # Act
    first = await evaluate_on_path(calibrated, build_scored_step(0.5), call_path=call_path)
    second = await evaluate_on_path(calibrated, other_step, call_path=call_path)

    # Assert
    assert first.suspicion != second.suspicion


async def test_a_seeded_draw_keeps_its_recorded_value(call_path: CallPath) -> None:
    """The docs promise that a seeded evaluation replays exactly.

    So a change to what the draw reads, or to how it joins the parts of a
    step, must show up here as a deliberate change, not as silently different
    suspicions in a replayed evaluation.
    """
    # Arrange: one honest score tied with the step's, so the suspicion is the draw itself;
    # the step has messages of two types, a task author and a proposal with a tool call.
    calibrated = CalibratedMonitor(monitor=ScoreInProposal(), honest_scores=[0.5], random_seed=3)
    step = MonitorInput(
        history=(HumanMessage("Summarise the report."), AIMessage("I will read it first.")),
        proposal=AIMessage(
            "score=0.5 Reading the report.",
            tool_calls=[{"name": "read_file", "args": {"path": "q3.md"}, "id": "call-1"}],
        ),
        task_author=TaskAuthor.PARENT_AGENT,
    )

    # Act
    verdict = await evaluate_on_path(calibrated, step, call_path=call_path)

    # Assert
    assert verdict.suspicion == RECORDED_SEEDED_DRAW


async def test_a_failed_repeat_cancels_the_other_calls_and_raises_its_own_error() -> None:
    # Arrange
    inner = SlowOrFailingMonitor(delay=0.2)
    repeated = RepeatedMonitor(monitor=inner, repeats=3)

    # Act
    with pytest.raises(TimeoutError, match="judge timed out"):
        await repeated.evaluate(build_scored_step(0.1))
    await asyncio.sleep(0.3)

    # Assert
    assert inner.started == 3
    assert inner.finished == []


async def test_the_wrappers_pass_the_inner_reason_on_without_numbers(call_path: CallPath) -> None:
    """Auto Mode shows the reason to the agent, so no score may reach it."""
    # Arrange
    careful = ScoreInProposal(reason=REASON_WITHOUT_NUMBERS)
    fast = ScoreInProposal(reason="The fast stage escalates.")
    stack = CalibratedMonitor(
        monitor=RepeatedMonitor(
            monitor=CascadeMonitor(fast=fast, careful=careful, escalate_at=0.5),
            repeats=3,
        ),
        honest_scores=[0.1, 0.2, 0.3],
        random_seed=5,
    )

    # Act
    verdict = await evaluate_on_path(stack, build_scored_step(0.9), call_path=call_path)

    # Assert
    assert verdict.reason == REASON_WITHOUT_NUMBERS
    assert not re.search(r"\d", verdict.reason)
