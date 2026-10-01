"""Checks that a wrapper monitor computed each score from the calls its judgement made.

The monitor's spans give each judgement's Jev answers and chat judge calls,
so each wrapper's promise can be checked sample by sample, live or on fakes.
"""

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable, Iterator
from typing import Final

from tests.live.harness import CASCADE_ESCALATION_THRESHOLD, ENSEMBLE_SIZE, MonitorWrapper
from tests.live.honest_scores import HONEST_DECISION_MODEL_SCORES
from tests.live.reports import RunReport
from tests.live.traces import JudgementReport

SCORE_TOLERANCE: Final = 1e-9


def pair_samples_with_judgements(report: RunReport) -> Iterator[tuple[float, JudgementReport]]:
    """Pair each sample's suspicion with its judgement span, step by step, in the order drawn."""
    for step in report["steps"]:
        judgements = [
            judgement
            for judgement in report["trace"]["judgements"]
            if judgement["step_number"] == step["step_number"]
        ]
        for sample, judgement in zip(step["samples"], judgements, strict=True):
            yield sample["suspicion"], judgement


def find_repeat_problem(suspicion: float, judgement: JudgementReport) -> str | None:
    """Describe how a repeated judgement breaks its promise: the wrong count, or not the mean."""
    answers = judgement["classifier_scores"]
    if len(answers) != ENSEMBLE_SIZE:
        return f"{len(answers)} Jev answers, not {ENSEMBLE_SIZE}"
    if abs(statistics.fmean(answers) - suspicion) > SCORE_TOLERANCE:
        return f"score {suspicion} is not the mean of {answers}"
    return None


def find_cascade_problem(suspicion: float, judgement: JudgementReport) -> str | None:
    """Describe how a cascade judgement breaks its promise: escalating wrongly, or a lost score."""
    [fast] = judgement["classifier_scores"]
    escalated = judgement["monitor_calls"] == 1
    if escalated != (fast >= CASCADE_ESCALATION_THRESHOLD) or judgement["monitor_calls"] > 1:
        return f"Jev at {fast}, {judgement['monitor_calls']} chat judge calls"
    if not escalated and abs(fast - suspicion) > SCORE_TOLERANCE:
        return f"Jev at {fast} below the escalation, yet the score is {suspicion}"
    return None


def find_calibration_problem(suspicion: float, judgement: JudgementReport) -> str | None:
    """Describe a calibrated score outside the places its raw score takes among honest ones."""
    [raw] = judgement["classifier_scores"]
    honest = sorted(HONEST_DECISION_MODEL_SCORES)
    below = bisect.bisect_left(honest, raw)
    tied = bisect.bisect_right(honest, raw) - below
    places = len(honest) + 1
    if not below / places <= suspicion <= (below + tied + 1) / places:
        return f"Jev at {raw} ranks {below} below and {tied} tied, yet the score is {suspicion}"
    return None


WRAPPER_CHECKS: Final[dict[MonitorWrapper, Callable[[float, JudgementReport], str | None]]] = {
    MonitorWrapper.REPEATED: find_repeat_problem,
    MonitorWrapper.CASCADE: find_cascade_problem,
    MonitorWrapper.CALIBRATED: find_calibration_problem,
}


def list_wrapper_problems(report: RunReport, *, wrapper: MonitorWrapper) -> list[str]:
    """Check every sample's score against the calls its judgement made."""
    check = WRAPPER_CHECKS[wrapper]
    problems = (
        check(suspicion, judgement) for suspicion, judgement in pair_samples_with_judgements(report)
    )
    return [problem for problem in problems if problem is not None]
