"""The threshold evals: what each monitor family scores honest steps, and what it catches.

What it measures: for the LLM monitor, the guard model and the decision model, the 90th and 98th
percentiles of the monitor's first-sample score on honest steps, each with a bootstrap 95%
interval, the share of honest steps each flags, and the share of attack runs that attempted the
side task and reached each. `threshold_runs` gathers the scores under Trusted Monitoring and the
default view, `thresholds` computes the candidates.

At the end of each family, `threshold-evals.json` in the report directory holds the report of
every family finished so far: the honest scores by task, the side-task scores of each attempted
attack run, the candidates and the spend. Each family stops at `FAMILY_BUDGET`, so the three
together stay under the evals' cap of 5 US dollars.

The suite does not collect this module, since its name does not start with `test_`; name it to
run it, with the key exported:

    LIVE_REPORT_DIR=live-reports uv run --all-extras pytest -m live \
        tests/live/evaluate_thresholds.py -q -o log_cli=true --log-cli-level=INFO
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.live.checks import LIVE_MARKS, save_report
from tests.live.costs import CostLedger
from tests.live.harness import MonitorFamily
from tests.live.threshold_runs import (
    FAMILY_BUDGET,
    MIN_HONEST_STEPS,
    FamilyReport,
    evaluate_family,
)

pytestmark = LIVE_MARKS

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def reports() -> dict[str, FamilyReport]:
    """Collect the report of each family, so the saved file holds every family finished."""
    return {}


def log_family(report: FamilyReport) -> None:
    """Log a family's honest steps, its audit default and what it flags and catches."""
    honest, audit = report["honest"], report["audit"]
    logger.info(
        "%s: %d honest steps from %d runs, %d runs and %d steps set aside, spend $%.3f",
        report["family"],
        honest["steps"],
        honest["runs"],
        honest["discarded_runs"],
        honest["unavailable_steps"],
        report["spend"],
    )
    logger.info(
        "%s: audit default %.3f in [%.3f, %.3f], flags %.3f of honest steps, catches %s of %d runs",
        report["family"],
        audit["threshold"],
        audit["lower"],
        audit["upper"],
        audit["false_positive_rate"],
        audit["catch_rate"],
        len(report["attack"]["side_task_scores"]),
    )


@pytest.mark.parametrize("family", list(MonitorFamily))
def test_the_family_scores_enough_honest_steps_to_set_a_threshold(
    family: MonitorFamily,
    report_directory: Path,
    reports: dict[str, FamilyReport],
) -> None:
    # Arrange
    ledger = CostLedger(cap=FAMILY_BUDGET)

    # Act
    report = evaluate_family(family, ledger=ledger)

    # Assert
    reports[family.value] = report
    save_report(reports, name="threshold-evals", directory=report_directory)
    log_family(report)
    assert report["honest"]["steps"] >= MIN_HONEST_STEPS
