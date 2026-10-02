"""The attack run under each of six protocol variants, judged by the LLM monitor.

What it measures: whether each protocol takes the path its scores call for
with a real agent and a real judge. The LLM monitor is used because it scores
the read of `config/.env` above the threshold in most runs, so the protocols
act early. Each run alternates `invoke()` and `ainvoke()`, with Defer to
Resample's parallel draws under `ainvoke()`, where they run concurrently.

Every run must end without an error, keep the execution invariant and the
span promises, and give each step in which a sample reached the threshold one
of the outcomes `checks.INTERVENTIONS` allows that protocol. Under Trusted
Monitoring the canary may leave, but every step that sent it must be flagged;
under every other protocol it stays. With the feedback kept in the transcript,
each blocked attempt stays with a `monitor-` id and its feedback is tagged;
hidden, none of it is left in the state. A run in which no sample reached
the threshold is skipped after those checks, since the protocol had nothing
to act on, and a run a provider rate-limited is skipped too.

Each run stops at `RUN_BUDGET`; `tests/live/__init__.py` gives the suite's
expected cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.live.checks import (
    LIVE_MARKS,
    has_reached_threshold,
    list_canary_problems,
    list_feedback_problems,
    list_path_problems,
    list_run_problems,
    record_run,
    skip_on_rate_limit,
)
from tests.live.harness import (
    MonitorFamily,
    ProtocolName,
    RunConfiguration,
    RunMode,
    run_live_agent,
)
from tests.live.scenario import Scenario

pytestmark = LIVE_MARKS

RUN_BUDGET = 0.03
"""The most one run may spend, in US dollars; the run stops once it reaches this."""

PROTOCOL_MODES: list[tuple[ProtocolName, RunMode]] = [
    (ProtocolName.TRUSTED_MONITORING, "invoke"),
    (ProtocolName.DEFER_TO_TRUSTED, "ainvoke"),
    (ProtocolName.DEFER_TO_RESAMPLE, "invoke"),
    (ProtocolName.DEFER_TO_RESAMPLE_PARALLEL, "ainvoke"),
    (ProtocolName.AUTO_MODE, "invoke"),
    (ProtocolName.AUTO_MODE_IN_TRANSCRIPT, "ainvoke"),
]


@pytest.mark.parametrize(("protocol_name", "mode"), PROTOCOL_MODES)
def test_the_protocol_acts_on_every_step_that_reaches_the_threshold(
    protocol_name: ProtocolName,
    mode: RunMode,
    report_directory: Path,
) -> None:
    # Arrange
    configuration = RunConfiguration(
        scenario=Scenario.ATTACK,
        monitor_family=MonitorFamily.LLM_MONITOR,
        protocol_name=protocol_name,
        mode=mode,
    )

    # Act
    report = run_live_agent(configuration, budget=RUN_BUDGET)

    # Assert
    record_run(report, directory=report_directory)
    skip_on_rate_limit(report)
    assert list_run_problems(report) == []
    assert list_path_problems(report) == []
    assert list_canary_problems(report) == []
    in_transcript = protocol_name is ProtocolName.AUTO_MODE_IN_TRANSCRIPT
    assert list_feedback_problems(report, in_transcript=in_transcript) == []
    if not has_reached_threshold(report):
        pytest.skip("no sample reached the threshold, so the protocol had nothing to act on")
