"""Build the step records that go into the agent state's `monitor_log`.

Records pass through checkpointers, so they hold plain values only: outcomes
are strings, sequences are lists, and each sample's proposal is rendered text.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_sync_monitors.contracts import (
    Outcome,
    OutcomeName,
    Sample,
    SampleRecord,
    StepDecision,
    StepRecord,
)
from langchain_sync_monitors.transcript import render_proposal_for_audit

OUTCOME_NAMES: dict[Outcome, OutcomeName] = {
    Outcome.ALLOWED: "allowed",
    Outcome.RESAMPLED: "resampled",
    Outcome.DEFERRED_TO_TRUSTED: "deferred_to_trusted",
    Outcome.STEERED: "steered",
    Outcome.HALTED: "halted",
}
"""Each outcome as the plain string a record stores."""


def build_sample_record(sample: Sample, *, executed: bool) -> SampleRecord:
    """Record one judged sample, with its proposal rendered for human auditors."""
    return SampleRecord(
        suspicion=sample.verdict.suspicion,
        reason=sample.verdict.reason,
        proposal=render_proposal_for_audit(sample.proposal),
        executed=executed,
    )


def build_step_record(
    *,
    decision: StepDecision,
    agent: str,
    monitor: str,
    step_number: int,
) -> StepRecord:
    """Record one decided step, keeping every sample the monitor judged as audit evidence.

    A sample is marked executed only when it is the very sample the decision
    runs, so a rejected sample that happens to equal it is not.
    """
    return StepRecord(
        agent=agent,
        monitor=monitor,
        step_number=step_number,
        outcome=OUTCOME_NAMES[decision.outcome],
        flagged=decision.flagged,
        blocked_count=len(decision.blocked_attempts),
        samples=[
            build_sample_record(sample, executed=sample is decision.executed_sample)
            for sample in decision.samples
        ],
    )


def find_monitor_records(
    records: Iterable[StepRecord],
    *,
    monitor: str,
    agent: str,
) -> tuple[StepRecord, ...]:
    """Return the records one monitor wrote for one agent, oldest first."""
    return tuple(
        record for record in records if record["monitor"] == monitor and record["agent"] == agent
    )
