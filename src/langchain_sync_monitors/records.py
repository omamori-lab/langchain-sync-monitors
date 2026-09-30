"""Build the step records that go into the agent state's `monitor_log`.

Records pass through checkpointers, so they hold plain values only: outcomes
are strings, sequences are lists, and each sample's proposal is rendered text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

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
    """Record one judged sample, with its proposal rendered for human auditors.

    The rendering holds every channel, malformed tool calls included, so a
    call that could not run still leaves evidence of what the agent tried.
    """
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
    delegation_id: str | None = None,
) -> StepRecord:
    """Record one decided step, keeping every sample the monitor judged as audit evidence.

    A sample is marked executed only when it is the very sample the decision
    runs, so a rejected sample that happens to equal it is not. A subagent's
    record also names its delegation, the tool call that started it.
    """
    record = StepRecord(
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
    if delegation_id is not None:
        record["delegation_id"] = delegation_id
    return record


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


def find_new_subagent_records(
    records: Sequence[StepRecord],
    *,
    agent: str,
) -> list[StepRecord]:
    """Return the records of other agents logged since this agent's last step.

    Those are the steps of the subagents this agent started since then,
    nested ones included, which this agent has not yet answered.
    """
    own_positions = [index for index, record in enumerate(records) if record["agent"] == agent]
    start = own_positions[-1] + 1 if own_positions else 0
    return [record for record in records[start:] if record["agent"] != agent]


def count_blocks(records: Iterable[StepRecord], *, monitor: str) -> int:
    """Return how many attempts one monitor blocked across the records."""
    return sum(record["blocked_count"] for record in records if record["monitor"] == monitor)


def count_blocks_by_monitor(
    records: Iterable[StepRecord],
    *,
    earlier_blocks: Mapping[str, int],
) -> dict[str, int]:
    """Return the blocks each monitor recorded, by label, adding the records to `earlier_blocks`."""
    blocks = Counter(earlier_blocks)
    for record in records:
        blocks[record["monitor"]] += record["blocked_count"]
    return dict(blocks)
