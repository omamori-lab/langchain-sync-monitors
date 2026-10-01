"""Build the step records that go into the agent state's `monitor_log`, and read them back.

Records pass through checkpointers, so they hold plain values only: outcomes
are strings, sequences are lists, and each sample's proposal is rendered text.

A record is this agent's own when it names this agent and this agent's
delegation, the tool call that started it, or no delegation in an agent no
monitored agent started. A subagent that shares its parent's name, such as a
fork or a compiled subagent left at the default `agent_name`, still records a
delegation of its own, so its records are never the parent's.

Every record read from the state is checked to be a whole `StepRecord` with
counts of zero or more. The records a tool returns are checked where they
are written, by `returned_records`, so a record that fails here came from
elsewhere, such as `update_state` or an older checkpoint, and raises
`MonitorError` rather than being skipped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sized

from pydantic import TypeAdapter, ValidationError

from langchain_sync_monitors._langchain import MONITOR_LOG_KEY
from langchain_sync_monitors.contracts import (
    Outcome,
    OutcomeName,
    Sample,
    SampleRecord,
    StepDecision,
    StepRecord,
)
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.options import name_type
from langchain_sync_monitors.transcript import render_proposal_for_audit

STEP_RECORD_ADAPTER = TypeAdapter(StepRecord)
"""Checks the shape of a record read from the state or returned by a tool."""

RECORD_NAME_FIELDS: dict[str, tuple[str, type[str] | type[int]]] = {
    "agent": ("agent", str),
    "monitor": ("monitor", str),
    "step_number": ("step number", int),
    "outcome": ("outcome", str),
    "delegation_id": ("delegation id", str),
}
"""The fields that name a record, each with the words that render it and the type it holds."""

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
    The suspicion is stored as a plain `float`, since a monitor may score
    with a number of another type, such as numpy's `float32`, which a
    checkpointer may not store and a strict read refuses.
    """
    return SampleRecord(
        suspicion=float(sample.verdict.suspicion),
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
    record also names its delegation, the tool call that started it. The
    flag is stored as a plain `bool`: a protocol that compares a numpy score
    with its threshold gets numpy's `bool`, which is not one.
    """
    record = StepRecord(
        agent=agent,
        monitor=monitor,
        step_number=step_number,
        outcome=OUTCOME_NAMES[decision.outcome],
        flagged=bool(decision.flagged),
        blocked_count=len(decision.blocked_attempts),
        samples=[
            build_sample_record(sample, executed=sample is decision.executed_sample)
            for sample in decision.samples
        ],
    )
    if delegation_id is not None:
        record["delegation_id"] = delegation_id
    return record


def is_own_record(record: StepRecord, *, agent: str, delegation_id: str | None) -> bool:
    """Tell whether a record is a step of this agent in this delegation.

    `delegation_id` is the id of the tool call that started the agent, or
    None in an agent that no monitored agent started. A subagent's records
    carry the id of the call that started it, so they differ from its
    parent's even when the two share a name.
    """
    return record["agent"] == agent and record.get("delegation_id") == delegation_id


def find_monitor_records(
    records: Iterable[StepRecord],
    *,
    monitor: str,
    agent: str,
    delegation_id: str | None,
) -> tuple[StepRecord, ...]:
    """Return the records one monitor wrote for this agent in this delegation, oldest first."""
    return tuple(
        record
        for record in records
        if record["monitor"] == monitor
        and is_own_record(record, agent=agent, delegation_id=delegation_id)
    )


def validate_step_record(value: object) -> StepRecord:
    """Return the value as a `StepRecord`, raising `ValueError` unless it is a whole one.

    The check is strict, so a count given as a bool, a float or a string is
    refused rather than converted, and both counts must be zero or more: a
    negative block count would lower Auto Mode's total.
    """
    record = STEP_RECORD_ADAPTER.validate_python(value, strict=True)
    if record["step_number"] < 0 or record["blocked_count"] < 0:
        message = "step_number and blocked_count must be zero or more"
        raise ValueError(message)
    return record


def render_value(value: object) -> str:
    """Name a value for an error or a warning without quoting any text it holds.

    A record holds the transcript, in each sample's proposal and reason, and
    no log line or error message may. So a mapping is named as a record, by
    the fields that name one, its agent, monitor, step number, outcome and
    delegation id, and by its number of samples. A field is quoted only when
    it holds the type a record gives it, and is otherwise named by its type.
    A mapping with none of those fields, and any other value, is named by its
    type, and by its length where it has one.
    """
    if isinstance(value, Mapping):
        fields = render_record_fields(value)
        if fields:
            return f"{name_value_type(value)} with {', '.join(fields)}"
    return describe_type(value)


def render_record_fields(record: Mapping[object, object]) -> list[str]:
    """Return the fields that name a record, rendered, then its number of samples."""
    fields = [
        render_record_field(record[key], words=words, field_type=field_type)
        for key, (words, field_type) in RECORD_NAME_FIELDS.items()
        if key in record
    ]
    if "samples" in record:
        fields.append(render_sample_count(record["samples"]))
    return fields


def render_record_field(value: object, *, words: str, field_type: type) -> str:
    """Quote a field that holds the type a record gives it, and name any other by its type."""
    # Python counts a bool as an int, but no record holds one as its step number.
    if isinstance(value, field_type) and not isinstance(value, bool):
        return f"{words} {value!r}"
    return f"{words} that is {describe_type(value)}"


def render_sample_count(samples: object) -> str:
    """Count a record's samples, or name its samples by their type when they are no list."""
    if isinstance(samples, list):
        return f"{len(samples)} sample(s)"
    return f"samples that are {describe_type(samples)}"


def describe_type(value: object) -> str:
    """Name a value by its type, and its length where it has one, as in `a str of length 12`."""
    if value is None:
        return "None"
    if isinstance(value, Sized):
        return f"{name_value_type(value)} of length {len(value)}"
    return name_value_type(value)


def name_value_type(value: object) -> str:
    """Name a value's type with its article, as in `a dict` or `an int`."""
    type_name = name_type(type(value))
    article = "an" if type_name[0].lower() in "aeiou" else "a"
    return f"{article} {type_name}"


def read_step_records(state: object) -> list[StepRecord]:
    """Return the records in an agent state, raising `MonitorError` for one that is malformed.

    A record that is not a whole `StepRecord` is never skipped: skipping a
    halt, or a block that Auto Mode counts, would fail open. The error names
    the record, as `render_value` does, and its position.
    """
    value = state.get(MONITOR_LOG_KEY) if isinstance(state, Mapping) else None
    if value is None:
        return []
    if not isinstance(value, list):
        message = f"{MONITOR_LOG_KEY} must be a list of step records, got {render_value(value)}."
        raise MonitorError(message)
    return [read_stored_record(item, position=position) for position, item in enumerate(value)]


def read_stored_record(item: object, *, position: int) -> StepRecord:
    """Return one record of the state's log, raising `MonitorError` that names it if malformed.

    The error is raised from None: pydantic's own error quotes the input, a
    sample's proposal included, and a traceback would print it. The message
    names the fields at fault instead.
    """
    try:
        return validate_step_record(item)
    except ValueError as error:
        message = (
            f"{MONITOR_LOG_KEY}[{position}] is not a step record the monitor can read "
            f"({describe_record_error(error)}): {render_value(item)}. The monitor skips no "
            "record, since that could hide a halt; repair or remove it, with update_state."
        )
        raise MonitorError(message) from None


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


def describe_record_error(error: ValueError) -> str:
    """Name what makes a record unreadable: the first fields that fail, or the rule it breaks."""
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(str(part) for part in problem['loc']) or 'record'}: {problem['msg']}"
            for problem in error.errors()[:3]
        ]
        return "; ".join(problems)
    return str(error)
