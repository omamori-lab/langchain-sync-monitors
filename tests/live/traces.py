"""Checks on the monitor's spans in a live run, read from a tracer that stays in process.

Every live run passes a `RecordingTracer` in its config, which hears the same
callbacks LangSmith's tracer and Langfuse's handler build their trees from,
and sends nothing anywhere. After the run, `build_trace_report` checks what
`langchain_sync_monitors.spans` promises against the run's `monitor_log`:

- the monitor's spans carry only the four fixed names, and its model calls
  the name `monitor call`;
- every judgement and decision sits in a step, and every classifier request
  and monitor call in a judgement; no other model call sits in a judgement;
- each record has its step span, with one judgement per judged sample and one
  decision, tagged with the outcome and, when flagged, `monitor:flagged`;
- every monitor span ended, under a parent the tracer heard of.

The checks cost nothing: they read the callbacks the run makes anyway.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TypedDict

from langchain_sync_monitors import StepRecord
from langchain_sync_monitors.model_calls import MONITOR_CALL_NAME
from langchain_sync_monitors.spans import (
    CLASSIFIER_SPAN_NAME,
    DECISION_SPAN_NAME,
    FLAGGED_TAG,
    JUDGEMENT_SPAN_NAME,
    MONITOR_TAG,
    STEP_SPAN_NAME,
)
from tests.support.tracing import MONITOR_SPAN_PREFIX, RecordedRun, RecordingTracer

MONITOR_SPAN_NAMES = frozenset(
    {STEP_SPAN_NAME, JUDGEMENT_SPAN_NAME, CLASSIFIER_SPAN_NAME, DECISION_SPAN_NAME},
)
MODEL_RUN_TYPES = frozenset({"chat_model", "llm"})
SPANS_INSIDE_A_STEP = frozenset({JUDGEMENT_SPAN_NAME, DECISION_SPAN_NAME})
RUNS_INSIDE_A_JUDGEMENT = frozenset({CLASSIFIER_SPAN_NAME, MONITOR_CALL_NAME})

type StepKey = tuple[str, str | None, int]
"""What names one step across a run: its agent, its delegation id, if any, and its number."""


class JudgementReport(TypedDict):
    """One judgement span: the step it belongs to, its classifier answers and its model calls."""

    agent: str
    step_number: int
    classifier_scores: list[float]
    monitor_calls: int


class TraceReport(TypedDict):
    """What the monitor's spans showed: how many of each, each judgement, and any problem."""

    span_counts: dict[str, int]
    judgements: list[JudgementReport]
    problems: list[str]


def find_nearest_ancestor(
    tracer: RecordingTracer,
    run: RecordedRun,
    *,
    name: str,
) -> RecordedRun | None:
    """Return the run's nearest ancestor with this name, or `None` when it has none."""
    while run.parent_run_id is not None and run.parent_run_id in tracer.runs:
        run = tracer.runs[run.parent_run_id]
        if run.name == name:
            return run
    return None


def is_monitor_run(run: RecordedRun) -> bool:
    """Tell whether the monitor opened the run: one of its spans, or one of its model calls."""
    return run.name.startswith(MONITOR_SPAN_PREFIX)


def find_unknown_names(tracer: RecordingTracer) -> list[str]:
    """Return the monitor span names other than the four fixed ones."""
    return [
        f"unknown monitor span name {run.name!r}"
        for run in tracer.find_monitor_spans()
        if run.name not in MONITOR_SPAN_NAMES
    ]


def find_misplaced_run(tracer: RecordingTracer, run: RecordedRun) -> str | None:
    """Describe where the run sits if that is not where the monitor's spans promise."""
    in_judgement = find_nearest_ancestor(tracer, run, name=JUDGEMENT_SPAN_NAME) is not None
    if run.name in SPANS_INSIDE_A_STEP and not find_nearest_ancestor(
        tracer,
        run,
        name=STEP_SPAN_NAME,
    ):
        return f"{run.name!r} outside a step"
    if run.name in RUNS_INSIDE_A_JUDGEMENT and not in_judgement:
        return f"{run.name!r} outside a judgement"
    if run.run_type in MODEL_RUN_TYPES and run.name != MONITOR_CALL_NAME and in_judgement:
        return f"model call {run.name!r} in a judgement, not named {MONITOR_CALL_NAME!r}"
    return None


def find_misplaced_runs(tracer: RecordingTracer) -> list[str]:
    """Return every run that sits where the monitor's spans promise it does not."""
    problems = (find_misplaced_run(tracer, run) for run in tracer.read_runs())
    return [problem for problem in problems if problem is not None]


def find_unfinished_runs(tracer: RecordingTracer) -> list[str]:
    """Return the monitor's runs that never ended, or whose parent the tracer never heard of."""
    problems: list[str] = []
    for run in filter(is_monitor_run, tracer.read_runs()):
        if not run.ended:
            problems.append(f"{run.name!r} never ended")
        if run.parent_run_id is not None and run.parent_run_id not in tracer.runs:
            problems.append(f"{run.name!r} has a parent the tracer never heard of")
    return problems


def read_step_key(run: RecordedRun) -> StepKey:
    """Read which step a step span belongs to from its metadata."""
    delegation_id = run.metadata.get("monitor_delegation_id")
    return (
        str(run.metadata.get("monitor_agent")),
        None if delegation_id is None else str(delegation_id),
        int(run.metadata.get("monitor_step_number", -1)),
    )


def read_record_key(record: StepRecord) -> StepKey:
    """Return the key of the step a record belongs to."""
    return (record["agent"], record.get("delegation_id"), record["step_number"])


def list_inner_spans(tracer: RecordingTracer, step: RecordedRun, *, name: str) -> list[RecordedRun]:
    """Return the spans with this name whose nearest step span is `step`."""
    return [
        run
        for run in tracer.find_runs(name)
        if find_nearest_ancestor(tracer, run, name=STEP_SPAN_NAME) is step
    ]


def check_decision_tags(decision: RecordedRun, *, record: StepRecord) -> list[str]:
    """Return a problem when the decision span's tags disagree with the record."""
    expected = {MONITOR_TAG, f"{MONITOR_TAG}:{record['outcome']}"}
    if record["flagged"]:
        expected.add(FLAGGED_TAG)
    if set(decision.tags) != expected:
        return [f"step {read_record_key(record)}: decision tags {decision.tags}, not {expected}"]
    return []


def check_step_span(tracer: RecordingTracer, step: RecordedRun, *, record: StepRecord) -> list[str]:
    """Check one step span against its record: a judgement per sample and one decision."""
    key = read_record_key(record)
    judgements = list_inner_spans(tracer, step, name=JUDGEMENT_SPAN_NAME)
    decisions = list_inner_spans(tracer, step, name=DECISION_SPAN_NAME)
    problems: list[str] = []
    if len(judgements) != len(record["samples"]):
        problems.append(
            f"step {key}: {len(judgements)} judgements for {len(record['samples'])} samples",
        )
    if len(decisions) != 1:
        return [*problems, f"step {key}: {len(decisions)} decision spans"]
    return [*problems, *check_decision_tags(decisions[0], record=record)]


def check_step_spans(tracer: RecordingTracer, *, records: Sequence[StepRecord]) -> list[str]:
    """Check that each record has exactly one step span, and that span against the record."""
    steps_by_key: dict[StepKey, list[RecordedRun]] = {}
    for step in tracer.find_runs(STEP_SPAN_NAME):
        steps_by_key.setdefault(read_step_key(step), []).append(step)
    problems: list[str] = []
    for record in records:
        steps = steps_by_key.get(read_record_key(record), [])
        if len(steps) != 1:
            problems.append(f"step {read_record_key(record)}: {len(steps)} step spans")
            continue
        problems.extend(check_step_span(tracer, steps[0], record=record))
    return problems


def read_classifier_scores(classifiers: Iterable[RecordedRun]) -> list[float]:
    """Return every probability of yes the classifier requests answered, in order."""
    scores: list[float] = []
    for classifier in classifiers:
        outputs = classifier.outputs if isinstance(classifier.outputs, dict) else {}
        answers = outputs.get("answers", {})
        scores.extend(float(value) for value in answers.values())
    return scores


def build_judgement_report(tracer: RecordingTracer, judgement: RecordedRun) -> JudgementReport:
    """Report one judgement: its step, its classifier answers and how many model calls it made."""
    inner = [
        run
        for run in tracer.read_runs()
        if find_nearest_ancestor(tracer, run, name=JUDGEMENT_SPAN_NAME) is judgement
    ]
    step = find_nearest_ancestor(tracer, judgement, name=STEP_SPAN_NAME)
    agent, _, step_number = read_step_key(step) if step else ("", None, -1)
    return {
        "agent": agent,
        "step_number": step_number,
        "classifier_scores": read_classifier_scores(
            run for run in inner if run.name == CLASSIFIER_SPAN_NAME
        ),
        "monitor_calls": sum(run.name == MONITOR_CALL_NAME for run in inner),
    }


def build_trace_report(tracer: RecordingTracer, *, records: Sequence[StepRecord]) -> TraceReport:
    """Count the monitor's spans, report each judgement, and list every broken promise."""
    counts = Counter(run.name for run in tracer.read_runs() if is_monitor_run(run))
    return {
        "span_counts": dict(sorted(counts.items())),
        "judgements": [
            build_judgement_report(tracer, judgement)
            for judgement in tracer.find_runs(JUDGEMENT_SPAN_NAME)
        ],
        "problems": [
            *find_unknown_names(tracer),
            *find_misplaced_runs(tracer),
            *find_unfinished_runs(tracer),
            *check_step_spans(tracer, records=records),
        ],
    }
