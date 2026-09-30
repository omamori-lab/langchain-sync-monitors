"""The spans a monitored step opens in LangChain tracers, and what each one carries.

Every step opens a `monitor step` span around the protocol, and the agent's
samples nest in it. Each sample the monitor judges gets a `monitor judgement`
span, around the monitor's own calls, and each Decisions API request a
`monitor classifier` span. Once the protocol has decided, a short
`monitor decision` span carries the outcome in its tags, which a span cannot
change after it starts. The names are fixed, so LangSmith and Langfuse filter
and count spans by name, and what varies goes in metadata under flat keys
that start with `monitor_` [@langsmith2026traces; @langfuse2026]. The
monitor's own model calls inside a judgement carry a fixed name too,
`model_calls.MONITOR_CALL_NAME`.

Every span is tagged `monitor`. The spans below the step span also carry
`ls_agent_type: "middleware"`, which keeps them out of LangSmith's Trajectory
view, as LangSmith advises for guardrails and custom middleware
[@langsmith2026trajectory]. The step span does not, because the agent's own
samples nest in it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import UUID

from langchain_core.utils.uuid import uuid7

from langchain_sync_monitors._langchain import (
    TracedRun,
    TraceSpan,
    TraceValue,
    TraceValues,
    label_monitor_spans,
    open_traced_run,
    open_traced_run_sync,
)
from langchain_sync_monitors.contracts import (
    Monitor,
    SampleRecord,
    StepDecision,
    StepRecord,
    Verdict,
)
from langchain_sync_monitors.records import build_step_record

STEP_SPAN_NAME = "monitor step"
JUDGEMENT_SPAN_NAME = "monitor judgement"
CLASSIFIER_SPAN_NAME = "monitor classifier"
DECISION_SPAN_NAME = "monitor decision"

MONITOR_TAG = "monitor"
"""The tag on every monitor span, so a consumer can drop them all, as with `exclude_tags`."""

FLAGGED_TAG = "monitor:flagged"
"""The tag on the decision span of a step the protocol flagged for audit."""

MONITOR_WORK_METADATA: TraceValues = MappingProxyType({"ls_agent_type": "middleware"})
"""The metadata of the spans that hold only the monitor's own work."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StepIdentity:
    """What names one monitored step, in its record and in every span it opens.

    `step_id` is the step span's run id. It is generated before the span
    starts, so every span of the step can carry it; Langfuse keeps no
    LangChain run ids, so the id is how a step is found again after the run
    [@langfuse2026]. `delegation_id` names the tool call that started a
    subagent, and is `None` in an agent no monitored agent started.
    """

    monitor: str
    agent: str
    step_number: int
    protocol: str
    delegation_id: str | None = None
    step_id: UUID = field(default_factory=uuid7)

    def build_labels(self) -> dict[str, TraceValue]:
        """Return the metadata every span of the step carries."""
        labels: dict[str, TraceValue] = {
            "monitor_name": self.monitor,
            "monitor_agent": self.agent,
            "monitor_step_number": self.step_number,
            "monitor_protocol": self.protocol,
            "monitor_step_id": str(self.step_id),
        }
        if self.delegation_id is not None:
            labels["monitor_delegation_id"] = self.delegation_id
        return labels

    def build_record(self, decision: StepDecision) -> StepRecord:
        """Record the decided step, once, for `monitor_log`, the custom stream and the spans."""
        return build_step_record(
            decision=decision,
            agent=self.agent,
            monitor=self.monitor,
            step_number=self.step_number,
            delegation_id=self.delegation_id,
        )

    def build_step_span(self) -> TraceSpan:
        """Return the step span, whose proposed step is added to its inputs when it ends."""
        return TraceSpan(
            name=STEP_SPAN_NAME,
            inputs={"step_number": self.step_number},
            tags=[MONITOR_TAG],
            run_id=self.step_id,
        )


def find_max_suspicion(samples: Sequence[SampleRecord]) -> float | None:
    """Return the highest suspicion among the samples, or None when the step judged none."""
    return max((sample["suspicion"] for sample in samples), default=None)


def build_step_span_inputs(
    *,
    step_number: int,
    samples: Sequence[SampleRecord],
) -> dict[str, TraceValue]:
    """Return the step span's inputs: the step's number and the step first proposed for it.

    A step with no sample, such as a halt after a subagent was halted, has
    no proposed step.
    """
    proposed_step = samples[0]["proposal"] if samples else None
    return {"step_number": step_number, "proposed_step": proposed_step}


def build_step_span_outputs(record: StepRecord) -> dict[str, TraceValue]:
    """Return the step span's outputs: the decision, and each sample's verdict and fate.

    The samples' proposals stay out: each one is already the output of the
    model call that drew it, and the record in `monitor_log` keeps them all.
    """
    return {
        "outcome": record["outcome"],
        "flagged": record["flagged"],
        "blocked_count": record["blocked_count"],
        "max_suspicion": find_max_suspicion(record["samples"]),
        "samples": [
            {
                "suspicion": sample["suspicion"],
                "reason": sample["reason"],
                "executed": sample["executed"],
            }
            for sample in record["samples"]
        ],
    }


def report_step_to_span(traced_step: TracedRun, *, record: StepRecord) -> None:
    """Hand the committed step to its span: the decision as outputs, the proposal as inputs."""
    traced_step.outputs = build_step_span_outputs(record)
    traced_step.inputs_at_end = build_step_span_inputs(
        step_number=record["step_number"],
        samples=record["samples"],
    )


def build_decision_span(record: StepRecord) -> TraceSpan:
    """Return the decision span, tagged with the outcome, and `monitor:flagged` when flagged.

    Its metadata repeats the outcome for filtering, and the highest
    suspicion when the step judged any sample.
    """
    tags = [MONITOR_TAG, f"{MONITOR_TAG}:{record['outcome']}"]
    if record["flagged"]:
        tags.append(FLAGGED_TAG)
    metadata: dict[str, TraceValue] = {
        **MONITOR_WORK_METADATA,
        "monitor_outcome": record["outcome"],
        "monitor_flagged": record["flagged"],
    }
    max_suspicion = find_max_suspicion(record["samples"])
    if max_suspicion is not None:
        metadata["monitor_max_suspicion"] = max_suspicion
    return TraceSpan(name=DECISION_SPAN_NAME, metadata=metadata, tags=tags)


def build_decision_span_outputs(record: StepRecord) -> dict[str, TraceValue]:
    """Return the decision span's outputs: the outcome, the flag and the highest suspicion."""
    return {
        "outcome": record["outcome"],
        "flagged": record["flagged"],
        "max_suspicion": find_max_suspicion(record["samples"]),
    }


def build_judgement_span(*, sample_number: int, monitor: Monitor) -> TraceSpan:
    """Return the span of one judgement: which sample of the step, judged by which monitor.

    The sample's proposal stays out, since the model call that drew it
    carries it, and the monitor's own calls, nested in the span, carry the
    transcript as the monitor read it.
    """
    return TraceSpan(
        name=JUDGEMENT_SPAN_NAME,
        inputs={"sample_number": sample_number, "monitor": type(monitor).__name__},
        metadata=MONITOR_WORK_METADATA,
        tags=[MONITOR_TAG],
    )


def build_verdict_outputs(verdict: Verdict) -> dict[str, TraceValue]:
    """Return a judgement span's outputs: the verdict's suspicion and reason."""
    return {"suspicion": verdict.suspicion, "reason": verdict.reason}


@contextmanager
def open_step_span_sync(identity: StepIdentity) -> Iterator[TracedRun]:
    """Open the step span for `invoke()`, labelling every monitor span inside it with the step."""
    # The labels are set first, so the step span itself carries them.
    with (
        label_monitor_spans(identity.build_labels()),
        open_traced_run_sync(identity.build_step_span()) as traced_step,
    ):
        yield traced_step


@asynccontextmanager
async def open_step_span(identity: StepIdentity) -> AsyncIterator[TracedRun]:
    """Open the step span for `ainvoke()`, labelling every monitor span inside it with the step."""
    # The labels are set first, so the step span itself carries them.
    with label_monitor_spans(identity.build_labels()):
        async with open_traced_run(identity.build_step_span()) as traced_step:
            yield traced_step


def trace_decision_sync(traced_step: TracedRun, *, record: StepRecord) -> None:
    """Trace the protocol's decision for `invoke()`: a decision span, and the step span's outcome.

    The decision span opens and closes at once, inside the step span.
    """
    with open_traced_run_sync(build_decision_span(record)) as traced_decision:
        traced_decision.outputs = build_decision_span_outputs(record)
    report_step_to_span(traced_step, record=record)


async def trace_decision(traced_step: TracedRun, *, record: StepRecord) -> None:
    """Trace the protocol's decision for `ainvoke()`: a decision span, and the step span's outcome.

    The decision span opens and closes at once, inside the step span.
    """
    async with open_traced_run(build_decision_span(record)) as traced_decision:
        traced_decision.outputs = build_decision_span_outputs(record)
    report_step_to_span(traced_step, record=record)
