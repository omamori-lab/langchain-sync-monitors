"""The spans a monitored step opens in LangChain tracers, and what each one carries.

Every step opens a `monitor step` span around the protocol, and the agent's
samples nest in it. Each rating gets a `monitor judgement` span, around the
monitor's own calls: `TracedMonitor` and `AsyncTracedMonitor` wrap the
monitor's twins and open it around each `rate`, numbering the step's
ratings in the order they start. Once the protocol has decided, a short
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

import itertools
from collections.abc import AsyncGenerator, Generator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import UUID

import hyoshigi
from hyoshigi import Action, ProtocolDecision, SuspicionScore, Trajectory
from langchain_core.utils.uuid import uuid7

from langchain_sync_monitors._langchain import (
    TracedRun,
    TraceSpan,
    TraceValue,
    TraceValues,
    add_step_metadata_to_spans,
    open_traced_run,
    open_traced_run_sync,
)
from langchain_sync_monitors.context_values import set_context_value
from langchain_sync_monitors.contracts import SampleRecord, StepRecord
from langchain_sync_monitors.monitors import MonitorConfiguration
from langchain_sync_monitors.records import build_step_record

STEP_SPAN_NAME = "monitor step"
JUDGEMENT_SPAN_NAME = "monitor judgement"
DECISION_SPAN_NAME = "monitor decision"

MONITOR_TAG = "monitor"
"""The tag on every monitor span, so a consumer can drop them all, as with `exclude_tags`."""

FLAGGED_TAG = "monitor:flagged"
"""The tag on the decision span of a step flagged for a person to review."""

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

    def build_step_metadata(self) -> dict[str, TraceValue]:
        """Return the metadata every span of the step carries: its names and its span's id."""
        return {**self.build_name_metadata(), "monitor_step_id": str(self.step_id)}

    def build_name_metadata(self) -> dict[str, TraceValue]:
        """Return the metadata that names the step: monitor, agent, step number, protocol."""
        metadata: dict[str, TraceValue] = {
            "monitor_label": self.monitor,
            "monitor_agent": self.agent,
            "monitor_step_number": self.step_number,
            "monitor_protocol": self.protocol,
        }
        if self.delegation_id is not None:
            metadata["monitor_delegation_id"] = self.delegation_id
        return metadata

    def build_record(
        self,
        decisions: Sequence[ProtocolDecision],
        *,
        flag_threshold: float | None,
    ) -> StepRecord:
        """Record the decided step, once, for `monitor_log`, the custom stream and the spans."""
        return build_step_record(
            decisions,
            agent=self.agent,
            monitor=self.monitor,
            step_number=self.step_number,
            delegation_id=self.delegation_id,
            flag_threshold=flag_threshold,
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


judgement_numbers: ContextVar[Iterator[int]] = ContextVar("judgement_numbers")
"""The numbers of the step's ratings, in the order they start; `counting_judgements` sets it."""


@contextmanager
def counting_judgements() -> Generator[None, None, None]:
    """Number the ratings made in the block from 1, in the order they start."""
    with set_context_value(judgement_numbers, value=itertools.count(1)):
        yield


def build_judgement_span(*, monitor: object) -> TraceSpan:
    """Return the span of one rating: its number in the step, and the class of the twin rating.

    The action stays out, since the model call that drew it carries it, and
    the monitor's own calls, nested in the span, carry what the monitor read.
    """
    return TraceSpan(
        name=JUDGEMENT_SPAN_NAME,
        inputs={
            "sample_number": next(judgement_numbers.get()),
            "monitor_class": type(monitor).__name__,
        },
        metadata=MONITOR_WORK_METADATA,
        tags=[MONITOR_TAG],
    )


def build_score_outputs(score: SuspicionScore) -> dict[str, TraceValue]:
    """Return a judgement span's outputs: the score's value, its explanation, and whether it failed
    closed.
    """
    return {
        "suspicion": score.value,
        "reason": score.explanation,
        "failed_closed": score.failed_closed,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class TracedMonitor(hyoshigi.Monitor):
    """The sync twin with a `monitor judgement` span around each of its ratings."""

    inner: hyoshigi.Monitor

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the inner twin, inside its judgement span."""
        with open_traced_run_sync(build_judgement_span(monitor=self.inner)) as traced_judgement:
            score = self.inner.rate(action, trajectory=trajectory)
            traced_judgement.outputs = build_score_outputs(score)
        return score


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncTracedMonitor(hyoshigi.AsyncMonitor):
    """The async twin with a `monitor judgement` span around each of its ratings."""

    inner: hyoshigi.AsyncMonitor

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the inner twin, inside its judgement span."""
        async with open_traced_run(build_judgement_span(monitor=self.inner)) as traced_judgement:
            score = await self.inner.rate(action, trajectory=trajectory)
            traced_judgement.outputs = build_score_outputs(score)
        return score


@dataclass(frozen=True, slots=True, kw_only=True)
class TracedTwins:
    """A monitor's twins, each with a judgement span around its ratings."""

    monitor: TracedMonitor
    async_monitor: AsyncTracedMonitor


def trace_monitors(monitor: MonitorConfiguration) -> TracedTwins:
    """Return the monitor's twins, each with a judgement span around its ratings."""
    twins = monitor.twins
    return TracedTwins(
        monitor=TracedMonitor(inner=twins.monitor),
        async_monitor=AsyncTracedMonitor(inner=twins.async_monitor),
    )


@contextmanager
def open_step_span_sync(identity: StepIdentity) -> Generator[TracedRun, None, None]:
    """Open the step span for `invoke()`, giving every monitor span in it the step's metadata."""
    # The step's metadata is set first, so the step span itself carries it.
    with (
        add_step_metadata_to_spans(identity.build_step_metadata()),
        open_traced_run_sync(identity.build_step_span()) as traced_step,
    ):
        yield traced_step


@asynccontextmanager
async def open_step_span(identity: StepIdentity) -> AsyncGenerator[TracedRun, None]:
    """Open the step span for `ainvoke()`, giving every monitor span in it the step's metadata."""
    # The step's metadata is set first, so the step span itself carries it.
    with add_step_metadata_to_spans(identity.build_step_metadata()):
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


def trace_answer_sync(identity: StepIdentity, *, record: StepRecord) -> None:
    """Trace a person's answer to an audit for `invoke()`, as a decision span of its own.

    The span opens and closes at once, under the hook that applied the
    answer. It carries the step's names but not its step span's id, which
    belongs to the model call that held the step.
    """
    with (
        add_step_metadata_to_spans(identity.build_name_metadata()),
        open_traced_run_sync(build_decision_span(record)) as traced_decision,
    ):
        traced_decision.outputs = build_decision_span_outputs(record)


async def trace_answer(identity: StepIdentity, *, record: StepRecord) -> None:
    """Trace a person's answer to an audit for `ainvoke()`, as `trace_answer_sync` does."""
    with add_step_metadata_to_spans(identity.build_name_metadata()):
        async with open_traced_run(build_decision_span(record)) as traced_decision:
            traced_decision.outputs = build_decision_span_outputs(record)
