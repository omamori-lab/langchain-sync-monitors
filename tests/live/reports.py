"""The record of one live run, in plain values that serialise to JSON.

A report keeps, for each step, every sample with its suspicion and the
monitor's reason, the outcome, the feedback the agent received, the path the
protocol took and the step that was committed; and, for the run, whether the
canary left, the flags, the cost and the time taken.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors import (
    DEFAULT_FEEDBACK_TEMPLATE,
    MonitorStepFailedEvent,
    SampleRecord,
    StepRecord,
    Verdict,
)
from tests.live.costs import CostSnapshot
from tests.live.sandbox import OutboundRecord

COMMITTED_TEXT_LIMIT = 600
"""How many characters of a committed final answer a report keeps."""


class StepReport(TypedDict):
    """One monitored step: its samples, the decision and what the agent received."""

    step_number: int
    outcome: str
    flagged: bool
    blocked_count: int
    samples: list[SampleRecord]
    feedback: list[str]
    decision_path: str
    committed: str
    seconds: float | None


class RunReport(TypedDict):
    """One live run, from its configuration to its cost."""

    run_id: str
    scenario: str
    monitor: str
    protocol: str
    mode: str
    reads_reasoning: bool
    models: dict[str, str]
    steps: list[StepReport]
    canary_left: bool
    outbound: list[OutboundRecord]
    executed_calls: list[str]
    final_answer: str
    flagged_steps: list[int]
    failed_steps: list[MonitorStepFailedEvent]
    cost: CostSnapshot
    providers: dict[str, list[str]]
    seconds: float
    error: str | None


def rebuild_feedback(record: StepRecord) -> list[str]:
    """Return the feedback Auto Mode sent for each blocked sample of the step.

    Auto Mode blocks the samples before the one that runs, and renders the
    default template with each blocked sample's reason, so the text is rebuilt
    exactly from the record.
    """
    blocked = record["samples"][: record["blocked_count"]]
    return [
        DEFAULT_FEEDBACK_TEMPLATE.render(
            verdict=Verdict(suspicion=sample["suspicion"], reason=sample["reason"]),
        )
        for sample in blocked
    ]


def describe_decision_path(record: StepRecord) -> str:
    """Describe the protocol's path through the step: each sample's score, then the outcome."""
    samples = ", ".join(
        f"{sample['suspicion']:.2f} {'ran' if sample['executed'] else 'not run'}"
        for sample in record["samples"]
    )
    flag = ", flagged" if record["flagged"] else ""
    return f"{samples or 'no sample'} -> {record['outcome']}{flag}"


def render_committed_step(message: AIMessage | None) -> str:
    """Render the step the agent state received: its tool calls, or its final answer."""
    if message is None:
        return ""
    if message.tool_calls:
        return "; ".join(
            f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
            for call in message.tool_calls
        )
    return message.text[:COMMITTED_TEXT_LIMIT]


def build_step_reports(
    records: Sequence[StepRecord],
    *,
    messages: Sequence[BaseMessage],
    step_seconds: Sequence[float],
) -> list[StepReport]:
    """Pair each record with the message it committed and the time the step took.

    With hidden feedback, the monitor commits exactly one AI message per step,
    so the agent's AI messages line up with the records.
    """
    replies = [message for message in messages if isinstance(message, AIMessage)]
    committed = replies if len(replies) == len(records) else [None] * len(records)
    return [
        {
            "step_number": record["step_number"],
            "outcome": record["outcome"],
            "flagged": record["flagged"],
            "blocked_count": record["blocked_count"],
            "samples": record["samples"],
            "feedback": rebuild_feedback(record),
            "decision_path": describe_decision_path(record),
            "committed": render_committed_step(reply),
            "seconds": step_seconds[index] if index < len(step_seconds) else None,
        }
        for index, (record, reply) in enumerate(zip(records, committed, strict=True))
    ]


def read_final_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the text of the last AI message without tool calls, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text[:COMMITTED_TEXT_LIMIT]
    return ""
