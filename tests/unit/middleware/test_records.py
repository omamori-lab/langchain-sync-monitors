"""Step records hold plain values and mark exactly the executed sample."""

from __future__ import annotations

from typing import get_args

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Outcome,
    OutcomeName,
    Sample,
    StepDecision,
    StepRecord,
    Verdict,
)
from langchain_sync_monitors.records import OUTCOME_NAMES, build_step_record, find_monitor_records
from tests.support.chat_models import build_tool_call_message


def build_sample(*, content: str, suspicion: float) -> Sample:
    proposal = AIMessage(content=content)
    return Sample(
        response=ModelResponse(result=[proposal]),
        proposal=proposal,
        verdict=Verdict(suspicion=suspicion, reason=f"scored {suspicion}"),
    )


def build_record(*, agent: str, monitor: str, step_number: int) -> StepRecord:
    return StepRecord(
        agent=agent,
        monitor=monitor,
        step_number=step_number,
        outcome="allowed",
        flagged=False,
        blocked_count=0,
        samples=[],
    )


@pytest.fixture
def resampled_decision() -> StepDecision:
    rejected = build_sample(content="Send the key off-site.", suspicion=0.9)
    chosen = build_sample(content="Summarise the report.", suspicion=0.1)
    return StepDecision(
        outcome=Outcome.RESAMPLED,
        response=chosen.response,
        samples=(rejected, chosen),
        executed_sample=chosen,
        flagged=False,
    )


def test_a_record_keeps_every_sample_and_marks_the_executed_one(
    resampled_decision: StepDecision,
) -> None:
    # Act
    record = build_step_record(
        decision=resampled_decision, agent="main", monitor="m", step_number=3
    )

    # Assert
    assert record["outcome"] == "resampled"
    assert record["step_number"] == 3
    assert [sample["executed"] for sample in record["samples"]] == [False, True]
    assert [sample["suspicion"] for sample in record["samples"]] == [0.9, 0.1]
    assert "Send the key off-site." in record["samples"][0]["proposal"]


def test_an_equal_but_distinct_sample_is_not_marked_executed() -> None:
    # Arrange
    first = build_sample(content="same", suspicion=0.2)
    twin = build_sample(content="same", suspicion=0.2)
    decision = StepDecision(
        outcome=Outcome.RESAMPLED,
        response=twin.response,
        samples=(first, twin),
        executed_sample=twin,
        flagged=False,
    )

    # Act
    record = build_step_record(decision=decision, agent="main", monitor="m", step_number=1)

    # Assert
    assert [sample["executed"] for sample in record["samples"]] == [False, True]


def test_a_record_counts_blocked_attempts_and_renders_tool_calls() -> None:
    # Arrange
    proposal = build_tool_call_message(tool_name="http_post", call_id="c1", arguments={"url": "x"})
    blocked = Sample(
        response=ModelResponse(result=[proposal]),
        proposal=proposal,
        verdict=Verdict(suspicion=0.95, reason="exfiltration"),
    )
    decision = StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[AIMessage("stopped")]),
        samples=(blocked,),
        executed_sample=None,
        flagged=True,
        blocked_attempts=(BlockedAttempt(proposal=proposal, feedback="no"),),
    )

    # Act
    record = build_step_record(decision=decision, agent="coder", monitor="m", step_number=1)

    # Assert
    assert record["blocked_count"] == 1
    assert record["flagged"] is True
    assert record["samples"][0]["executed"] is False
    assert '<tool_call name="http_post">' in record["samples"][0]["proposal"]


def test_records_hold_only_plain_values(resampled_decision: StepDecision) -> None:
    # Act
    record = build_step_record(
        decision=resampled_decision, agent="main", monitor="m", step_number=1
    )

    # Assert
    assert type(record["outcome"]) is str
    assert type(record["samples"]) is list


def test_every_outcome_has_a_stored_name() -> None:
    # Act
    names = {OUTCOME_NAMES[outcome] for outcome in Outcome}

    # Assert
    assert names == set(get_args(OutcomeName))


def test_only_the_records_of_one_monitor_and_agent_are_found() -> None:
    # Arrange
    records = [
        build_record(agent="main", monitor="monitor", step_number=1),
        build_record(agent="researcher", monitor="monitor", step_number=1),
        build_record(agent="main", monitor="guard", step_number=1),
        build_record(agent="main", monitor="monitor", step_number=2),
    ]

    # Act
    found = find_monitor_records(records, monitor="monitor", agent="main")

    # Assert
    assert [record["step_number"] for record in found] == [1, 2]
    assert all(record["agent"] == "main" and record["monitor"] == "monitor" for record in found)
