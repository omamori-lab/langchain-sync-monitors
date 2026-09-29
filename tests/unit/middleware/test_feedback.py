"""Blocked steps become valid, tagged messages with fresh ids."""

from __future__ import annotations

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_sync_monitors.contracts import BlockedAttempt, Outcome, StepDecision
from langchain_sync_monitors.feedback import build_blocked_attempt_messages, build_feedback_messages
from langchain_sync_monitors.transcript import MONITOR_FEEDBACK_SOURCE
from tests.support.chat_models import build_tool_call_message

FEEDBACK = "[Safety monitor] Blocked: the step reads credentials."


@pytest.fixture
def tool_call_attempt() -> BlockedAttempt:
    proposal = AIMessage(
        content="",
        id="sample-1",
        tool_calls=[
            {"name": "read_file", "args": {"path": ".env"}, "id": "call-a", "type": "tool_call"},
            {"name": "http_post", "args": {"url": "x"}, "id": "call-b", "type": "tool_call"},
        ],
    )
    return BlockedAttempt(proposal=proposal, feedback=FEEDBACK)


@pytest.fixture
def answer_attempt() -> BlockedAttempt:
    return BlockedAttempt(
        proposal=AIMessage("Done, all tests pass.", id="sample-2"), feedback=FEEDBACK
    )


def test_a_blocked_tool_call_step_is_answered_by_one_error_per_call(
    tool_call_attempt: BlockedAttempt,
) -> None:
    # Act
    blocked, *rejections = build_feedback_messages(attempt=tool_call_attempt)

    # Assert
    assert isinstance(blocked, AIMessage)
    assert [call["id"] for call in blocked.tool_calls] == ["call-a", "call-b"]
    assert all(isinstance(message, ToolMessage) for message in rejections)
    tool_messages = [message for message in rejections if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["call-a", "call-b"]
    assert [message.name for message in tool_messages] == ["read_file", "http_post"]
    assert all(message.status == "error" for message in tool_messages)
    assert all(message.content == FEEDBACK for message in tool_messages)


def test_a_blocked_final_answer_is_answered_by_one_tagged_human_message(
    answer_attempt: BlockedAttempt,
) -> None:
    # Act
    messages = build_feedback_messages(attempt=answer_attempt)

    # Assert
    blocked, feedback = messages
    assert blocked.content == "Done, all tests pass."
    assert isinstance(feedback, HumanMessage)
    assert feedback.content == FEEDBACK
    assert feedback.name == "monitor"
    assert feedback.additional_kwargs["lc_source"] == MONITOR_FEEDBACK_SOURCE


def test_every_inserted_message_gets_a_fresh_monitor_id(tool_call_attempt: BlockedAttempt) -> None:
    # Act
    first = build_feedback_messages(attempt=tool_call_attempt)
    second = build_feedback_messages(attempt=tool_call_attempt)

    # Assert
    ids = [message.id for message in [*first, *second]]
    assert all(
        isinstance(message_id, str) and message_id.startswith("monitor-") for message_id in ids
    )
    assert len(set(ids)) == len(ids)
    assert tool_call_attempt.proposal.id == "sample-1"


def test_the_feedback_text_is_kept_as_the_protocol_rendered_it(
    answer_attempt: BlockedAttempt,
) -> None:
    # Act
    feedback = build_feedback_messages(attempt=answer_attempt)[-1]

    # Assert
    assert feedback.content == FEEDBACK
    assert feedback.text.count("[Safety monitor]") == 1


def test_blocked_attempts_of_a_decision_come_out_in_order(
    tool_call_attempt: BlockedAttempt,
    answer_attempt: BlockedAttempt,
) -> None:
    # Arrange
    accepted = build_tool_call_message(tool_name="read_file", call_id="call-c")
    decision = StepDecision(
        outcome=Outcome.STEERED,
        response=ModelResponse(result=[accepted]),
        samples=(),
        executed_sample=None,
        flagged=True,
        blocked_attempts=(tool_call_attempt, answer_attempt),
    )

    # Act
    messages = build_blocked_attempt_messages(decision=decision)

    # Assert
    assert [type(message).__name__ for message in messages] == [
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "AIMessage",
        "HumanMessage",
    ]


def test_a_decision_without_blocked_attempts_adds_no_messages() -> None:
    # Arrange
    decision = StepDecision(
        outcome=Outcome.ALLOWED,
        response=ModelResponse(result=[AIMessage("ok")]),
        samples=(),
        executed_sample=None,
        flagged=False,
    )

    # Act
    messages = build_blocked_attempt_messages(decision=decision)

    # Assert
    assert messages == []
