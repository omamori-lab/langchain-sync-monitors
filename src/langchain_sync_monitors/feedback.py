"""Turn a blocked step and its feedback into messages the agent can read.

A blocked proposal is kept, answered by the feedback, so the conversation stays
valid for every provider: each tool call, malformed ones included, gets its own
error result, and a blocked final answer gets one tagged human message. Every
message inserted here gets a fresh id, because LangGraph merges messages that
share an id and would leave tool results without their call.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)

from langchain_sync_monitors.contracts import BlockedAttempt, StepDecision
from langchain_sync_monitors.transcript import MONITOR_FEEDBACK_SOURCE

MONITOR_MESSAGE_NAME = "monitor"
"""The `name` on feedback messages. Some providers drop it, so the text carries its own marker."""


def build_monitor_message_id() -> str:
    """Return a fresh id for a message the monitor inserts into a conversation."""
    return f"monitor-{uuid4()}"


def build_tool_call_rejection(
    *,
    tool_call: ToolCall | InvalidToolCall,
    feedback: str,
) -> ToolMessage:
    """Answer one blocked tool call with an error result that carries the feedback.

    This is the reject pattern of LangChain's `HumanInTheLoopMiddleware`: the
    call stays in the transcript and its result says why it did not run
    [@langchain2026]. A malformed call is answered the same way, because
    OpenAI-compatible providers receive it as an ordinary tool call and reject
    a request that leaves it unanswered; Deep Agents' `PatchToolCallsMiddleware`
    answers malformed calls for the same reason [@deepagents2026].
    """
    return ToolMessage(
        content=feedback,
        tool_call_id=tool_call["id"] or "",
        name=tool_call["name"],
        status="error",
        id=build_monitor_message_id(),
        additional_kwargs={"lc_source": MONITOR_FEEDBACK_SOURCE},
    )


def build_answer_feedback(*, feedback: str) -> HumanMessage:
    """Answer a blocked final answer with a human message tagged as the monitor's.

    The `lc_source` tag and the `name` follow the revision messages of Deep
    Agents' `RubricMiddleware` [@deepagents2026], so in-process readers can tell
    the monitor's words from the user's.
    """
    return HumanMessage(
        content=feedback,
        name=MONITOR_MESSAGE_NAME,
        id=build_monitor_message_id(),
        additional_kwargs={"lc_source": MONITOR_FEEDBACK_SOURCE},
    )


def build_feedback_messages(*, attempt: BlockedAttempt) -> list[BaseMessage]:
    """Return the blocked proposal, with a fresh id, followed by the feedback on it.

    The feedback text is used as the protocol rendered it; its template already
    starts with the monitor's marker. Every tool call of the proposal, malformed
    ones included, is answered by an error result; only a proposal without any
    call is answered by a human message.
    """
    blocked: AIMessage = attempt.proposal.model_copy(update={"id": build_monitor_message_id()})
    tool_calls: list[ToolCall | InvalidToolCall] = [
        *blocked.tool_calls,
        *blocked.invalid_tool_calls,
    ]
    if not tool_calls:
        return [blocked, build_answer_feedback(feedback=attempt.feedback)]
    rejections = [
        build_tool_call_rejection(tool_call=tool_call, feedback=attempt.feedback)
        for tool_call in tool_calls
    ]
    return [blocked, *rejections]


def build_blocked_attempt_messages(*, decision: StepDecision) -> list[BaseMessage]:
    """Return the messages of every blocked attempt of a step, in the order they happened."""
    return [
        message
        for attempt in decision.blocked_attempts
        for message in build_feedback_messages(attempt=attempt)
    ]
