"""Turn a blocked step and its feedback into messages the agent can read, and build halts.

A blocked proposal is kept, answered by the feedback, so the conversation stays
valid for every provider: each tool call, malformed ones included, gets its own
error result, and a blocked final answer gets one tagged human message. When a
blocked proposal is committed, its text is withheld, wherever the message keeps
it, so a chat interface does not show it; its tool calls, arguments included,
stay. A halt is a flagged decision whose one message is the monitor's final
word. Every
message inserted here gets a fresh id, because LangGraph merges messages that
share an id and would leave tool results without their call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Final
from uuid import uuid4

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)

from langchain_sync_monitors.contracts import BlockedAttempt, Outcome, Sample, StepDecision
from langchain_sync_monitors.transcript import (
    GROUNDING_METADATA_KEY,
    MONITOR_FEEDBACK_SOURCE,
    PROSE_BLOCK_TYPES,
    REFUSAL_KEY,
)

WITHHELD_TEXT_MESSAGE: Final = "[Safety monitor] This step was blocked, so its text is withheld."
"""The text a blocked attempt keeps when `FeedbackVisibility.IN_TRANSCRIPT` commits it."""

CITATION_KEYS = frozenset({"annotations", "citations"})
"""The keys of a text block that cite sources for its text: OpenAI's and Anthropic's."""

GROUNDING_SUPPORTS_KEY = "grounding_supports"
"""The key of Gemini's grounding metadata that quotes each grounded sentence of the answer."""

MONITOR_MESSAGE_NAME = "monitor"
"""The `name` on feedback messages. Some providers drop it, so the text carries its own marker."""


def build_monitor_message_id() -> str:
    """Return a fresh id for a message the monitor inserts into a conversation."""
    return f"monitor-{uuid4()}"


def build_halt_decision(
    content: str,
    *,
    samples: tuple[Sample, ...] = (),
    blocked_attempts: tuple[BlockedAttempt, ...] = (),
) -> StepDecision:
    """Return a flagged halt whose final message is `content`, with a fresh id and no tool calls."""
    message = AIMessage(content=content, id=build_monitor_message_id())
    return StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[message]),
        samples=samples,
        executed_sample=None,
        flagged=True,
        blocked_attempts=blocked_attempts,
    )


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


def build_blocked_attempt_messages(attempts: Sequence[BlockedAttempt]) -> list[BaseMessage]:
    """Return the messages of every blocked attempt of a step, in the order they happened."""
    return [message for attempt in attempts for message in build_feedback_messages(attempt=attempt)]


def build_committed_attempt_messages(attempts: Sequence[BlockedAttempt]) -> list[BaseMessage]:
    """Return what `FeedbackVisibility.IN_TRANSCRIPT` commits of a step's blocked attempts.

    These are the messages the step's retries saw, with each proposal's text
    withheld, since a chat interface shows committed text to the user.
    """
    withheld = [
        replace(attempt, proposal=build_withheld_proposal(attempt.proposal)) for attempt in attempts
    ]
    return build_blocked_attempt_messages(withheld)


def build_withheld_proposal(proposal: AIMessage) -> AIMessage:
    """Return a copy of a blocked proposal whose text is `WITHHELD_TEXT_MESSAGE`.

    The text is replaced wherever the message keeps it: string content, each
    text block and refusal, a refusal langchain-openai keeps in
    `additional_kwargs`, and the grounding supports in which Gemini quotes the
    answer, which LangChain turns back into citations of the text. A withheld
    text block also loses its citations, which describe the old text. The tool
    calls stay, since the feedback answers them, and so do the reasoning and
    every other block, with a text block's id or signature, because some
    providers need an earlier reply back as they sent it. Blank text stays
    blank. Each changed part is a new value, so the proposal itself, which the
    retries and `monitor_log` keep, is untouched. This is the library's own
    rule, not one from a control evaluation.
    """
    content = proposal.content
    withheld = (
        replace_prose(content)
        if isinstance(content, str)
        else [replace_block_prose(block) for block in content]
    )
    return proposal.model_copy(
        update={
            "content": withheld,
            "additional_kwargs": replace_kept_refusal(proposal.additional_kwargs),
            "response_metadata": remove_grounding_supports(proposal.response_metadata),
        },
    )


def replace_prose(text: str) -> str:
    """Return `WITHHELD_TEXT_MESSAGE` in place of text, or the text itself when it is blank."""
    return WITHHELD_TEXT_MESSAGE if text.strip() else text


def replace_block_prose(block: str | dict[str, object]) -> str | dict[str, object]:
    """Return a content block with its text withheld: a plain string, a text block or a refusal.

    A refusal arrives as the provider's own block, or wrapped as a
    `non_standard` block whose `value` is that block.
    """
    if isinstance(block, str):
        return replace_prose(block)
    value = block.get("value")
    if block.get("type") == "non_standard" and isinstance(value, dict):
        return {**block, "value": replace_block_prose(value)}
    key = read_prose_key(block)
    if key is None:
        return block
    kept = {name: part for name, part in block.items() if name not in CITATION_KEYS}
    return {**kept, key: replace_prose(str(block[key]))}


def read_prose_key(block: dict[str, object]) -> str | None:
    """Return the key a text block or a refusal holds its text under, or None for other blocks.

    A text block holds its text under `text`, and a refusal under its own type.
    """
    key = str(block.get("type"))
    is_prose = key == "text" or key in PROSE_BLOCK_TYPES
    return key if is_prose and isinstance(block.get(key), str) else None


def replace_kept_refusal(additional_kwargs: dict[str, object]) -> dict[str, object]:
    """Return a message's extra fields, with a refusal kept there withheld, as a new dict."""
    refusal = additional_kwargs.get(REFUSAL_KEY)
    if not isinstance(refusal, str):
        return {**additional_kwargs}
    return {**additional_kwargs, REFUSAL_KEY: replace_prose(refusal)}


def remove_grounding_supports(response_metadata: dict[str, object]) -> dict[str, object]:
    """Return response metadata without Gemini's grounding supports, as a new dict.

    The monitor reads the searches and the sources, which stay; the supports
    only quote the answer it withholds.
    """
    grounding = response_metadata.get(GROUNDING_METADATA_KEY)
    if not isinstance(grounding, dict):
        return {**response_metadata}
    kept = {key: value for key, value in grounding.items() if key != GROUNDING_SUPPORTS_KEY}
    return {**response_metadata, GROUNDING_METADATA_KEY: kept}
