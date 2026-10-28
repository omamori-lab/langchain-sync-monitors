"""Read who wrote a LangChain message, and the provider blocks LangChain could not map.

Every human message another component wrote carries an `lc_source` tag
naming it; LangChain and Deep Agents tag some of the human messages their
middleware writes, such as a summary of earlier messages or a rubric
grader's revision request [@langchain2026; @deepagents2026], and the
monitor tags its own feedback `monitor`. A block LangChain could not map to
a standard one keeps the provider's payload under `value`, and the readers
here take it apart without trusting its shape.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from langchain_core.messages import AIMessage, BaseMessage, NonStandardContentBlock

MONITOR_FEEDBACK_SOURCE = "monitor"
"""The `lc_source` tag on messages the monitor itself inserts into a conversation."""

REASONING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})
"""The types of provider blocks that hold the model's reasoning, encrypted or not."""

REFUSAL_KEY = "refusal"
"""The key under which langchain-openai keeps a refusal in `additional_kwargs`."""

GROUNDING_METADATA_KEY = "grounding_metadata"
"""The key of a Gemini reply's `response_metadata` that holds what its grounding tools did."""


def read_unrecognised_block_value(block: NonStandardContentBlock) -> Mapping[str, object]:
    """Return what a block LangChain could not map holds, under `value` when it is no mapping."""
    value: object = block.get("value", {})
    return value if isinstance(value, Mapping) else {"value": value}


def read_unrecognised_block_name(value: Mapping[str, object]) -> str | None:
    """Return the `type` that a block LangChain could not map gives itself, if it is a string."""
    block_type = value.get("type")
    return block_type if isinstance(block_type, str) else None


def is_repeated_tool_call(value: Mapping[str, object], *, known_call_ids: Collection[str]) -> bool:
    """Tell whether a block carries the id of one of the message's tool calls, and so repeats it."""
    block_id = value.get("id")
    return isinstance(block_id, str) and block_id in known_call_ids


def read_kept_refusal(message: AIMessage) -> str | None:
    """Return a refusal langchain-openai keeps only in `additional_kwargs`, or None.

    Under OpenAI's Chat Completions API, langchain-openai keeps a refusal in
    `additional_kwargs` and leaves the content empty [@langchainopenai2026],
    and LangChain's translator gives no block for it [@langchaincore2026], so
    the monitor would not read text the user may see. A streamed reply does
    not keep the field there. Under the Responses API at
    `output_version="v0"` the translator gives the same field as a refusal
    block, which is read there, so a refusal whose text a block already
    shows is not read twice; one with other text is read, so no refusal goes
    unread.
    """
    refusal = message.additional_kwargs.get(REFUSAL_KEY)
    if not isinstance(refusal, str) or not refusal.strip():
        return None
    return None if refusal in read_block_refusals(message) else refusal


def read_block_refusals(message: AIMessage) -> set[str]:
    """Return the text of every refusal block LangChain gives for a message."""
    values = [
        read_unrecognised_block_value(block)
        for block in message.content_blocks
        if block["type"] == "non_standard"
    ]
    return {
        str(value[REFUSAL_KEY])
        for value in values
        if read_unrecognised_block_name(value) == REFUSAL_KEY and REFUSAL_KEY in value
    }


def read_message_source(message: BaseMessage) -> str | None:
    """Return the `lc_source` tag of a message another component wrote, or None without one."""
    source = message.additional_kwargs.get("lc_source")
    return None if source is None else str(source)


def is_monitor_feedback(message: BaseMessage) -> bool:
    """Tell whether the monitor itself inserted this message into the conversation.

    Any code in the application can set the tag, so tool and middleware code
    is trusted here, as it is everywhere else in the agent.
    """
    return read_message_source(message) == MONITOR_FEEDBACK_SOURCE
