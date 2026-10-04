"""Render a conversation and a proposed step into the text a monitor reads.

Each transcript entry belongs to one `Channel`, and a monitor only receives the
entries whose channel is in its `MonitorView`. Entries are wrapped in tags, and
their content is HTML-escaped, so text inside a tool result cannot close a tag
and pose as another entry. Server-side tools, which a model provider runs
inside the model call, are rendered too, with the tool calls and the tool
results.

Only a human message without an `lc_source` tag is rendered as the task
author's words. LangChain and Deep Agents tag some of the human messages their
middleware writes, such as a summary of earlier messages or a rubric grader's
revision request [@langchain2026; @deepagents2026], and those are rendered as
context notes that authorise nothing. Deep Agents leaves others untagged, such
as the frames of a video that `read_file` attaches, with the agent's own path
in their text [@deepagents2026]. So the middleware passes a monitor only the
human messages that arrived as a run's input untagged, and `task_authorship`
tags every other one.
"""

from __future__ import annotations

import html
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict, TypeGuard

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ContentBlock,
    HumanMessage,
    InvalidToolCall,
    NonStandardContentBlock,
    ServerToolCall,
    ServerToolCallChunk,
    ServerToolResult,
    ToolCall,
    ToolMessage,
)

from langchain_sync_monitors.contracts import Channel, MonitorView, TaskAuthor

MONITOR_FEEDBACK_SOURCE = "monitor"
"""The `lc_source` tag on messages the monitor itself inserts into a conversation."""

TASK_AUTHOR_TAGS = {TaskAuthor.USER: "user", TaskAuthor.PARENT_AGENT: "delegator"}

REASONING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})
"""The types of provider blocks that hold the model's reasoning, encrypted or not."""

PROSE_BLOCK_TYPES = frozenset({"refusal"})
"""The types of provider blocks that hold text the model wrote, under a key named after the type.

OpenAI gives a refusal as a `refusal` item, which LangChain keeps as a block it does not map.
"""

REFUSAL_KEY = "refusal"
"""The key under which langchain-openai keeps a refusal in `additional_kwargs`."""

GROUNDING_METADATA_KEY = "grounding_metadata"
"""The key of a Gemini reply's `response_metadata` that holds what its grounding tools did."""

GROUNDING_QUERY_KEYS = ("web_search_queries", "image_search_queries")
"""The keys of Gemini's `grounding_metadata` that hold the searches its server tools ran."""


class ServerToolCallDetails(TypedDict):
    """What the entry of a server tool call shows.

    `args` is a dictionary once the call is complete; a streamed part of a
    call holds its arguments as a JSON fragment.
    """

    args: Mapping[str, object] | str
    extras: NotRequired[Mapping[str, object]]


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptEntry:
    """One rendered piece of a conversation, with the channel it belongs to.

    `from_task_author` marks a message from the user, or from the delegating
    agent, which an entry limit never drops.
    """

    channel: Channel
    text: str
    from_task_author: bool = False


def extract_reasoning_text(message: AIMessage) -> str:
    """Return the agent's chain of thought from a model reply, or an empty string.

    Reads LangChain's standard reasoning content blocks first, which already
    include OpenRouter's documented `reasoning` field, as langchain-openrouter
    maps it [@langchaincore2026; @langchainopenrouter2026]. A reply that
    carries its reasoning only as `reasoning_details` summaries would have no
    such block; that shape has not been seen from a real provider, but it is
    read as a fallback because the check costs nothing.
    """
    reasoning_parts = [
        block.get("reasoning", "")
        for block in message.content_blocks
        if block["type"] == "reasoning"
    ]
    reasoning = "\n".join(part for part in reasoning_parts if part)
    if reasoning:
        return reasoning
    details = message.additional_kwargs.get("reasoning_details")
    if not isinstance(details, list):
        return ""
    return "\n".join(read_reasoning_detail_text(detail) for detail in details).strip()


def read_reasoning_detail_text(detail: object) -> str:
    """Return the text or summary of one OpenRouter reasoning detail entry."""
    if not isinstance(detail, dict):
        return ""
    text = detail.get("text") or detail.get("summary") or ""
    return text if isinstance(text, str) else ""


def wrap_in_tag(
    *,
    tag: str,
    content: str,
    name: str | None = None,
    source: str | None = None,
) -> str:
    """Wrap escaped content in a tag, with optional escaped `name` and `source` attributes."""
    # Quotes are escaped only in attributes, where one could end the value; content keeps them.
    attributes = "".join(
        f' {attribute}="{html.escape(value)}"'
        for attribute, value in (("name", name), ("source", source))
        if value
    )
    return f"<{tag}{attributes}>{html.escape(content, quote=False)}</{tag}>"


def render_json(value: object) -> str:
    """Render a value as stable, readable JSON, with anything JSON cannot hold as text."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def render_tool_call(tool_call: ToolCall) -> str:
    """Render a tool call as a tag holding its arguments as stable, readable JSON."""
    arguments = render_json(tool_call["args"])
    return wrap_in_tag(tag="tool_call", content=arguments, name=tool_call["name"])


def render_malformed_tool_call(tool_call: InvalidToolCall) -> str:
    """Render a tool call whose arguments could not be parsed, with its raw argument text.

    Such a call never runs, but it shows what the agent tried to do.
    """
    return wrap_in_tag(
        tag="malformed_tool_call",
        content=tool_call["args"] or "",
        name=tool_call["name"],
    )


def is_server_tool_call(block: ContentBlock) -> TypeGuard[ServerToolCall | ServerToolCallChunk]:
    """Tell whether a content block is a server tool call the provider ran, whole or streamed."""
    return block["type"] == "server_tool_call" or block["type"] == "server_tool_call_chunk"


def render_server_tool_call(block: ServerToolCall | ServerToolCallChunk) -> str:
    """Render a server tool call that the model provider ran inside the model call.

    The content holds the call's arguments and, when the block has them, its
    provider `extras`, where a remote MCP call keeps the name of the tool it
    ran and any arguments that could not be parsed.
    """
    details = ServerToolCallDetails(args=block.get("args", {}))
    if "extras" in block:
        details["extras"] = block["extras"]
    return wrap_in_tag(
        tag="server_tool_call",
        content=render_json(details),
        name=block.get("name"),
    )


def render_server_tool_result(block: ServerToolResult, *, tool_name: str) -> str:
    """Render what a server tool returned: its output as text, or else as JSON."""
    output = block.get("output")
    if output is None:
        content = ""
    elif isinstance(output, str):
        content = output
    else:
        content = render_json(output)
    return wrap_in_tag(tag="server_tool_result", content=content, name=tool_name)


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


def build_unrecognised_block_entry(
    block: NonStandardContentBlock,
    *,
    known_call_ids: Collection[str],
) -> TranscriptEntry | None:
    """Render a part of a reply that LangChain could not map to a standard block, whole.

    The block's own `type`, when it has one, is its name. It may be a tool
    call the provider ran, in a reply without `model_provider` in its
    metadata, so it sits with the tool calls rather than being dropped. A
    block of reasoning, such as Anthropic's `thinking`, sits with the
    reasoning instead, so a view without it still leaves it out. A refusal is
    text the model wrote to the user, so it renders as the agent's prose,
    which is how the monitor reads it in the step it judges. A block that carries
    the id of one of the message's tool calls repeats that call, which is
    rendered already, so it renders as nothing.
    """
    value = read_unrecognised_block_value(block)
    if is_repeated_tool_call(value, known_call_ids=known_call_ids):
        return None
    name = read_unrecognised_block_name(value)
    prose = value.get(name) if name in PROSE_BLOCK_TYPES else None
    if isinstance(prose, str):
        return TranscriptEntry(
            channel=Channel.AGENT_TEXT, text=wrap_in_tag(tag="agent", content=prose)
        )
    channel = Channel.REASONING if name in REASONING_BLOCK_TYPES else Channel.TOOL_CALLS
    text = wrap_in_tag(tag="unrecognised_block", content=render_json(value), name=name)
    return TranscriptEntry(channel=channel, text=text)


def build_server_tool_entries(
    blocks: Sequence[ContentBlock],
    *,
    tool_names_by_call: Mapping[str, str],
    known_call_ids: Collection[str] = frozenset(),
) -> Iterator[TranscriptEntry]:
    """Yield the server tool calls the provider ran inside a model call, and their results.

    Tools such as Anthropic's web fetch or OpenAI's web search run at the
    provider, before the reply reaches the monitor. LangChain's translators
    give them as standard `server_tool_call` and `server_tool_result` blocks
    for Anthropic, OpenAI's Responses API and Gemini's code execution, and in
    a reply already in LangChain's standard blocks [@langchaincore2026]. A
    call sits with the tool calls and a result with the tool results, each in
    a tag that says the provider ran it. A block LangChain could not map sits
    with the tool calls too, so it is never dropped unseen.
    """
    for block in blocks:
        if is_server_tool_call(block):
            yield TranscriptEntry(
                channel=Channel.TOOL_CALLS,
                text=render_server_tool_call(block),
            )
        elif block["type"] == "server_tool_result":
            tool_name = tool_names_by_call.get(block.get("tool_call_id", ""), "unknown")
            yield TranscriptEntry(
                channel=Channel.TOOL_RESULTS,
                text=render_server_tool_result(block, tool_name=tool_name),
            )
        elif block["type"] == "non_standard":
            entry = build_unrecognised_block_entry(block, known_call_ids=known_call_ids)
            if entry is not None:
                yield entry


def build_grounding_entries(message: AIMessage) -> Iterator[TranscriptEntry]:
    """Yield the searches of Gemini's grounding tools as a server tool call and result.

    langchain-google-genai keeps the queries of Gemini's Google Search only
    in the reply's `grounding_metadata`, which LangChain's Gemini translator
    reads for citations [@langchaincore2026], with no `server_tool_call`
    block. The queries are the call, and the sources found are its result.
    """
    metadata = message.response_metadata.get(GROUNDING_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        return
    queries = {key: metadata[key] for key in GROUNDING_QUERY_KEYS if metadata.get(key)}
    if queries:
        details = ServerToolCallDetails(args=queries)
        yield TranscriptEntry(
            channel=Channel.TOOL_CALLS,
            text=wrap_in_tag(
                tag="server_tool_call", content=render_json(details), name="grounding"
            ),
        )
    sources = metadata.get("grounding_chunks")
    if sources:
        yield TranscriptEntry(
            channel=Channel.TOOL_RESULTS,
            text=wrap_in_tag(
                tag="server_tool_result", content=render_json(sources), name="grounding"
            ),
        )


def build_agent_entries(
    message: AIMessage,
    *,
    tool_names_by_call: Mapping[str, str],
) -> Iterator[TranscriptEntry]:
    """Yield the reasoning, server tools, text, tool calls and malformed calls of a message.

    `tool_names_by_call` names the server tool call that each server tool
    result answers.
    """
    reasoning = extract_reasoning_text(message)
    if reasoning:
        yield TranscriptEntry(
            channel=Channel.REASONING,
            text=wrap_in_tag(tag="agent_reasoning", content=reasoning),
        )
    calls: list[ToolCall | InvalidToolCall] = [*message.tool_calls, *message.invalid_tool_calls]
    yield from build_server_tool_entries(
        message.content_blocks,
        tool_names_by_call=tool_names_by_call,
        known_call_ids={call["id"] for call in calls if call["id"]},
    )
    yield from build_grounding_entries(message)
    yield from build_text_entries(message)
    for tool_call in message.tool_calls:
        yield TranscriptEntry(channel=Channel.TOOL_CALLS, text=render_tool_call(tool_call))
    for invalid_tool_call in message.invalid_tool_calls:
        yield TranscriptEntry(
            channel=Channel.TOOL_CALLS,
            text=render_malformed_tool_call(invalid_tool_call),
        )


def build_text_entries(message: AIMessage) -> Iterator[TranscriptEntry]:
    """Yield the text a message gives the user: its content's text, then a refusal kept aside."""
    texts = [message.text, read_kept_refusal(message) or ""]
    for text in texts:
        if text.strip():
            yield TranscriptEntry(
                channel=Channel.AGENT_TEXT,
                text=wrap_in_tag(tag="agent", content=text),
            )


def read_kept_refusal(message: AIMessage) -> str | None:
    """Return a refusal langchain-openai keeps only in `additional_kwargs`, or None.

    Under OpenAI's Chat Completions API, langchain-openai keeps a refusal in
    `additional_kwargs` and leaves the content empty [@langchainopenai2026],
    and LangChain's translator gives no block for it [@langchaincore2026], so
    the monitor would not read text the user may see. Under the Responses
    API at `output_version="v0"` the translator gives the same field as a
    refusal block, which is read there, so it is not read twice.
    """
    refusal = message.additional_kwargs.get(REFUSAL_KEY)
    if not isinstance(refusal, str) or not refusal.strip():
        return None
    if any(is_refusal_block(block) for block in message.content_blocks):
        return None
    return refusal


def is_refusal_block(block: ContentBlock) -> bool:
    """Tell whether a content block is a refusal LangChain could not map."""
    if block["type"] != "non_standard":
        return False
    return read_unrecognised_block_name(read_unrecognised_block_value(block)) == REFUSAL_KEY


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


def build_monitor_feedback_entry(
    message: BaseMessage, *, name: str | None = None
) -> TranscriptEntry:
    """Render the monitor's own feedback, which is shown wherever the task is shown.

    A blocked tool call is answered by a tool message that carries the
    feedback, and it is named after the tool, so the monitor can tell which
    call did not run.
    """
    return TranscriptEntry(
        channel=Channel.USER,
        text=wrap_in_tag(tag="monitor_feedback", content=message.text, name=name),
    )


def build_tool_entry(
    message: ToolMessage,
    *,
    tool_names_by_call: dict[str, str],
    view: MonitorView,
) -> TranscriptEntry:
    """Render a tool result, as a subagent report when it comes from a delegation tool.

    A tool message the monitor wrote to reject a blocked call is its feedback,
    not the tool's output. Any other error result says that the call failed or
    did not run: a person rejected it, which LangChain's
    `HumanInTheLoopMiddleware` answers with an error result [@langchain2026];
    the tool does not exist; or the tool raised. It is rendered as a
    `tool_error` beside the tool calls, so every view that shows the call also
    shows that it did not complete, and why.
    """
    tool_name = message.name or tool_names_by_call.get(message.tool_call_id, "unknown")
    # The monitor's own rejections carry the error status too, so they are told apart first.
    if is_monitor_feedback(message):
        return build_monitor_feedback_entry(message, name=tool_name)
    if message.status == "error":
        channel, tag = Channel.TOOL_CALLS, "tool_error"
    elif tool_name in view.delegation_tools:
        channel, tag = Channel.SUBAGENT_REPORTS, "subagent_report"
    else:
        channel, tag = Channel.TOOL_RESULTS, "tool_result"
    return TranscriptEntry(
        channel=channel, text=wrap_in_tag(tag=tag, content=message.text, name=tool_name)
    )


def build_human_entry(message: HumanMessage, *, task_author: TaskAuthor) -> TranscriptEntry:
    """Render a human message as the task author's words, a context note or monitor feedback.

    A human message tagged with any `lc_source` other than the monitor's was
    written by another part of the application, often with the agent's own
    model, so it becomes a `context_note` that names its source and
    authorises nothing. Inside an agent, `task_authorship` has already tagged
    every human message that did not arrive as a run's input.
    """
    source = read_message_source(message)
    if source is None:
        return TranscriptEntry(
            channel=Channel.USER,
            text=wrap_in_tag(tag=TASK_AUTHOR_TAGS[task_author], content=message.text),
            from_task_author=True,
        )
    if source == MONITOR_FEEDBACK_SOURCE:
        return build_monitor_feedback_entry(message)
    return TranscriptEntry(
        channel=Channel.USER,
        text=wrap_in_tag(tag="context_note", content=message.text, source=source),
    )


def build_message_entries(
    message: BaseMessage,
    *,
    tool_names_by_call: dict[str, str],
    view: MonitorView,
    task_author: TaskAuthor,
) -> list[TranscriptEntry]:
    """Render one message; system messages and unknown message types render as nothing."""
    if isinstance(message, AIMessage):
        return list(build_agent_entries(message, tool_names_by_call=tool_names_by_call))
    if isinstance(message, ToolMessage):
        return [build_tool_entry(message, tool_names_by_call=tool_names_by_call, view=view)]
    if isinstance(message, HumanMessage):
        return [build_human_entry(message, task_author=task_author)]
    return []


def read_server_tool_names_by_call(message: AIMessage) -> dict[str, str]:
    """Return the name of each server tool call the provider ran in a message, by id."""
    names: dict[str, str] = {}
    for block in message.content_blocks:
        if is_server_tool_call(block):
            call_id = block.get("id")
            name = block.get("name")
            if call_id and name:
                names[call_id] = name
    return names


def read_tool_names_by_call(message: AIMessage) -> dict[str, str]:
    """Return the tool name of each call in an agent message by id.

    Malformed calls and server tool calls are included.
    """
    calls: list[ToolCall | InvalidToolCall] = [*message.tool_calls, *message.invalid_tool_calls]
    names = {call["id"]: call["name"] for call in calls if call["id"] and call["name"]}
    return {**names, **read_server_tool_names_by_call(message)}


def build_transcript_entries(
    history: Sequence[BaseMessage],
    *,
    view: MonitorView,
    task_author: TaskAuthor,
) -> list[TranscriptEntry]:
    """Turn a conversation into entries in order, before any channel filtering."""
    tool_names_by_call: dict[str, str] = {}
    entries: list[TranscriptEntry] = []
    for message in history:
        # A tool result may name its call by id alone, so the AI messages before it supply names.
        if isinstance(message, AIMessage):
            tool_names_by_call.update(read_tool_names_by_call(message))
        entries.extend(
            build_message_entries(
                message,
                tool_names_by_call=tool_names_by_call,
                view=view,
                task_author=task_author,
            ),
        )
    return entries


def select_visible_entries(
    entries: list[TranscriptEntry],
    *,
    view: MonitorView,
) -> list[TranscriptEntry]:
    """Keep the entries the view can read, limited to the most recent ones.

    When the limit drops older entries, every message from the task author is
    kept anyway, in order: the first states the task, and a later one can
    narrow or revoke it. They are few, and a monitor needs them all to judge a
    step. A context note is never kept in their place.
    """
    visible = [entry for entry in entries if entry.channel in view.channels]
    limit = view.most_recent_entries
    if limit is None or len(visible) <= limit:
        return visible
    cut = len(visible) - limit
    earlier_task_entries = [entry for entry in visible[:cut] if entry.from_task_author]
    return [*earlier_task_entries, *visible[cut:]]


def render_transcript(
    history: Sequence[BaseMessage],
    *,
    view: MonitorView,
    task_author: TaskAuthor,
) -> str:
    """Render the conversation so far as the monitor sees it through its view."""
    entries = build_transcript_entries(history, view=view, task_author=task_author)
    return "\n".join(entry.text for entry in select_visible_entries(entries, view=view))


def render_proposed_step(proposal: AIMessage, *, view: MonitorView) -> str:
    """Render the step the agent proposes, wrapped in a `proposed_step` tag.

    The view decides what the monitor reads of the history, not of the step it
    judges. The step's tool calls, malformed ones and server tool calls
    included, are its action, so they are shown whatever the view. Its text,
    a refusal included, is shown whatever the view too: the step commits it
    to the conversation, where a chat interface shows it to the user, whether
    it is a final answer or sits beside a tool call. Left out, it would let a
    message the monitor must judge reach the user beside any harmless call.
    Malformed calls cannot run but are shown as evidence of what the agent
    tried. The step's reasoning, and what its server tools returned, follow
    the view's channels.
    """
    channels = view.channels | Channel.TOOL_CALLS | Channel.AGENT_TEXT
    entries = build_agent_entries(
        proposal,
        tool_names_by_call=read_server_tool_names_by_call(proposal),
    )
    parts = [entry.text for entry in entries if entry.channel in channels]
    return "<proposed_step>\n" + "\n".join(parts) + "\n</proposed_step>"


def render_proposal_for_audit(proposal: AIMessage) -> str:
    """Render a proposal with every channel, for the human auditors reading `monitor_log`.

    Malformed tool calls are included, so the record keeps what the agent
    tried even though nothing ran, and so are server tool calls and their
    full results, which ran before the monitor judged.
    """
    return render_proposed_step(proposal, view=MonitorView(channels=Channel.ALL))
