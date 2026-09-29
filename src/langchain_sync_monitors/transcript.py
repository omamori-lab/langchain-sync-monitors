"""Render a conversation and a proposed step into the text a monitor reads.

Each transcript entry belongs to one `Channel`, and a monitor only receives the
entries whose channel is in its `MonitorView`. Entries are wrapped in tags, and
their content is HTML-escaped, so text inside a tool result cannot close a tag
and pose as another entry.

Only a human message without an `lc_source` tag is rendered as the task
author's words. LangChain and Deep Agents tag the human messages their
middleware writes, such as a summary of earlier messages or a rubric grader's
revision request [@langchain2026; @deepagents2026], and those are rendered as
context notes that authorise nothing.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage

from langchain_sync_monitors.contracts import Channel, MonitorView, TaskAuthor

MONITOR_FEEDBACK_SOURCE = "monitor"
"""The `lc_source` tag on messages the monitor itself inserts into a conversation."""

TASK_AUTHOR_TAGS = {TaskAuthor.USER: "user", TaskAuthor.PARENT_AGENT: "delegator"}


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
    include OpenRouter's documented `reasoning` field. A reply that carries its
    reasoning only as `reasoning_details` summaries would have no such block;
    that shape has not been seen from a real provider, but it is read as a
    fallback because the check costs nothing.
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
    attributes = "".join(
        f' {attribute}="{html.escape(value)}"'
        for attribute, value in (("name", name), ("source", source))
        if value
    )
    return f"<{tag}{attributes}>{html.escape(content, quote=False)}</{tag}>"


def render_tool_call(tool_call: ToolCall) -> str:
    """Render a tool call as a tag holding its arguments as stable, readable JSON."""
    arguments = json.dumps(tool_call["args"], ensure_ascii=False, sort_keys=True, default=str)
    return wrap_in_tag(tag="tool_call", content=arguments, name=tool_call["name"])


def build_agent_entries(message: AIMessage) -> Iterator[TranscriptEntry]:
    """Yield the reasoning, text and tool calls of one agent message."""
    reasoning = extract_reasoning_text(message)
    if reasoning:
        yield TranscriptEntry(
            channel=Channel.REASONING,
            text=wrap_in_tag(tag="agent_reasoning", content=reasoning),
        )
    if message.text.strip():
        yield TranscriptEntry(
            channel=Channel.AGENT_TEXT,
            text=wrap_in_tag(tag="agent", content=message.text),
        )
    for tool_call in message.tool_calls:
        yield TranscriptEntry(channel=Channel.TOOL_CALLS, text=render_tool_call(tool_call))


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
    did not run: a person rejected it, the tool does not exist, or the tool
    raised. It is rendered as a `tool_error` beside the tool calls, so every
    view that shows the call also shows that it did not complete, and why.
    """
    tool_name = message.name or tool_names_by_call.get(message.tool_call_id, "unknown")
    if is_monitor_feedback(message):
        return build_monitor_feedback_entry(message, name=tool_name)
    if message.status == "error":
        return TranscriptEntry(
            channel=Channel.TOOL_CALLS,
            text=wrap_in_tag(tag="tool_error", content=message.text, name=tool_name),
        )
    if tool_name in view.delegation_tools:
        return TranscriptEntry(
            channel=Channel.SUBAGENT_REPORTS,
            text=wrap_in_tag(tag="subagent_report", content=message.text, name=tool_name),
        )
    return TranscriptEntry(
        channel=Channel.TOOL_RESULTS,
        text=wrap_in_tag(tag="tool_result", content=message.text, name=tool_name),
    )


def build_human_entry(message: HumanMessage, *, task_author: TaskAuthor) -> TranscriptEntry:
    """Render a human message as the task author's words, a context note or monitor feedback.

    A human message tagged with any `lc_source` other than the monitor's was
    written by another part of the application, often with the agent's own
    model, so it becomes a `context_note` that names its source and
    authorises nothing.
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
        return list(build_agent_entries(message))
    if isinstance(message, ToolMessage):
        return [build_tool_entry(message, tool_names_by_call=tool_names_by_call, view=view)]
    if isinstance(message, HumanMessage):
        return [build_human_entry(message, task_author=task_author)]
    return []


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
        if isinstance(message, AIMessage):
            tool_names_by_call.update(
                (call["id"], call["name"]) for call in message.tool_calls if call["id"]
            )
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

    A proposal without tool calls is a final answer, and the answer is then the
    step's action, so its text is shown even when the view leaves out agent
    prose.
    """
    is_final_answer = not proposal.tool_calls
    channels = view.channels | Channel.AGENT_TEXT if is_final_answer else view.channels
    parts = [entry.text for entry in build_agent_entries(proposal) if entry.channel in channels]
    return "<proposed_step>\n" + "\n".join(parts) + "\n</proposed_step>"


def render_proposal_for_audit(proposal: AIMessage) -> str:
    """Render a proposal with every channel, for the human auditors reading `monitor_log`."""
    return render_proposed_step(proposal, view=MonitorView(channels=Channel.ALL))
