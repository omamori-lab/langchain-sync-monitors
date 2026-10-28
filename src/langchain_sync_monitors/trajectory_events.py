"""The actions and observations a LangChain message adds to a hyoshigi trajectory.

Two readers divide the work. `read_proposed_action` reads a reply the monitor
rates. `read_events` reads a message that enters the history without being
rated: an input, a tool result, a system or developer message, or an AI
message the monitor never saw. Each message is appended once, by its id, by
whichever path sees it first.

Both read LangChain's standard content blocks [@langchaincore2026], so every
provider takes one path, and follow hyoshigi's mapping [@hyoshigi2026]. Tool
calls come from a message's `tool_calls` and `invalid_tool_calls`, not from
its blocks: given a reply with one valid and one malformed `tool_use`,
langchain-core 1.7.0's Anthropic translator gives the valid call for both. A
call block whose call the message lacks is an opaque part, so its step fails
closed. OpenRouter's `reasoning_details` are read from `additional_kwargs`,
because the blocks hold only its `reasoning` text [@langchainopenrouter2026],
and an encrypted detail would be lost.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from hyoshigi import (
    Action,
    Author,
    Event,
    Observation,
    OpaquePart,
    Part,
    TextPart,
    ToolCall,
    ToolObservation,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ContentBlock,
    HumanMessage,
    InvalidToolCall,
    NonStandardContentBlock,
    ReasoningContentBlock,
    ServerToolCall,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import ToolCall as LangChainToolCall

from langchain_sync_monitors.content_parts import read_content_part, read_unmapped_kind
from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.errors import MessageFormatError
from langchain_sync_monitors.message_sources import (
    GROUNDING_METADATA_KEY,
    MONITOR_FEEDBACK_SOURCE,
    REASONING_BLOCK_TYPES,
    is_monitor_feedback,
    is_repeated_tool_call,
    read_kept_refusal,
    read_message_source,
    read_unrecognised_block_value,
)
from langchain_sync_monitors.task_authorship import UNCONFIRMED_INPUT_SOURCE

SERVER_SOURCE = "server"
"""The source of the observation of a server tool's result."""

SERVER_RESULT_PLACEHOLDER = OpaquePart(kind="server_tool_result")
"""What a server tool's result is in a reply the monitor rates."""

HIDDEN_REASONING = OpaquePart(kind="reasoning")
"""What reasoning the provider hid, or that holds no readable text, is in a thought."""

INVALID_CALL_PLACEHOLDER = OpaquePart(kind="invalid_tool_call")
"""What a malformed tool call without an id or a name is in an action's parts."""

GROUNDING_TOOL_NAME = "grounding"
"""The name of the server tool call that stands for a Gemini reply's grounding searches."""

GROUNDING_SOURCES_KEY = "grounding_chunks"
"""The key of Gemini's `grounding_metadata` that holds the sources its searches found."""

DEVELOPER_ROLE_KEY = "__openai_role__"
"""The key with which langchain-core marks a system message as OpenAI's developer message."""

REASONING_DETAILS_KEY = "reasoning_details"
"""The key of `additional_kwargs` that holds OpenRouter's reasoning details."""

ENCRYPTED_CONTENT_KEY = "encrypted_content"
"""The key of a reasoning block that holds reasoning the provider encrypted."""

READ_ELSEWHERE_TYPES = frozenset({"reasoning", "server_tool_call"})
"""The block types an AI message's parts leave out: its thought and tool calls hold them."""

TOOL_BLOCK_TYPES = frozenset({"tool_call", "invalid_tool_call", "server_tool_result"})
"""The block types of a call, which the message's own calls hold, and of a server result."""

GROUNDING_QUERY_KEYS = ("web_search_queries", "image_search_queries", "retrieval_queries")
"""The fields of Gemini's `grounding_metadata` that hold the queries its searches ran."""

TASK_AUTHORS = {TaskAuthor.USER: Author.USER, TaskAuthor.PARENT_AGENT: Author.DELEGATOR}
"""The author of an untagged human message, by who wrote the task."""


def read_events(message: BaseMessage, *, task_author: TaskAuthor) -> list[Event]:
    """Read a message that enters the history unrated into the events it adds.

    An AI message adds its action, then the observation of each server tool
    result it holds, in order. Any other message adds one observation; an
    untagged human message is written by `task_author`. Raises
    `MessageFormatError` on an AI message without an id, on a message that is
    not an AI, human, system or tool message, and on a block of a known kind
    that lacks what its kind requires.
    """
    if isinstance(message, AIMessage):
        return [read_action(message, server_result=()), *read_server_observations(message)]
    return [read_observation(message, task_author=task_author)]


def read_proposed_action(message: AIMessage) -> Action:
    """Read a reply the monitor rates into the action it rates.

    Each server tool result in the reply is an opaque part where it stood, so
    the rating fails closed. Raises `MessageFormatError` as `read_events` does.
    """
    return read_action(message, server_result=(SERVER_RESULT_PLACEHOLDER,))


def read_observation(message: BaseMessage, *, task_author: TaskAuthor) -> Observation:
    """Read a human, system or tool message into the observation it is."""
    path = type(message).__name__
    if isinstance(message, HumanMessage):
        return read_human_observation(message, task_author=task_author)
    if isinstance(message, SystemMessage):
        source = read_system_source(message)
        return Observation(author=Author.SCAFFOLD, source=source, parts=read_parts(message))
    if isinstance(message, ToolMessage):
        return read_tool_observation(message)
    raise MessageFormatError(path=path, expected="an AI, human, system or tool message")


def read_human_observation(message: HumanMessage, *, task_author: TaskAuthor) -> Observation:
    """Read a human message as written by the author its `lc_source` tag names.

    An untagged message is the task author's, the monitor's tag its feedback,
    and unconfirmed input an unknown author's. Any other tag names a note the
    scaffold wrote, such as a summary.
    """
    parts = read_parts(message)
    source = read_message_source(message)
    if source is None:
        return Observation(author=TASK_AUTHORS[task_author], parts=parts)
    if source == MONITOR_FEEDBACK_SOURCE:
        return Observation(author=Author.MONITOR, parts=parts)
    if source == UNCONFIRMED_INPUT_SOURCE:
        return Observation(author=Author.UNKNOWN, source=source, parts=parts)
    return Observation(author=Author.SCAFFOLD, source=source, parts=parts)


def read_system_source(message: SystemMessage) -> str:
    """Return `developer` for OpenAI's developer message, as langchain-core marks it, else `system`.

    langchain-core keeps a developer message as a system message marked in
    `additional_kwargs` [@langchaincore2026].
    """
    return (
        "developer"
        if message.additional_kwargs.get(DEVELOPER_ROLE_KEY) == "developer"
        else "system"
    )


def read_tool_observation(message: ToolMessage) -> ToolObservation:
    """Read a tool result, an error when its status says so; the monitor's feedback is its own."""
    author = Author.MONITOR if is_monitor_feedback(message) else Author.TOOL
    return ToolObservation(
        call_id=message.tool_call_id,
        is_error=message.status == "error",
        author=author,
        parts=read_parts(message),
    )


def read_parts(message: BaseMessage) -> tuple[Part, ...]:
    """Read the content blocks of a human, system or tool message into parts."""
    path = type(message).__name__
    return tuple(
        read_content_part(block, path=f"{path}.content_blocks[{index}]")
        for index, block in enumerate(message.content_blocks)
    )


def read_action(message: AIMessage, *, server_result: tuple[Part, ...]) -> Action:
    """Read an AI message into its action; `server_result` is what a server tool result is in it."""
    path = type(message).__name__
    if not message.id:
        raise MessageFormatError(path=f"{path}.id", expected="an id")
    # langchain-core's Gemini translator fails on grounding metadata that is not a mapping.
    read_grounding_metadata(message)
    blocks = message.content_blocks
    return Action(
        id=message.id,
        thought=read_thought(message, blocks=blocks),
        parts=read_agent_parts(message, blocks=blocks, server_result=server_result),
        tool_calls=read_tool_calls(message, blocks=blocks),
    )


def read_thought(message: AIMessage, *, blocks: Sequence[ContentBlock]) -> tuple[Part, ...]:
    """Read the reasoning: OpenRouter's reasoning details when there are any, else the blocks.

    hyoshigi reads OpenRouter's details in place of its reasoning text
    [@hyoshigi2026], and they hold the text the blocks hold.
    """
    path = type(message).__name__
    details: object = message.additional_kwargs.get(REASONING_DETAILS_KEY)
    details_path = f"{path}.additional_kwargs.{REASONING_DETAILS_KEY}"
    match details:
        case list() if details:
            return tuple(
                read_reasoning_detail(detail, path=f"{details_path}[{index}]")
                for index, detail in enumerate(details)
            )
        case list() | None:
            return tuple(
                part
                for index, block in enumerate(blocks)
                for part in read_reasoning_block(block, path=f"{path}.content_blocks[{index}]")
            )
        case _:
            raise MessageFormatError(path=details_path, expected="a list of reasoning details")


def read_reasoning_detail(detail: object, *, path: str) -> Part:
    """Read one OpenRouter reasoning detail: its text or summary; another type is opaque.

    The types are those of OpenRouter's reasoning tokens guide, as hyoshigi
    reads them [@hyoshigi2026].
    """
    match detail:
        case {"type": "reasoning.text", "text": str(text)} | {
            "type": "reasoning.summary",
            "summary": str(text),
        }:
            return TextPart(text=text)
        case {"type": "reasoning.text" | "reasoning.summary"}:
            raise MessageFormatError(
                path=path,
                expected="a reasoning detail with its text or summary",
            )
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise MessageFormatError(path=path, expected="a reasoning detail with a type")


def read_reasoning_block(block: ContentBlock, *, path: str) -> tuple[Part, ...]:
    """Read the reasoning in a block: a reasoning block, or reasoning LangChain could not map."""
    if block["type"] == "reasoning":
        return read_reasoning(block, path=path)
    if block["type"] == "non_standard":
        return read_unmapped_reasoning(block)
    return ()


def read_reasoning(block: ReasoningContentBlock, *, path: str) -> tuple[Part, ...]:
    """Read a reasoning block: its text, then an opaque part when it carries encrypted reasoning.

    A block with no readable text is hidden reasoning too, recorded as an
    opaque part. hyoshigi's Open Responses reader drops an empty reasoning item
    instead [@hyoshigi2026].
    """
    values: Mapping[str, object] = block
    text = values.get("reasoning", "")
    if not isinstance(text, str):
        raise MessageFormatError(path=path, expected="a reasoning block with its text")
    if not text:
        return (HIDDEN_REASONING,)
    return (TextPart(text=text), *((HIDDEN_REASONING,) if has_encrypted_reasoning(values) else ()))


def has_encrypted_reasoning(block: Mapping[str, object]) -> bool:
    """Tell whether a reasoning block carries encrypted reasoning, at its top or in its extras.

    langchain-core's OpenAI translator keeps `encrypted_content` in the extras
    of a reasoning item without a summary, and at the top level of the first
    block of one with a summary [@langchaincore2026].
    """
    extras = block.get("extras")
    nested = extras.get(ENCRYPTED_CONTENT_KEY) if isinstance(extras, Mapping) else None
    return isinstance(block.get(ENCRYPTED_CONTENT_KEY), str) or isinstance(nested, str)


def read_unmapped_reasoning(block: NonStandardContentBlock) -> tuple[Part, ...]:
    """Read reasoning LangChain could not map, as in an Anthropic reply without `model_provider`.

    Anthropic's thinking is text; empty thinking and any other reasoning, such
    as `redacted_thinking`, is opaque.
    """
    match read_unrecognised_block_value(block):
        case {"type": "thinking", "thinking": str(text)} if text:
            return (TextPart(text=text),)
        case {"type": str(kind)} if kind in REASONING_BLOCK_TYPES:
            return (OpaquePart(kind=kind),)
        case _:
            return ()


def read_agent_parts(
    message: AIMessage,
    *,
    blocks: Sequence[ContentBlock],
    server_result: tuple[Part, ...],
) -> tuple[Part, ...]:
    """Read what an AI message says, its blocks in order, then what it holds outside them."""
    path = type(message).__name__
    calls = [*message.tool_calls, *message.invalid_tool_calls]
    call_ids = {call_id for call in calls if (call_id := call["id"])}
    content = tuple(
        part
        for index, block in enumerate(blocks)
        for part in read_agent_block(
            block,
            call_ids=call_ids,
            server_result=server_result,
            path=f"{path}.content_blocks[{index}]",
        )
    )
    return (*content, *read_parts_outside_blocks(message, server_result=server_result))


def read_parts_outside_blocks(
    message: AIMessage,
    *,
    server_result: tuple[Part, ...],
) -> tuple[Part, ...]:
    """Read a refusal kept aside, each malformed call without an id or a name, then grounding.

    Gemini's grounding result is what `server_result` makes a server tool
    result.
    """
    refusal = read_kept_refusal(message)
    kept = () if refusal is None else (TextPart(text=refusal),)
    malformed = tuple(
        INVALID_CALL_PLACEHOLDER
        for call in message.invalid_tool_calls
        if not read_invalid_call(call)
    )
    grounding = server_result if read_grounding_result(message) else ()
    return (*kept, *malformed, *grounding)


def read_agent_block(
    block: ContentBlock,
    *,
    call_ids: set[str],
    server_result: tuple[Part, ...],
    path: str,
) -> tuple[Part, ...]:
    """Read one block of an AI message into the parts it adds."""
    if block["type"] in READ_ELSEWHERE_TYPES:
        return ()
    if block["type"] in TOOL_BLOCK_TYPES:
        return read_tool_block(block, call_ids=call_ids, server_result=server_result, path=path)
    if block["type"] == "non_standard":
        return read_unmapped_agent_block(block, call_ids=call_ids)
    return (read_content_part(block, path=path),)


def read_tool_block(
    block: ContentBlock,
    *,
    call_ids: set[str],
    server_result: tuple[Part, ...],
    path: str,
) -> tuple[Part, ...]:
    """Read a server tool's result, or a call block, which is opaque unless it repeats a call.

    A server tool result is checked as the history checks it, then is what
    `server_result` makes it. A call block whose id is not text, or not among
    the message's tool calls and malformed calls, is an opaque part, so it
    fails closed.
    """
    if block["type"] == "server_tool_result":
        read_server_call_id(block, path=path)
        return server_result
    values: Mapping[str, object] = block
    if is_repeated_tool_call(values, known_call_ids=call_ids):
        return ()
    return (OpaquePart(kind=block["type"]),)


def read_unmapped_agent_block(
    block: NonStandardContentBlock,
    *,
    call_ids: set[str],
) -> tuple[Part, ...]:
    """Read a block of an AI message that LangChain could not map.

    A refusal is text. A block that repeats one of the message's tool calls,
    and reasoning, which the thought holds, add nothing. Any other block is
    opaque.
    """
    value = read_unrecognised_block_value(block)
    if is_repeated_tool_call(value, known_call_ids=call_ids):
        return ()
    match value:
        case {"type": "refusal", "refusal": str(text)}:
            return (TextPart(text=text),)
        case {"type": str(kind)} if kind in REASONING_BLOCK_TYPES:
            return ()
        case _:
            return (OpaquePart(kind=read_unmapped_kind(block)),)


def read_tool_calls(message: AIMessage, *, blocks: Sequence[ContentBlock]) -> tuple[ToolCall, ...]:
    """Read the server tool calls, Gemini's grounding, the tool calls, then the malformed calls."""
    path = type(message).__name__
    client = tuple(
        read_client_call(call, path=f"{path}.tool_calls[{index}]")
        for index, call in enumerate(message.tool_calls)
    )
    malformed = tuple(
        tool_call for call in message.invalid_tool_calls for tool_call in read_invalid_call(call)
    )
    return (
        *read_server_calls(blocks, path=path),
        *read_grounding_call(message),
        *client,
        *malformed,
    )


def read_client_call(call: LangChainToolCall, *, path: str) -> ToolCall:
    """Read a tool call, with its arguments as JSON text; a call without an id raises."""
    call_id = call["id"]
    if not call_id:
        raise MessageFormatError(path=path, expected="a tool call with an id")
    return ToolCall(id=call_id, name=call["name"], arguments=render_arguments(call["args"]))


def read_invalid_call(call: InvalidToolCall) -> tuple[ToolCall, ...]:
    """Read a malformed call with its raw argument text, or nothing without an id or a name."""
    call_id = call["id"]
    name = call["name"]
    if not call_id or not name:
        return ()
    return (ToolCall(id=call_id, name=name, arguments=call["args"] or ""),)


def read_server_calls(blocks: Sequence[ContentBlock], *, path: str) -> Iterator[ToolCall]:
    """Yield the server tool calls among the blocks, in order."""
    for index, block in enumerate(blocks):
        if block["type"] == "server_tool_call":
            yield read_server_call(block, path=f"{path}.content_blocks[{index}]")


def read_server_call(block: ServerToolCall, *, path: str) -> ToolCall:
    """Read a server tool call, with its arguments as JSON text."""
    values: Mapping[str, object] = block
    match values:
        case {"id": str(call_id), "name": str(name), "args": dict() as arguments}:
            return ToolCall(id=call_id, name=name, arguments=render_arguments(arguments))
        case _:
            raise MessageFormatError(
                path=path,
                expected="a server tool call with an id, a name and its arguments",
            )


def read_server_observations(message: AIMessage) -> list[ToolObservation]:
    """Read the server tool results of an AI message, in order, then Gemini's grounding result."""
    path = type(message).__name__
    results = [
        read_server_result(block, path=f"{path}.content_blocks[{index}]")
        for index, block in enumerate(message.content_blocks)
        if block["type"] == "server_tool_result"
    ]
    return [*results, *read_grounding_result(message)]


def read_server_result(block: ContentBlock, *, path: str) -> ToolObservation:
    """Read a server tool's result into the server's observation, an error when it failed."""
    values: Mapping[str, object] = block
    call_id = read_server_call_id(block, path=path)
    output = values.get("output")
    if output is None:
        parts: tuple[Part, ...] = ()
    else:
        parts = (TextPart(text=output if isinstance(output, str) else render_arguments(output)),)
    return ToolObservation(
        call_id=call_id,
        source=SERVER_SOURCE,
        is_error=values.get("status") == "error",
        parts=parts,
    )


def read_server_call_id(block: ContentBlock, *, path: str) -> str:
    """Return the id of the call a server tool's result answers; a result without one raises."""
    values: Mapping[str, object] = block
    call_id = values.get("tool_call_id")
    if not isinstance(call_id, str):
        raise MessageFormatError(path=path, expected="a server tool result with its call id")
    return call_id


def read_grounding_metadata(message: AIMessage) -> Mapping[str, object]:
    """Return the `grounding_metadata` of a Gemini reply, or nothing.

    langchain-google-genai keeps the searches of Gemini's grounding tools
    there, with no server tool call block, only for a grounded reply, and
    dumps every field of google-genai's `GroundingMetadata`
    [@langchaingooglegenai2026]. So metadata that holds anything is one
    server tool use, whatever its keys. Raises `MessageFormatError` on
    metadata that is not a mapping; only an absent key or None means no
    grounding.
    """
    metadata = message.response_metadata.get(GROUNDING_METADATA_KEY)
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise MessageFormatError(
            path=f"{type(message).__name__}.response_metadata.{GROUNDING_METADATA_KEY}",
            expected="a mapping of grounding metadata",
        )
    return metadata


def build_grounding_call_id(message: AIMessage) -> str:
    """Return the id of the grounding call, named by its path: the message's id and the key."""
    return f"{message.id}.{GROUNDING_METADATA_KEY}"


def read_grounding_call(message: AIMessage) -> tuple[ToolCall, ...]:
    """Read a Gemini reply's grounding as one server tool call when its metadata holds anything.

    The arguments hold each list of queries the metadata holds, by its field.
    """
    metadata = read_grounding_metadata(message)
    if not metadata:
        return ()
    queries = {key: metadata[key] for key in GROUNDING_QUERY_KEYS if metadata.get(key)}
    call = ToolCall(
        id=build_grounding_call_id(message),
        name=GROUNDING_TOOL_NAME,
        arguments=render_arguments(queries),
    )
    return (call,)


def read_grounding_result(message: AIMessage) -> tuple[ToolObservation, ...]:
    """Read what a Gemini reply's grounding found as the server's observation.

    The observation is there whenever the grounding call is, without parts
    when the searches found nothing.
    """
    metadata = read_grounding_metadata(message)
    if not metadata:
        return ()
    sources = metadata.get(GROUNDING_SOURCES_KEY)
    parts = (TextPart(text=render_arguments(sources)),) if sources else ()
    observation = ToolObservation(
        call_id=build_grounding_call_id(message),
        source=SERVER_SOURCE,
        parts=parts,
    )
    return (observation,)


def render_arguments(value: object) -> str:
    """Render a value as JSON text in the order it has, with anything JSON cannot hold as text."""
    return json.dumps(value, ensure_ascii=False, default=str)
