"""A scripted chat model for offline tests of agents and middleware."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Returns queued AI messages in order and records every conversation it receives.

    `bind_tools` returns the model itself, so it works inside `create_agent`
    and `create_deep_agent`. `delay` makes async calls sleep first, so
    concurrent draws overlap.
    """

    responses: list[AIMessage]
    delay: float = 0.0
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        self.bound_tool_names.append(names)
        return self

    def _take_next_response(self, messages: list[BaseMessage]) -> AIMessage:
        with self._lock:
            self.calls.append(list(messages))
            if not self.responses:
                message = f"ScriptedChatModel ran out of responses after {len(self.calls)} calls"
                raise AssertionError(message)
            response = self.responses.pop(0)
        return response.model_copy(deep=True)

    def _build_next_result(self, messages: list[BaseMessage]) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._take_next_response(messages))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._build_next_result(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._build_next_result(messages)


class OverlapCountingChatModel(ScriptedChatModel):
    """A scripted model that counts the most calls it was answering at once.

    Give it a `delay`, so that async calls drawn at once overlap.
    """

    answering: int = 0
    most_at_once: int = 0

    def start_call(self) -> None:
        self.answering += 1
        self.most_at_once = max(self.most_at_once, self.answering)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.start_call()
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        finally:
            self.answering -= 1

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.start_call()
        try:
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        finally:
            self.answering -= 1


class SummarisingScriptedChatModel(ScriptedChatModel):
    """A scripted agent model that also writes the summaries Deep Agents asks it for.

    Deep Agents summarises the history with the agent's own model, in a
    request of one message, while every agent step carries the system prompt.
    How often it summarises depends on the Deep Agents version, so a summary
    request always gets `summary` and never uses up a scripted step.
    """

    summary: AIMessage

    def _take_next_response(self, messages: list[BaseMessage]) -> AIMessage:
        if len(messages) != 1:
            return super()._take_next_response(messages)
        with self._lock:
            self.calls.append(list(messages))
        return self.summary.model_copy(deep=True)


class StreamingScriptedChatModel(ScriptedChatModel):
    """A scripted model that streams each reply as providers do.

    The text arrives word by word, and the tool calls arrive in a last chunk,
    so a test sees the tokens a `stream_mode="messages"` consumer would.
    """

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in build_reply_chunks(self._take_next_response(messages)):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.delay:
            await asyncio.sleep(self.delay)
        for chunk in build_reply_chunks(self._take_next_response(messages)):
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def build_reply_chunks(reply: AIMessage) -> list[ChatGenerationChunk]:
    """Split a reply into one chunk per word, then one last chunk with its tool calls."""
    words = [word for word in reply.text.split(" ") if word]
    chunks = [
        AIMessageChunk(content=word if index == 0 else f" {word}")
        for index, word in enumerate(words)
    ]
    tool_call_chunks = [
        tool_call_chunk(
            name=call["name"], args=json.dumps(call["args"]), id=call["id"], index=index
        )
        for index, call in enumerate(reply.tool_calls)
    ]
    chunks.append(
        AIMessageChunk(content="", tool_call_chunks=tool_call_chunks, chunk_position="last")
    )
    return [ChatGenerationChunk(message=chunk) for chunk in chunks]


def build_tool_call_message(
    *,
    tool_name: str,
    call_id: str,
    arguments: dict[str, Any] | None = None,
    content: str = "",
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": tool_name, "args": arguments or {}, "id": call_id, "type": "tool_call"},
        ],
    )
