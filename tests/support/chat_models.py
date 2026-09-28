"""A scripted chat model for offline tests of agents and middleware."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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

    def _build_next_result(self, messages: list[BaseMessage]) -> ChatResult:
        with self._lock:
            self.calls.append(list(messages))
            if not self.responses:
                message = f"ScriptedChatModel ran out of responses after {len(self.calls)} calls"
                raise AssertionError(message)
            response = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=response.model_copy(deep=True))])

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
