"""A chat model and a monitor that fail on cue, for tests of steps that raise partway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.transcript import render_proposal_for_audit


class FlakyChatModel(BaseChatModel):
    """Returns queued replies in order, raising a queued exception in place of a reply.

    `delays` sets how long each async call sleeps first, by position.
    `started_calls` counts every call and `finished_calls` the calls that
    returned, so a test can see calls that kept running after another failed.
    """

    replies: list[Any]
    delays: list[float] = Field(default_factory=list)
    started_calls: int = 0
    finished_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def take_next_reply(self) -> tuple[Any, float]:
        position = self.started_calls
        self.started_calls += 1
        if position >= len(self.replies):
            message = f"FlakyChatModel ran out of replies after {position} calls"
            raise AssertionError(message)
        delay = self.delays[position] if position < len(self.delays) else 0.0
        return self.replies[position], delay

    def finish_reply(self, reply: Any) -> ChatResult:
        if isinstance(reply, BaseException):
            raise reply
        self.finished_calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy(deep=True))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply, _delay = self.take_next_reply()
        return self.finish_reply(reply)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply, delay = self.take_next_reply()
        if delay:
            await asyncio.sleep(delay)
        return self.finish_reply(reply)


@dataclass(kw_only=True)
class TriggeredFailureMonitor(Monitor):
    """Delegates to `inner`, but raises `TimeoutError` on a proposal that contains `trigger`."""

    inner: Monitor
    trigger: str

    def check(self, monitor_input: MonitorInput) -> None:
        if self.trigger in render_proposal_for_audit(monitor_input.proposal):
            message = "monitor provider timed out"
            raise TimeoutError(message)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        self.check(monitor_input)
        return await self.inner.evaluate(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        self.check(monitor_input)
        return self.inner.evaluate_sync(monitor_input)
