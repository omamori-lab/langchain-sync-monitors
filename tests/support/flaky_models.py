"""A chat model and a monitor that fail on cue, for tests of steps that raise partway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import hyoshigi
from hyoshigi import Action, SuspicionScore, Trajectory
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_sync_monitors import MonitorTwins
from tests.support.monitors import read_action_text


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


def check_trigger(action: Action, *, trigger: str) -> None:
    """Raise `TimeoutError` when the action holds the trigger."""
    if trigger in read_action_text(action):
        message = "monitor provider timed out"
        raise TimeoutError(message)


@dataclass(frozen=True, kw_only=True)
class TriggeredFailureRater(hyoshigi.Monitor):
    """Rates with `inner`, but raises `TimeoutError` on an action that holds `trigger`."""

    inner: hyoshigi.Monitor
    trigger: str

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        check_trigger(action, trigger=self.trigger)
        return self.inner.rate(action, trajectory=trajectory)


@dataclass(frozen=True, kw_only=True)
class AsyncTriggeredFailureRater(hyoshigi.AsyncMonitor):
    """Rates as `TriggeredFailureRater` does, awaiting `inner`."""

    inner: hyoshigi.AsyncMonitor
    trigger: str

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        check_trigger(action, trigger=self.trigger)
        return await self.inner.rate(action, trajectory=trajectory)


@dataclass(frozen=True, kw_only=True)
class TriggeredFailureMonitor(MonitorTwins):
    """The twins of `inner`, raising `TimeoutError` on an action that holds `trigger`."""

    inner: MonitorTwins
    trigger: str
    monitor: hyoshigi.Monitor = field(init=False)
    async_monitor: hyoshigi.AsyncMonitor = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "monitor",
            TriggeredFailureRater(inner=self.inner.monitor, trigger=self.trigger),
        )
        object.__setattr__(
            self,
            "async_monitor",
            AsyncTriggeredFailureRater(inner=self.inner.async_monitor, trigger=self.trigger),
        )
        super().__post_init__()
