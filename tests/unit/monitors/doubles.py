"""Doubles for the monitor tests: a scripted chat model, a rating on either side, a log reader."""

from __future__ import annotations

from typing import Literal

import pytest
from hyoshigi import Step, SuspicionScore
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langchain_sync_monitors.monitors import MonitorConfiguration

type CallPath = Literal["async", "sync"]

PLANTED_SECRET = "planted-secret-9f3c1e"
"""A credential the transcript holds, in the user's words and in a tool call's argument."""


class ScriptedChatModel(BaseChatModel):
    """Replies with the scripted messages in turn and records every call it receives."""

    replies: list[str | AIMessage]
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)
    received_options: list[dict[str, object]] = Field(default_factory=list)
    received_metadata: list[dict[str, object]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        index = len(self.received_messages)
        self.received_messages.append(messages)
        self.received_options.append(dict(kwargs))
        self.received_metadata.append(dict(run_manager.metadata) if run_manager else {})
        reply = self.replies[index % len(self.replies)]
        message = AIMessage(content=reply) if isinstance(reply, str) else reply
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        # Stay on the event loop: the default runs _generate in threads, which
        # would race on the call counter.
        sync_manager = run_manager.get_sync() if run_manager else None
        return self._generate(messages, stop=stop, run_manager=sync_manager, **kwargs)


async def rate_on_path(
    monitor: MonitorConfiguration,
    step: Step,
    *,
    call_path: CallPath,
) -> SuspicionScore:
    """Rate the step's action with the monitor's async or sync twin, as the test asks."""
    twins = monitor.twins
    if call_path == "async":
        return await twins.async_monitor.rate(step.action, trajectory=step.trajectory)
    return twins.monitor.rate(step.action, trajectory=step.trajectory)


def read_logged_lines(caplog: pytest.LogCaptureFixture, *, logger: str) -> list[tuple[str, str]]:
    """Return the level and text of every line the named logger wrote, in order."""
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == logger
    ]
