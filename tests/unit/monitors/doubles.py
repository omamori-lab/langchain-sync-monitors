"""Test doubles for the monitor tests: a scripted chat model and a scripted monitor."""

from __future__ import annotations

from typing import Literal

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict

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


class KeywordRejectingChatModel(ScriptedChatModel):
    """Raises `TypeError(rejection)` on a call that passes `logprobs`, or on every call.

    langchain-anthropic passes bound keywords on to the Anthropic SDK, whose
    `Messages.create()` takes no `logprobs`, so Python raises before any
    request is sent.
    """

    rejection: str = "Messages.create() got an unexpected keyword argument 'logprobs'"
    rejects_every_call: bool = False

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.rejects_every_call or "logprobs" in kwargs:
            raise TypeError(self.rejection)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ScriptedMonitor(Monitor):
    """Returns the given suspicions in turn, one per call, on both paths."""

    def __init__(self, *, suspicions: list[float]) -> None:
        self.suspicions = suspicions
        self.calls = 0

    def build_next_verdict(self) -> Verdict:
        """Return the next scripted verdict and count the call."""
        suspicion = self.suspicions[self.calls % len(self.suspicions)]
        self.calls += 1
        return Verdict(suspicion=suspicion, reason=f"call {self.calls} scored {suspicion}")

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return self.build_next_verdict()

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return self.build_next_verdict()


async def evaluate_on_path(
    monitor: Monitor,
    monitor_input: MonitorInput,
    *,
    call_path: CallPath,
) -> Verdict:
    """Score the step through `evaluate` or `evaluate_sync`, as the test asks."""
    if call_path == "async":
        return await monitor.evaluate(monitor_input)
    return monitor.evaluate_sync(monitor_input)
