"""What writes a step when no untrusted sample is acceptable.

A control protocol hands the step to a `Fallback` once it has rejected every
sample, or once Auto Mode reaches its block limits. Fallbacks hold
configuration only, so one instance can serve parallel runs.

A fallback must not escalate to a person by calling LangGraph's `interrupt()`.
The fallback runs inside the model node, and on resume LangGraph runs the
whole node again [@langgraph2026]: the protocol draws fresh samples, and the
person's answer applies to whatever the new samples propose, not to the step
they reviewed. To ask a person, end the step instead, for example with
`HaltRun`, and ask outside the model node, as `HumanInTheLoopMiddleware` does
for tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from uuid import uuid4

from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Fallback,
    Outcome,
    PendingStep,
    Sample,
    StepDecision,
)
from langchain_sync_monitors.model_calls import resolve_chat_model

DEFAULT_HALT_MESSAGE: Final = (
    "[Safety monitor] This run was stopped because the agent's proposed steps looked "
    "unsafe. The rejected steps are recorded in monitor_log for review."
)
"""The final message `HaltRun` writes when it is given no message of its own."""


class DeferToTrustedModel(Fallback):
    """The trusted model writes the step, with the same conversation and tools.

    This is the fallback of Defer to Trusted [@greenblatt2023aicontrol] and of
    Ctrl-Z's Defer to Resample [@bhatt2025ctrlz]. The trusted model never sees
    the rejected samples. A model string is resolved once, when the fallback is
    built.
    """

    def __init__(self, *, trusted_model: str | BaseChatModel) -> None:
        """Keep the trusted model, initialising it first when given a provider string."""
        self.trusted_model = resolve_chat_model(trusted_model)

    async def take_over(
        self,
        step: PendingStep,
        *,
        samples: tuple[Sample, ...],
        flagged: bool,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
    ) -> StepDecision:
        """Have the trusted model write the step, keeping the samples as audit evidence."""
        response = await step.request_trusted_step(trusted_model=self.trusted_model)
        return StepDecision(
            outcome=Outcome.DEFERRED_TO_TRUSTED,
            response=response,
            samples=samples,
            executed_sample=None,
            flagged=flagged,
            blocked_attempts=blocked_attempts,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HaltRun(Fallback):
    """End the agent's run with a final message.

    The message has no tool calls, and the monitor middleware routes the agent
    to its end after a halted step, so the run stops there even in an agent
    that would otherwise loop until it has a structured response. A halted
    step is always flagged, since a person should look at why the run stopped.
    """

    message: str = DEFAULT_HALT_MESSAGE

    async def take_over(
        self,
        step: PendingStep,
        *,
        samples: tuple[Sample, ...],
        flagged: bool,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
    ) -> StepDecision:
        """Write the halt message as the step, keeping the samples as audit evidence.

        The message gets a fresh id on every call, because LangGraph's message
        reducer replaces messages that share an id.
        """
        halt_message = AIMessage(content=self.message, id=f"monitor-{uuid4()}")
        return StepDecision(
            outcome=Outcome.HALTED,
            response=ModelResponse(result=[halt_message]),
            samples=samples,
            executed_sample=None,
            flagged=True,
            blocked_attempts=blocked_attempts,
        )
