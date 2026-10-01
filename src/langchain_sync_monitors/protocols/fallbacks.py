"""What writes a step when no untrusted sample is acceptable.

A control protocol hands the step to a `Fallback` once it has rejected every
sample, or once Auto Mode reaches its block limits. Fallbacks hold
configuration only, so one instance can serve parallel runs.

A fallback must not escalate to a person by calling LangGraph's `interrupt()`,
and neither may a monitor or a protocol. The fallback runs inside the model
node, and on resume LangGraph runs the whole node again [@langgraph2026]: the
protocol draws fresh samples, the monitor judges them again, and the person's
answer applies to whatever the new samples propose, not to the step they
reviewed. To ask a person, end the step instead, for example with `HaltRun`,
and ask outside the model node, as `HumanInTheLoopMiddleware` does for tool
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    ControlProtocol,
    Fallback,
    Outcome,
    PendingStep,
    Sample,
    StepDecision,
)
from langchain_sync_monitors.feedback import build_halt_decision
from langchain_sync_monitors.model_calls import resolve_chat_model
from langchain_sync_monitors.options import check_instance_option

DEFAULT_HALT_MESSAGE: Final = (
    "[Safety monitor] This run was stopped because the agent's proposed steps looked "
    "unsafe. The rejected steps are recorded in monitor_log for review."
)
"""The final message `HaltRun` writes when it is given no message of its own."""


def check_fallback_option(value: object, *, parameter_name: str) -> None:
    """Raise `ConfigurationError` unless `value` is a `Fallback`, naming a protocol mix-up.

    `DeferToTrusted` is a protocol and `DeferToTrustedModel` its fallback, and
    both take `trusted_model`, so the two are easy to swap. A protocol given
    here would fail only when the fallback is first needed.
    """
    hint = "Use HaltRun() or DeferToTrustedModel(trusted_model=...)."
    if isinstance(value, ControlProtocol):
        hint = (
            f"{type(value).__name__} is a control protocol, which decides whole steps. To "
            "have the trusted model write the step, use DeferToTrustedModel(trusted_model=...)."
        )
    check_instance_option(value, option_type=Fallback, parameter_name=parameter_name, hint=hint)


class DeferToTrustedModel(Fallback):
    """The trusted model writes the step, with the same conversation and tools.

    This is the fallback of Defer to Trusted [@greenblatt2023aicontrol]
    (section 3.3) and of Ctrl-Z's Defer to Resample [@bhatt2025ctrlz]
    (section 3.2). The trusted model never sees
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

    The halt stands until the thread records another run input. When a hook
    sends the finished run back to the model, as Deep Agents'
    `RubricMiddleware` does [@deepagents2026], the monitor middleware halts
    each further step itself, without drawing a sample, and with a final
    message of its own.

    The flag, the standing halt and the use of `HaltRun` as Defer to
    Resample's fallback are the library's own rules; the design explanation's
    "Protocols and outcomes" section says why.
    """

    message: str = DEFAULT_HALT_MESSAGE

    def __post_init__(self) -> None:
        """Reject a message that is not a string, which could not be the final message."""
        check_instance_option(self.message, option_type=str, parameter_name="message")

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
        return build_halt_decision(self.message, samples=samples, blocked_attempts=blocked_attempts)
