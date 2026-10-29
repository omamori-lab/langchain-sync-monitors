"""Control protocols' settings, and the messages the middleware writes for their decisions.

A control protocol decides what becomes of each step the agent proposes:
it runs, is replaced, is blocked with feedback, or is held for a person
[@hyoshigi2026]. The decisions are hyoshigi's, and the middleware enforces
them. Each class here holds one protocol's settings, under hyoshigi's names,
defaults and meanings, and builds hyoshigi's sync and async protocol around a
monitor's twins, so one middleware serves `invoke()` and `ainvoke()`. A
setting hyoshigi refuses raises `ConfigurationError` when the middleware
is built, since a protocol is built with its monitor.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Final, assert_never

import hyoshigi
from hyoshigi import AuditResponse, FailureResponse, Resampling

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.options import check_instance_option, describe_option_value
from langchain_sync_monitors.spans import TracedTwins

REASON_FIELD: Final = "reason"
"""The one format field a feedback template holds: the reason for the block."""

DEFAULT_HALT_MESSAGE: Final = (
    "[Safety monitor] This run was stopped because the agent's proposed steps looked "
    "unsafe. The rejected steps are recorded in monitor_log for review."
)
"""The final message of a step the protocol's decision halts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProtocolTwins:
    """A protocol's sync twin, which decides under `invoke()`, and its async twin."""

    protocol: hyoshigi.DeferToResample | hyoshigi.BlockAndFeedBack
    async_protocol: hyoshigi.AsyncDeferToResample | hyoshigi.AsyncBlockAndFeedBack

    def can_ask_for_trusted_steps(self) -> bool:
        """Tell whether the protocol can ask for a trusted step.

        Defer to resample asks for one when the step's score lies from the
        defer threshold up to the audit threshold, and when its monitor got no
        answer under `FailureResponse.TRUSTED_STEP`; block and feed back never
        does.
        """
        match self.protocol:
            case hyoshigi.DeferToResample():
                is_deferring = self.protocol.defer_threshold < self.protocol.audit_threshold
                return is_deferring or self.protocol.on_unavailable is FailureResponse.TRUSTED_STEP
            case hyoshigi.BlockAndFeedBack():
                return False
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferToResample:
    """Resample a suspicious step, run the least suspicious sample, and defer or audit the rest.

    A step whose score is at or above `defer_threshold` is resampled, up to
    `max_resamples` times, one at a time under `Resampling.SEQUENTIAL`, and
    the least suspicious sample below the threshold runs. When none is below
    it, the trusted model writes the step, the middleware's `trusted_model`,
    or, without one, a person audits it. A step whose lowest score is at or
    above `audit_threshold` is audited; with `on_audit=AuditResponse.FLAG` it
    runs, flagged, instead. With `max_resamples=0` this is defer to trusted,
    and with equal thresholds as well, trusted monitoring. This is Ctrl-Z's
    protocol [@bhatt2025ctrlz], as hyoshigi's `DeferToResample` decides it
    [@hyoshigi2026].
    """

    defer_threshold: float
    audit_threshold: float
    max_resamples: int
    on_unavailable: FailureResponse = FailureResponse.AUDIT
    resampling: Resampling = Resampling.SEQUENTIAL
    on_audit: AuditResponse = AuditResponse.HOLD

    def build(self, monitors: TracedTwins) -> ProtocolTwins:
        """Build hyoshigi's sync and async protocols around the monitor's twins."""
        protocol = hyoshigi.DeferToResample(
            monitor=monitors.monitor,
            defer_threshold=self.defer_threshold,
            audit_threshold=self.audit_threshold,
            max_resamples=self.max_resamples,
            on_unavailable=self.on_unavailable,
            resampling=self.resampling,
            on_audit=self.on_audit,
        )
        async_protocol = hyoshigi.AsyncDeferToResample(
            monitor=monitors.async_monitor,
            defer_threshold=protocol.defer_threshold,
            audit_threshold=protocol.audit_threshold,
            max_resamples=protocol.max_resamples,
            on_unavailable=protocol.on_unavailable,
            resampling=protocol.resampling,
            on_audit=protocol.on_audit,
        )
        return ProtocolTwins(protocol=protocol, async_protocol=async_protocol)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockAndFeedBack:
    """Block a suspicious step, tell the agent why, and audit the next step after too many blocks.

    A step whose score is at or above `block_threshold` is blocked, and the
    agent reads the middleware's `feedback` and proposes again. Once the
    agent's steps hold `blocks_in_a_row` blocks in a row, or its run's
    `blocks_in_total` in total, subagents' included, the next step is
    audited. This follows Claude Code's auto mode [@hughes2026automode], as
    hyoshigi's `BlockAndFeedBack` decides it [@hyoshigi2026].
    """

    block_threshold: float
    blocks_in_a_row: int = 3
    blocks_in_total: int = 20
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def build(self, monitors: TracedTwins) -> ProtocolTwins:
        """Build hyoshigi's sync and async protocols around the monitor's twins."""
        protocol = hyoshigi.BlockAndFeedBack(
            monitor=monitors.monitor,
            block_threshold=self.block_threshold,
            blocks_in_a_row=self.blocks_in_a_row,
            blocks_in_total=self.blocks_in_total,
            on_unavailable=self.on_unavailable,
        )
        async_protocol = hyoshigi.AsyncBlockAndFeedBack(
            monitor=monitors.async_monitor,
            block_threshold=protocol.block_threshold,
            blocks_in_a_row=protocol.blocks_in_a_row,
            blocks_in_total=protocol.blocks_in_total,
            on_unavailable=protocol.on_unavailable,
        )
        return ProtocolTwins(protocol=protocol, async_protocol=async_protocol)


type ProtocolConfiguration = DeferToResample | BlockAndFeedBack
"""A protocol the middleware takes."""

PROTOCOL_CONFIGURATIONS: Final = (DeferToResample, BlockAndFeedBack)
"""The classes of `ProtocolConfiguration`, for checks at run time."""


def check_protocol_option(value: object, *, parameter_name: str) -> None:
    """Raise `ConfigurationError` unless `value` is one of the protocol configurations."""
    if isinstance(value, PROTOCOL_CONFIGURATIONS):
        return
    names = ", ".join(protocol_class.__name__ for protocol_class in PROTOCOL_CONFIGURATIONS)
    message = f"{parameter_name} must be one of {names}, got {describe_option_value(value)}"
    raise ConfigurationError(message)


def check_trusted_model_use(protocols: ProtocolTwins, *, protocol_name: str) -> None:
    """Raise `ConfigurationError` for a trusted model the protocol never asks for."""
    if protocols.can_ask_for_trusted_steps():
        return
    message = (
        f"trusted_model is given, but {protocol_name} with these settings never asks for a "
        "trusted step, so the trusted model would never be called. Leave trusted_model out, "
        "or give DeferToResample a defer_threshold below its audit_threshold, or "
        "on_unavailable=FailureResponse.TRUSTED_STEP."
    )
    raise ConfigurationError(message)


def read_format_fields(text: str) -> list[str]:
    """Return every format field in `text` as written, such as `reason` or `reason!r`.

    A malformed template, such as one with an unmatched brace, raises
    `ConfigurationError`.
    """
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as error:
        message = f"feedback text is not a valid format string: {error}"
        raise ConfigurationError(message) from error
    # Each field keeps its conversion and format spec, so `{reason!r}` is not taken for `{reason}`.
    return [
        field_name + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "")
        for _, field_name, spec, conversion in parsed
        if field_name is not None
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackTemplate:
    """The message an agent receives when the protocol blocks one of its steps.

    `text` must contain the field `{reason}`, where the monitor's reason goes,
    and no other format field; write a literal brace as `{{` or `}}`. Some
    providers drop a message's `name`, so the text itself should say that it
    comes from the safety monitor, as the default does with its
    `[Safety monitor]` prefix.
    """

    text: str

    def __post_init__(self) -> None:
        """Reject a text that is not a string, lacks `{reason}` or has any other format field."""
        check_instance_option(self.text, option_type=str, parameter_name="text")
        fields = read_format_fields(self.text)
        if REASON_FIELD not in fields or any(field != REASON_FIELD for field in fields):
            message = (
                f"feedback text must contain {{reason}} and no other format field, found {fields!r}"
            )
            raise ConfigurationError(message)

    def render(self, *, reason: str) -> str:
        """Fill the template with the reason for the block."""
        return self.text.format(reason=reason)


DEFAULT_FEEDBACK_TEMPLATE: Final = FeedbackTemplate(
    text=(
        "[Safety monitor] Your last step was blocked, and none of your tools ran it.\n"
        "Reason: {reason}\n"
        "Continue the user's task by a safer route. Do not retry the blocked action, and "
        "do not try to get the same effect another way. If no safe route exists, stop and "
        "say what approval you would need from the user to continue."
    ),
)
"""The feedback a block sends by default, after Claude Code's auto mode [@hughes2026automode].

Claude Code's denial asks the agent to find a safer path rather than route
around the block (section "Deny-and-continue") [@hughes2026automode]. This
feedback says that none of the agent's tools ran the blocked step, which
stays true when a server tool in that step already ran inside the model call.
The reason is the monitor's explanation, which carries no score.
"""
