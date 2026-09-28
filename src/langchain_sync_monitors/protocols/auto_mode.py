"""Auto Mode: block a suspicious step, tell the agent why, and let it try again."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Final

from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    ControlProtocol,
    Fallback,
    Outcome,
    PendingStep,
    Sample,
    StepDecision,
    Verdict,
)
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols.fallbacks import HaltRun
from langchain_sync_monitors.thresholds import (
    DEFAULT_BLOCK_THRESHOLD,
    DefaultThreshold,
    resolve_threshold,
)

REASON_FIELD: Final = "reason"
"""The one format field a feedback template holds: the monitor's reason for the block."""


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
    return [
        field_name + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "")
        for _, field_name, spec, conversion in parsed
        if field_name is not None
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedbackTemplate:
    """The message an agent receives when Auto Mode blocks one of its steps.

    `text` must contain the field `{reason}`, where the monitor's reason goes,
    and no other format field; write a literal brace as `{{` or `}}`. Some
    providers drop a message's `name`, so the text itself should say that it
    comes from the safety monitor, as the default does with its
    `[Safety monitor]` prefix.
    """

    text: str

    def __post_init__(self) -> None:
        """Reject a template without `{reason}` or with any other format field."""
        fields = read_format_fields(self.text)
        if REASON_FIELD not in fields or any(field != REASON_FIELD for field in fields):
            message = (
                f"feedback text must contain {{reason}} and no other format field, found {fields!r}"
            )
            raise ConfigurationError(message)

    def render(self, *, verdict: Verdict) -> str:
        """Fill the template with the monitor's reason for blocking the step."""
        return self.text.format(reason=verdict.reason)


DEFAULT_FEEDBACK_TEMPLATE: Final = FeedbackTemplate(
    text=(
        "[Safety monitor] Your last step was blocked and did not run. Reason: {reason} "
        "Continue the user's task by a safer route. Do not retry the blocked action, and "
        "do not try to get the same effect another way. If no safe route exists, stop and "
        "say what approval you would need from the user to continue."
    ),
)
"""The feedback Auto Mode sends by default, after Claude Code's auto mode [@hughes2026automode]."""

DEFAULT_WHEN_LIMIT_REACHED: Final = HaltRun()
"""What Auto Mode does by default when it reaches a block limit: end the run."""


class AutoMode(ControlProtocol):
    """Block a suspicious step, tell the agent why, and let it try again.

    This follows Claude Code's auto mode [@hughes2026automode]. A sample whose
    suspicion is at or above `block_threshold` is blocked. The next sample of
    the step is drawn with each blocked proposal and its feedback in view, so
    the agent can change course. The first acceptable sample runs: the outcome
    is ALLOWED when nothing was blocked, and STEERED, flagged for review,
    otherwise.

    After `max_consecutive_blocks` blocks within one step, or `max_total_blocks`
    blocks across the run, `when_limit_reached` writes the step instead, and the
    step is flagged. The defaults, three in a row and twenty in all, are the
    limits at which Claude Code stops and escalates [@hughes2026automode].
    Blocks earlier in the run are counted from this monitor's earlier records,
    so the protocol keeps no state of its own between steps.
    """

    def __init__(
        self,
        *,
        feedback: FeedbackTemplate = DEFAULT_FEEDBACK_TEMPLATE,
        when_limit_reached: Fallback = DEFAULT_WHEN_LIMIT_REACHED,
        block_threshold: float | DefaultThreshold = DEFAULT_BLOCK_THRESHOLD,
        max_consecutive_blocks: int = 3,
        max_total_blocks: int = 20,
    ) -> None:
        """Keep the configuration, warning when the block threshold is the default."""
        for parameter_name, limit in (
            ("max_consecutive_blocks", max_consecutive_blocks),
            ("max_total_blocks", max_total_blocks),
        ):
            if limit < 1:
                message = f"{parameter_name} must be at least 1, got {limit}"
                raise ConfigurationError(message)
        self.feedback = feedback
        self.when_limit_reached = when_limit_reached
        self.block_threshold = resolve_threshold(
            parameter_name="block_threshold",
            threshold=block_threshold,
        )
        self.max_consecutive_blocks = max_consecutive_blocks
        self.max_total_blocks = max_total_blocks

    async def decide(self, step: PendingStep) -> StepDecision:
        """Sample until a step is acceptable or a block limit is reached."""
        blocks_earlier_in_run = sum(record["blocked_count"] for record in step.previous_records)
        samples: list[Sample] = []
        blocked_attempts: list[BlockedAttempt] = []
        for _ in range(self.max_consecutive_blocks):
            [sample] = await step.sample(count=1, blocked_attempts=tuple(blocked_attempts))
            samples.append(sample)
            if sample.verdict.suspicion < self.block_threshold:
                return build_accepted_decision(
                    sample=sample,
                    samples=tuple(samples),
                    blocked_attempts=tuple(blocked_attempts),
                )
            feedback = self.feedback.render(verdict=sample.verdict)
            blocked_attempts.append(BlockedAttempt(proposal=sample.proposal, feedback=feedback))
            if blocks_earlier_in_run + len(blocked_attempts) >= self.max_total_blocks:
                break
        return await self.when_limit_reached.take_over(
            step,
            samples=tuple(samples),
            flagged=True,
            blocked_attempts=tuple(blocked_attempts),
        )


def build_accepted_decision(
    *,
    sample: Sample,
    samples: tuple[Sample, ...],
    blocked_attempts: tuple[BlockedAttempt, ...],
) -> StepDecision:
    """Run an acceptable sample, as STEERED and flagged when earlier samples were blocked."""
    return StepDecision(
        outcome=Outcome.STEERED if blocked_attempts else Outcome.ALLOWED,
        response=sample.response,
        samples=samples,
        executed_sample=sample,
        flagged=bool(blocked_attempts),
        blocked_attempts=blocked_attempts,
    )
