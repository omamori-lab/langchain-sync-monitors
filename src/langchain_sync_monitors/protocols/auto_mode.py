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
from langchain_sync_monitors.options import (
    check_instance_option,
    read_count_option,
    read_limit_option,
)
from langchain_sync_monitors.protocols.fallbacks import HaltRun, check_fallback_option
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
    # Each field keeps its conversion and format spec, so `{reason!r}` is not taken for `{reason}`.
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
        """Reject a text that is not a string, lacks `{reason}` or has any other format field."""
        check_instance_option(self.text, option_type=str, parameter_name="text")
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
        "[Safety monitor] Your last step was blocked, and none of your tools ran it.\n"
        "Reason: {reason}\n"
        "Continue the user's task by a safer route. Do not retry the blocked action, and "
        "do not try to get the same effect another way. If no safe route exists, stop and "
        "say what approval you would need from the user to continue."
    ),
)
"""The feedback Auto Mode sends by default, after Claude Code's auto mode [@hughes2026automode].

It says that none of the agent's tools ran the blocked step, which stays true
when a provider's built-in tool in that step already ran inside the model
call. Like every reason, it carries no score.
"""

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
    blocks across the conversation thread, `when_limit_reached` writes the step
    instead, and the step is flagged. The defaults, three in a row and twenty
    in all, are the limits at which Claude Code stops and escalates within a
    session [@hughes2026automode].

    The total counts this monitor's blocks in the thread's `monitor_log`: this
    agent's earlier steps, which a checkpointed thread keeps across its turns,
    and the steps of the subagents it started, nested ones included, which
    reach the log when each subagent returns. A subagent starts from the total
    recorded before it was started, so delegating again does not reset the
    budget. When blocks inside subagents since this agent's last step leave
    the thread at or over the total, this agent's next step goes to
    `when_limit_reached` without being sampled. Subagents that run in parallel
    do not see each other's blocks, so together they can pass the total; their
    parent counts every one of them at its next step. The total never resets:
    once a thread has reached it, the first block of every later step goes to
    `when_limit_reached`. The protocol keeps no state of its own between steps.

    A subagent whose run raises returns no records, so the total misses the
    blocks it recorded. They count only when the failed run is resumed from
    its checkpoint with `None` as input. They never count when the thread goes
    on with new input, or when a middleware such as LangChain's
    `ToolRetryMiddleware` or `ToolErrorMiddleware` answers the failed call
    with an error message or runs it again; a retry starts the subagent again
    from the same count. `check_monitor_placement` warns about such
    middleware.

    A second monitor placed inside this one returns its record as a command,
    and LangChain keeps the commands of the last model call only, so that
    monitor loses its judgement of every blocked sample, unless
    `max_consecutive_blocks` or `max_total_blocks` is 1 and
    `when_limit_reached` is a `HaltRun` (not a subclass), so that a step draws
    at most one sample.
    Inside another monitor, this one should keep
    `FeedbackVisibility.HIDDEN`: with `IN_TRANSCRIPT` its blocked attempts
    come first in the step, and the outer monitor judges the first blocked
    proposal instead of the step that runs. `check_monitor_placement` warns
    about both.

    Each option is checked when the protocol is built: `feedback` must be a
    `FeedbackTemplate`, `when_limit_reached` a `Fallback`, and each limit a
    whole number of at least 1, numpy's integers included; `max_total_blocks`
    may also be `math.inf`, for no total limit. Anything else raises
    `ConfigurationError`.
    """

    def __init__(
        self,
        *,
        feedback: FeedbackTemplate = DEFAULT_FEEDBACK_TEMPLATE,
        when_limit_reached: Fallback = DEFAULT_WHEN_LIMIT_REACHED,
        block_threshold: float | DefaultThreshold = DEFAULT_BLOCK_THRESHOLD,
        max_consecutive_blocks: int = 3,
        max_total_blocks: int | float = 20,
    ) -> None:
        """Keep the configuration, warning when the block threshold is the default."""
        check_instance_option(
            feedback,
            option_type=FeedbackTemplate,
            parameter_name="feedback",
            hint="Wrap the text in FeedbackTemplate(text=...).",
        )
        check_fallback_option(when_limit_reached, parameter_name="when_limit_reached")
        self.feedback = feedback
        self.when_limit_reached = when_limit_reached
        self.block_threshold = resolve_threshold(
            parameter_name="block_threshold",
            threshold=block_threshold,
        )
        self.max_consecutive_blocks = read_count_option(
            max_consecutive_blocks,
            parameter_name="max_consecutive_blocks",
            minimum=1,
        )
        self.max_total_blocks = read_limit_option(
            max_total_blocks,
            parameter_name="max_total_blocks",
            minimum=1,
        )

    async def decide(self, step: PendingStep) -> StepDecision:
        """Sample until a step is acceptable or a block limit is reached.

        A step that follows subagent blocks which left the thread at or over
        its total goes to `when_limit_reached` without being sampled.
        """
        if self.is_total_reached_inside_subagents(step):
            return await self.when_limit_reached.take_over(step, samples=(), flagged=True)
        # Read once: this step's own blocks are counted in `blocked_attempts` as they happen.
        blocks_earlier_in_thread = step.blocks_in_thread
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
            # Checked after the block is kept, so the fallback records the one that hit the total.
            if blocks_earlier_in_thread + len(blocked_attempts) >= self.max_total_blocks:
                break
        return await self.when_limit_reached.take_over(
            step,
            samples=tuple(samples),
            flagged=True,
            blocked_attempts=tuple(blocked_attempts),
        )

    def is_total_reached_inside_subagents(self, step: PendingStep) -> bool:
        """Tell whether blocks inside subagents since this agent's last step reached the total."""
        # With no new subagent blocks, a thread already at the total is still sampled, and only
        # its first block goes to the fallback.
        return bool(step.new_subagent_blocks) and step.blocks_in_thread >= self.max_total_blocks


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
