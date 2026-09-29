"""When a halted run stays halted: until the task author writes again.

A halt ends the run with a message that has no tool calls, and the monitor
middleware routes the agent to its end. That end is the agent's exit node,
which is the first `after_agent` hook when there is one [@langchain2026], and
such a hook can send the agent back to the model, as Deep Agents'
`RubricMiddleware` does when it grades the task unmet [@deepagents2026]. So the
middleware halts each further step again, without a sample, while this
monitor's last step is a halt and the task author has not written since.

The task author's messages are the ones the monitor reads as the user's or the
delegating agent's: human messages without an `lc_source` tag.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_sync_monitors.contracts import Outcome, StepDecision, StepRecord
from langchain_sync_monitors.feedback import build_monitor_message_id
from langchain_sync_monitors.transcript import read_message_source

STANDING_HALT_MESSAGE = (
    "[Safety monitor] This run stays stopped: the safety monitor halted it, and no new "
    "message from the user has arrived since. The rejected steps are recorded in "
    "monitor_log for review."
)
"""The final message of a step the monitor halts again because its earlier halt stands."""


def is_task_author_message(message: BaseMessage) -> bool:
    """Tell whether the monitor reads a message as the task author's: an untagged human one."""
    return isinstance(message, HumanMessage) and read_message_source(message) is None


def has_task_author_written_since_last_step(messages: Sequence[BaseMessage]) -> bool:
    """Tell whether a message from the task author follows the conversation's last AI message."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return False
        if is_task_author_message(message):
            return True
    return False


def is_halt_standing(
    previous_records: Sequence[StepRecord],
    *,
    messages: Sequence[BaseMessage],
) -> bool:
    """Tell whether this monitor's last step halted and the task author has not written since.

    After a halt nothing runs until a hook sends the agent back to the model,
    so the halt message is the last AI message, and only a message the task
    author wrote after it lifts the halt.
    """
    return (
        bool(previous_records)
        and previous_records[-1]["outcome"] == "halted"
        and not has_task_author_written_since_last_step(messages)
    )


def build_standing_halt_decision() -> StepDecision:
    """Return the decision that halts the run again, without a sample, while an earlier halt stands.

    It is flagged, as every halt is: a hook tried to send a halted run back
    to the model, which a person should see.
    """
    message = AIMessage(content=STANDING_HALT_MESSAGE, id=build_monitor_message_id())
    return StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[message]),
        samples=(),
        executed_sample=None,
        flagged=True,
    )
