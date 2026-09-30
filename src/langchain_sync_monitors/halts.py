"""When a monitored run halts, and how long the halt stands.

A halt ends the run with a message that has no tool calls. The model node
cannot route the agent to its end itself, so the monitor middleware's
`after_model` hook does, right after a step this monitor halted
[@langgraph2026]. That end is the agent's exit node, which is the first
`after_agent` hook when there is one [@langchain2026], and such a hook can send
the agent back to the model, as Deep Agents' `RubricMiddleware` does when it
grades the task unmet [@deepagents2026]. So the halt stands: while this
monitor's last step is a halt, every further step halts again, without a
sample, until a message the monitor recorded as a run's input follows the
halt.

A run's input is what `task_authorship` records under `TASK_MESSAGES_KEY` at
the start of a run, and nothing a hook writes during a run can join it. A
human message written after the halt, tagged or not, never lifts it: not a
grader's revision request, not a harness nudge, not a note from
`unconfirmed_input`. Only a later run with a new message from the user does.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors._langchain import AgentStateUpdate, read_monitor_log
from langchain_sync_monitors.contracts import Outcome, StepDecision, StepRecord, SubagentHalt
from langchain_sync_monitors.delegation import (
    build_subagent_halt_decision,
    find_new_subagent_halts,
)
from langchain_sync_monitors.feedback import build_monitor_message_id
from langchain_sync_monitors.records import find_monitor_records
from langchain_sync_monitors.task_authorship import (
    TASK_MESSAGES_KEY,
    read_message_ids,
    read_state_messages,
)

STANDING_HALT_MESSAGE = (
    "[Safety monitor] This run stays stopped: the safety monitor halted it, and no message "
    "the monitor could confirm as new input from the user has arrived since. The rejected "
    "steps are recorded in monitor_log for review."
)
"""The final message of a step the monitor halts again because its earlier halt stands."""


def build_end_run_update() -> AgentStateUpdate:
    """Return the update with which an `after_model` hook ends the agent's run [@langchain2026]."""
    return {"jump_to": "end"}


def has_just_halted(state: Mapping[str, object], *, monitor: str, agent: str) -> bool:
    """Tell whether the step just committed is this monitor's halt.

    The model node's own `jump_to` would not do: a routing edge reads a fresh
    copy of the state in which only its own node's writes survive, and
    `jump_to` is cleared everywhere else [@langgraph2026]. With any
    `after_model` hook in the agent, the model node has no routing edge of its
    own, so the hook that follows it has to write `jump_to` itself. The last
    message must be the halt, a final message with no tool calls, so an older
    halt record never ends a later turn.
    """
    own_records = find_monitor_records(read_monitor_log(state), monitor=monitor, agent=agent)
    messages = read_state_messages(state)
    last_message = messages[-1] if messages else None
    return (
        bool(own_records)
        and own_records[-1]["outcome"] == "halted"
        and isinstance(last_message, AIMessage)
        and not last_message.tool_calls
    )


def has_run_input_arrived_since_last_step(
    messages: Sequence[BaseMessage],
    *,
    task_message_ids: Collection[str],
) -> bool:
    """Tell whether a message recorded as a run's input follows the last AI message."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return False
        if message.id is not None and message.id in task_message_ids:
            return True
    return False


def is_halt_standing(
    previous_records: Sequence[StepRecord],
    *,
    messages: Sequence[BaseMessage],
    task_message_ids: Collection[str],
) -> bool:
    """Tell whether this monitor's last step halted and no run's input has arrived since.

    After a halt nothing runs until a hook sends the agent back to the model,
    so the halt message is the last AI message, and only a run's input after
    it lifts the halt.
    """
    return (
        bool(previous_records)
        and previous_records[-1]["outcome"] == "halted"
        and not has_run_input_arrived_since_last_step(messages, task_message_ids=task_message_ids)
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


def find_halt_decision(
    state: Mapping[str, object],
    *,
    previous_records: Sequence[StepRecord],
    agent: str,
    when_subagent_halts: SubagentHalt,
) -> StepDecision | None:
    """Return the halt a step gets without a sample, or None when the protocol decides it.

    The step halts again while this monitor's own halt stands, and halts when
    a subagent halted since this agent's last step and the monitor stops the
    whole run.
    """
    if is_halt_standing(
        previous_records,
        messages=read_state_messages(state),
        task_message_ids=read_message_ids(state, key=TASK_MESSAGES_KEY),
    ):
        return build_standing_halt_decision()
    if when_subagent_halts is SubagentHalt.STOP_SUBAGENT_ONLY:
        return None
    halts = find_new_subagent_halts(read_monitor_log(state), agent=agent)
    if not halts:
        return None
    return build_subagent_halt_decision(subagent_names=[record["agent"] for record in halts])
