"""When a monitored run halts, and how long the halt stands.

A halt ends the run with a message that has no tool calls. The model node
cannot route the agent to its end itself, so the monitor middleware's
`after_model` hook does, right after a step this monitor halted
[@langgraph2026]. That end is the agent's exit node, which is the first
`after_agent` hook when there is one [@langchain2026], and such a hook can send
the agent back to the model, as Deep Agents' `RubricMiddleware` does when it
grades the task unmet [@deepagents2026]. So the halt stands: while this
monitor's last step is a halt, every further step halts again, without a
sample, until the thread has recorded more run inputs than it had at the
halt.

A halt is this monitor's own when its record names this agent and this
agent's delegation, so the steps of a subagent that shares this agent's name,
a fork under the main agent's monitor or a compiled subagent whose monitor
keeps the default `agent_name`, never count as this agent's. A subagent's
halt reaches this agent through `subagent_returns`, where `returned_records`
stores it when the tool call that started the subagent returns, and this
agent's next step answers it as `SubagentHalt` says.

A run's input is what `task_authorship` records under `TASK_MESSAGES_KEY` at
the start of a run. Each halt stores how many inputs the thread held then,
under `INPUTS_AT_HALT_KEY`, and the halt stands while that count has not
grown. The rule counts rather than reading where messages sit, so a hook that
rewrites the history, trims it or removes the halt message cannot lift the
halt, and one that writes messages after the run's input cannot keep it
standing. No human message written between a run's start and its end lifts
it: not a grader's revision request, not a harness nudge, not a note from
`unconfirmed_input`. A later run with a new message from the user lifts it,
and so do the two open paths `task_authorship` names: an untagged human
message that a middleware listed before the monitor writes from its
`before_agent` hook, or from its `after_agent` hook without sending the run
back to the model, counts as a run's input. The same hooks of a middleware
listed after the monitor run on the other side of the monitor's own, so what
they write stays a note.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from langchain_sync_monitors._langchain import AgentStateUpdate
from langchain_sync_monitors.contracts import Outcome, StepDecision, StepRecord, SubagentHalt
from langchain_sync_monitors.delegation import build_subagent_halt_decision, read_delegation_id
from langchain_sync_monitors.feedback import build_monitor_message_id
from langchain_sync_monitors.records import find_monitor_records, read_step_records
from langchain_sync_monitors.state_keys import INPUTS_AT_HALT_KEY, TASK_MESSAGES_KEY
from langchain_sync_monitors.subagent_returns import SubagentReturn, find_halted_subagents
from langchain_sync_monitors.task_authorship import read_message_ids, read_state_messages

STANDING_HALT_MESSAGE = (
    "[Safety monitor] This run is stopped: the safety monitor halted an earlier step, and "
    "that halt stands until a later run brings new input the monitor can confirm. The "
    "rejected steps are recorded in monitor_log for review."
)
"""The final message of a step halted again, without a sample, while an earlier halt stands.

The earlier halt may belong to this run or to an earlier one, so the message
does not say which.
"""


class InputsAtHalt(TypedDict):
    """How many run inputs the thread had recorded when one monitor last halted.

    `monitor` is the middleware's name, such as `monitor[main]`, which is
    unique within an agent, so stacked monitors count apart.
    """

    monitor: str
    run_inputs: int


def merge_inputs_at_halt(  # lanorme: ignore[KWARG-001]
    recorded: list[InputsAtHalt],
    new: list[InputsAtHalt],
) -> list[InputsAtHalt]:
    """Keep one entry per monitor, the latest, so stacked monitors can write in one node.

    LangGraph calls a reducer with both values by position [@langgraph2026].
    """
    latest = {entry["monitor"]: entry for entry in [*recorded, *new]}
    return list(latest.values())


def count_run_inputs(state: Mapping[str, object]) -> int:
    """Return how many human messages the thread has recorded as a run's input."""
    return len(read_message_ids(state, key=TASK_MESSAGES_KEY))


def read_run_inputs_at_halt(state: Mapping[str, object], *, monitor: str) -> int | None:
    """Return how many run inputs the thread had at this monitor's latest halt, or None."""
    entries = state.get(INPUTS_AT_HALT_KEY)
    if not isinstance(entries, list):
        return None
    counts = [
        entry.get("run_inputs")
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("monitor") == monitor
    ]
    latest = counts[-1] if counts else None
    # A count that is missing or not an integer reads as None, which keeps a halt standing.
    return latest if isinstance(latest, int) else None


def build_halt_inputs_update(
    record: StepRecord,
    *,
    state: Mapping[str, object],
    monitor: str,
) -> AgentStateUpdate:
    """Return the update that stores the run inputs at a halted step, or none for another step."""
    if record["outcome"] != "halted":
        return {}
    entry = InputsAtHalt(monitor=monitor, run_inputs=count_run_inputs(state))
    return {INPUTS_AT_HALT_KEY: [entry]}


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
    own_records = find_monitor_records(
        read_step_records(state),
        monitor=monitor,
        agent=agent,
        delegation_id=read_delegation_id(state),
    )
    messages = read_state_messages(state)
    last_message = messages[-1] if messages else None
    return (
        bool(own_records)
        and own_records[-1]["outcome"] == "halted"
        and isinstance(last_message, AIMessage)
        and not last_message.tool_calls
    )


def is_halt_standing(
    previous_records: Sequence[StepRecord],
    *,
    run_inputs: int,
    run_inputs_at_halt: int | None,
) -> bool:
    """Tell whether this monitor's last step halted and no run's input has been recorded since.

    A halt whose count is missing stands, so the rule fails closed.
    """
    if not previous_records or previous_records[-1]["outcome"] != "halted":
        return False
    # Only a checkpoint written before the count existed holds a halt without one,
    # and 0.1.0 is the first release, so no released thread does: fail closed.
    return run_inputs_at_halt is None or run_inputs <= run_inputs_at_halt


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
    returns: Sequence[SubagentReturn],
    monitor: str,
    when_subagent_halts: SubagentHalt,
) -> StepDecision | None:
    """Return the halt a step gets without a sample, or None when the protocol decides it.

    `monitor` is the middleware's name, and `returns` holds what this
    agent's tool calls returned since its last step. The step halts again
    while this monitor's own halt stands, and halts when one of those calls
    returned a subagent's halt and the monitor stops the whole run.
    """
    if is_halt_standing(
        previous_records,
        run_inputs=count_run_inputs(state),
        run_inputs_at_halt=read_run_inputs_at_halt(state, monitor=monitor),
    ):
        return build_standing_halt_decision()
    if when_subagent_halts is SubagentHalt.STOP_SUBAGENT_ONLY:
        return None
    halted_subagents = find_halted_subagents(returns)
    if not halted_subagents:
        return None
    return build_subagent_halt_decision(subagent_names=halted_subagents)
