"""The graph state a monitor adds to the agent it sits in.

`MonitorMiddleware` declares this schema, and LangChain merges it into the
agent's state [@langchain2026]. It holds the log of every monitored step, the
delegation a subagent was started with, the record of which human messages
were a run's input and their text, and how many of those the thread had at
each monitor's latest halt.
"""

import operator
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentState,
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
)

from langchain_sync_monitors.contracts import Delegation, StepRecord
from langchain_sync_monitors.halts import InputsAtHalt, merge_inputs_at_halt
from langchain_sync_monitors.run_inputs import RunInput, merge_run_inputs
from langchain_sync_monitors.task_authorship import keep_latest_flag, merge_message_ids


class MonitorState(AgentState):
    """The agent state with the log of every monitored step.

    The reducer comes last in the annotation because LangGraph reads it only
    from the last metadata position; anywhere else the log silently keeps only
    its last write [@langgraph2026]. `OmitFromInput` keeps the log out of a
    subagent's input, so a subagent never copies its parent's records back.

    `monitor_delegation` holds the `Delegation` a subagent was started with.
    The monitor adds it to the state each tool call sees, and Deep Agents'
    `task` tool passes that state on to the subagent it starts.
    `OmitFromOutput` keeps it out of the subagent's result, so it never flows
    back into the parent, as Deep Agents does for its own forked-context flag
    [@deepagents2026].

    `monitor_task_messages` holds the ids of the untagged human messages that
    arrived as a run's input, the only ones a monitor reads as the task
    author's, and `monitor_seen_human_messages` the ids of every untagged
    human message the monitor has seen, so a later run can tell its input
    from a message written during an earlier run. `monitor_run_inputs` holds
    the text of each run's input, and of input a run could not confirm, so
    the monitor reads it even once the model request no longer holds it;
    `run_inputs` has the rule. `monitor_run_open`
    is true from the start of a run until it reaches the monitor's
    `after_agent` hook, so a run can tell that the last one stopped early.

    `monitor_inputs_at_halt` holds, for each monitor that halted, how many
    run inputs the thread had at its latest halt, so the halt stands until
    that count grows. It is a list of `InputsAtHalt` entries rather than a
    mapping, since state records hold plain values, lists and TypedDicts,
    which pass through every checkpointer as `monitor_log` does, and its
    reducer keeps the latest entry for each monitor.

    These five are private, so a subagent starts without its parent's and
    returns none of its own, and a run's result leaves them out, though
    `stream_mode="values"`, `stream_mode="updates"` and `get_state` show
    them. Their reducers let stacked monitors write them in one node.
    """

    monitor_log: NotRequired[Annotated[list[StepRecord], OmitFromInput, operator.add]]
    monitor_delegation: NotRequired[Annotated[Delegation, OmitFromOutput]]
    monitor_task_messages: NotRequired[Annotated[list[str], PrivateStateAttr, merge_message_ids]]
    monitor_seen_human_messages: NotRequired[
        Annotated[list[str], PrivateStateAttr, merge_message_ids]
    ]
    monitor_run_inputs: NotRequired[Annotated[list[RunInput], PrivateStateAttr, merge_run_inputs]]
    monitor_run_open: NotRequired[Annotated[bool, PrivateStateAttr, keep_latest_flag]]
    monitor_inputs_at_halt: NotRequired[
        Annotated[list[InputsAtHalt], PrivateStateAttr, merge_inputs_at_halt]
    ]
