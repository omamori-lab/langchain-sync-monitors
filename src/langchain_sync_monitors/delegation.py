"""What a monitor carries across the subagents its agent starts.

A subagent starts every delegation with an empty `monitor_log`, and its
records flow back into its parent's log when it returns. Before each tool
call, the monitor adds a `Delegation` to the state the tool sees, so a
subagent that the call starts knows which call started it and how many blocks
the thread had recorded by then. From the log and the delegation, the monitor
counts the blocks that Auto Mode limits, and finds the subagent halts its
agent has not yet answered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain.agents.middleware.types import ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage

from langchain_sync_monitors._langchain import (
    build_tool_request_with_delegation,
    read_delegation,
    read_monitor_log,
)
from langchain_sync_monitors.contracts import Delegation, Outcome, StepDecision, StepRecord
from langchain_sync_monitors.feedback import build_monitor_message_id
from langchain_sync_monitors.records import (
    count_blocks,
    count_blocks_by_monitor,
    find_new_subagent_records,
)


def read_delegation_id(state: object) -> str | None:
    """Return the id of the tool call that started this agent, or None if none did."""
    delegation = read_delegation(state)
    return None if delegation is None else delegation["tool_call_id"]


def add_delegation(request: ToolCallRequest, *, agent: str) -> ToolCallRequest:
    """Return the tool request with the `Delegation` a subagent started by the call receives.

    `agent` names the agent making the call. The blocks count from this
    agent's log, which holds its subagents' records too, on top of the blocks
    recorded before this agent started. A request that already carries this
    agent's delegation for this very call, added by a monitor further out, is
    returned as it is, so stacked monitors do not count the blocks twice. A
    delegation this agent inherited is replaced even when the call reuses its
    id, so a nested subagent still counts this agent's blocks. A call without
    an id, from which Deep Agents refuses to start a subagent
    [@deepagents2026], is returned as it is.
    """
    tool_call_id = request.tool_call["id"]
    if tool_call_id is None:
        return request
    earlier = read_delegation(request.state)
    if is_delegation_of_call(earlier, tool_call_id=tool_call_id, agent=agent):
        return request
    blocks_before = count_blocks_by_monitor(
        read_monitor_log(request.state),
        earlier_blocks=earlier["blocks_before"] if earlier else {},
    )
    delegation = Delegation(
        tool_call_id=tool_call_id,
        delegating_agent=agent,
        blocks_before=blocks_before,
    )
    return build_tool_request_with_delegation(request, delegation=delegation)


def is_delegation_of_call(
    delegation: Delegation | None,
    *,
    tool_call_id: str,
    agent: str,
) -> bool:
    """Tell whether the delegation is the one this agent made for this tool call."""
    return (
        delegation is not None
        and delegation["tool_call_id"] == tool_call_id
        and delegation["delegating_agent"] == agent
    )


def count_blocks_in_thread(state: Mapping[str, object], *, monitor: str) -> int:
    """Return the blocks one monitor recorded in the thread, as the agent with this state sees it.

    The agent's log holds its own records and its subagents'. Inside a
    subagent, the blocks recorded before it started come from its delegation.
    """
    delegation = read_delegation(state)
    earlier_blocks = delegation["blocks_before"].get(monitor, 0) if delegation else 0
    return earlier_blocks + count_blocks(read_monitor_log(state), monitor=monitor)


def count_new_subagent_blocks(records: Sequence[StepRecord], *, agent: str, monitor: str) -> int:
    """Return the blocks one monitor recorded inside subagents since this agent's last step."""
    return count_blocks(find_new_subagent_records(records, agent=agent), monitor=monitor)


def find_new_subagent_halts(
    records: Sequence[StepRecord],
    *,
    agent: str,
) -> list[StepRecord]:
    """Return the halts of other agents logged since this agent's last step.

    Only halts this agent has not yet answered count, so a halted run does not
    stay halted on the next turn of a checkpointed thread.
    """
    return [
        record
        for record in find_new_subagent_records(records, agent=agent)
        if record["outcome"] == "halted"
    ]


def build_subagent_halt_decision(*, subagent_names: Sequence[str]) -> StepDecision:
    """Return the decision that ends the run because a subagent was halted."""
    distinct_names = sorted(set(subagent_names))
    subject = "subagent" if len(distinct_names) == 1 else "subagents"
    message = AIMessage(
        content=(
            f"[Safety monitor] Stopped: the safety monitor halted the {subject} "
            f"{', '.join(distinct_names)}, so this agent stops too."
        ),
        id=build_monitor_message_id(),
    )
    return StepDecision(
        outcome=Outcome.HALTED,
        response=ModelResponse(result=[message]),
        samples=(),
        executed_sample=None,
        flagged=True,
    )
