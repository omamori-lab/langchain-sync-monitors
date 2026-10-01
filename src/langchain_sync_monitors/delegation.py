"""What a monitor carries across the subagents its agent starts.

A subagent starts every delegation with an empty `monitor_log`, and its
records flow back into its parent's log when it returns. Before each tool
call, the monitor adds a `Delegation` to the state the tool sees, so a
subagent that the call starts knows which call started it and how many blocks
the thread had recorded by then. From the log and the delegation, the monitor
counts the blocks that Auto Mode limits. `returned_records` checks the records
a call returns, and `subagent_returns` keeps the subagent halts and blocks the
agent has not yet answered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain.agents.middleware.types import ToolCallRequest

from langchain_sync_monitors._langchain import build_tool_request_with_delegation, read_delegation
from langchain_sync_monitors.contracts import Delegation, StepDecision
from langchain_sync_monitors.feedback import build_halt_decision
from langchain_sync_monitors.records import count_blocks, count_blocks_by_monitor, read_step_records


def read_delegation_id(state: object) -> str | None:
    """Return the id of the tool call that started this agent, or None if none did."""
    delegation = read_delegation(state)
    return None if delegation is None else delegation["tool_call_id"]


def add_delegation(request: ToolCallRequest, *, agent: str) -> ToolCallRequest:
    """Return the tool request with the `Delegation` a subagent started by the call receives.

    `agent` names the agent making the call. The blocks count from this
    agent's log, which holds its subagents' records too, on top of the blocks
    recorded before this agent started, which its own delegation holds. Both
    are read from the request's state, which stays the agent's own, so a
    delegation this agent inherited is replaced even when the call reuses
    its id, and a nested subagent still counts this agent's blocks. A call
    without an id, from which Deep Agents refuses to start a subagent
    [@deepagents2026], is returned as it is.
    """
    tool_call_id = request.tool_call["id"]
    if tool_call_id is None:
        return request
    own = read_delegation(request.state)
    blocks_before = count_blocks_by_monitor(
        read_step_records(request.state),
        earlier_blocks=own["blocks_before"] if own else {},
    )
    delegation = Delegation(
        tool_call_id=tool_call_id,
        delegating_agent=agent,
        blocks_before=blocks_before,
    )
    return build_tool_request_with_delegation(request, delegation=delegation)


def count_blocks_in_thread(state: Mapping[str, object], *, monitor: str) -> int:
    """Return the blocks one monitor recorded in the thread, as the agent with this state sees it.

    The agent's log holds its own records and its subagents'. Inside a
    subagent, the blocks recorded before it started come from its delegation.
    """
    delegation = read_delegation(state)
    earlier_blocks = delegation["blocks_before"].get(monitor, 0) if delegation else 0
    return earlier_blocks + count_blocks(read_step_records(state), monitor=monitor)


def build_subagent_halt_decision(*, subagent_names: Sequence[str]) -> StepDecision:
    """Return the decision that ends the run because a subagent was halted."""
    distinct_names = sorted(set(subagent_names))
    subject = "subagent" if len(distinct_names) == 1 else "subagents"
    return build_halt_decision(
        f"[Safety monitor] Stopped: the safety monitor halted the {subject} "
        f"{', '.join(distinct_names)}, so this agent stops too."
    )
