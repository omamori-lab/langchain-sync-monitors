"""What a monitor carries across the subagents its agent starts.

A subagent starts every delegation with an empty `monitor_log`, and its
records flow back into its parent's log when it returns. Before each tool
call, the monitor adds a `Delegation` to the state the tool sees, so a
subagent that the call starts knows which call started it, and the run its
trajectory joins under each monitor label, as `trajectory_runs` says, whose
records a protocol counts. `returned_records` checks the records a call
returns, and `subagent_returns` keeps the subagent halts the agent has not yet
answered.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage

from langchain_sync_monitors._langchain import build_tool_request_with_delegation, read_delegation
from langchain_sync_monitors.contracts import Delegation
from langchain_sync_monitors.feedback import build_halt_message
from langchain_sync_monitors.trajectory_runs import build_delegated_runs


def read_delegation_id(state: object) -> str | None:
    """Return the id of the tool call that started this agent, or None if none did."""
    delegation = read_delegation(state)
    return None if delegation is None else delegation["tool_call_id"]


def add_delegation(request: ToolCallRequest, *, agent: str) -> ToolCallRequest:
    """Return the tool request with the `Delegation` a subagent started by the call receives.

    `agent` names the agent making the call. The delegation is built from the
    request's state, which stays the agent's own, so a delegation this agent
    inherited is replaced even when the call reuses its id. A call without an
    id, from which Deep Agents refuses to start a subagent [@deepagents2026],
    is returned as it is. The runs are those of the labels whose trajectory
    holds the call.
    """
    tool_call_id = request.tool_call["id"]
    if tool_call_id is None:
        return request
    delegation = Delegation(
        tool_call_id=tool_call_id,
        delegating_agent=agent,
        runs=build_delegated_runs(request.state, call_id=tool_call_id),
    )
    return build_tool_request_with_delegation(request, delegation=delegation)


def build_subagent_halt_message(*, subagent_names: Sequence[str]) -> AIMessage:
    """Return the message that ends the run because a subagent was halted."""
    distinct_names = sorted(set(subagent_names))
    subject = "subagent" if len(distinct_names) == 1 else "subagents"
    return build_halt_message(
        f"[Safety monitor] Stopped: the safety monitor halted the {subject} "
        f"{', '.join(distinct_names)}, so this agent stops too."
    )
