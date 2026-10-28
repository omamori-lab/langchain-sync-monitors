"""The state a step reads once, before the protocol draws any sample.

`prepare_step` names the step, finds the halt it gets without a sample, if any,
and gathers the keywords of its pending step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_sync_monitors._langchain import AgentModelRequest
from langchain_sync_monitors.delegation import count_blocks_in_thread, read_delegation_id
from langchain_sync_monitors.halts import find_halt_decision, find_lost_write_halt
from langchain_sync_monitors.pending_steps import PendingStepOptions, PreparedStep
from langchain_sync_monitors.records import find_monitor_records, read_step_records
from langchain_sync_monitors.run_inputs import read_current_run_inputs
from langchain_sync_monitors.server_tools import warn_about_server_tools
from langchain_sync_monitors.spans import StepIdentity
from langchain_sync_monitors.state_keys import REWRITTEN_INPUTS_KEY, TASK_MESSAGES_KEY
from langchain_sync_monitors.subagent_returns import count_returned_blocks, read_subagent_returns
from langchain_sync_monitors.task_authorship import read_message_ids
from langchain_sync_monitors.trajectories import load_step_trajectory

if TYPE_CHECKING:
    # The middleware module imports this one.
    from langchain_sync_monitors.middleware import MonitorMiddleware


def prepare_step(request: AgentModelRequest, *, middleware: MonitorMiddleware) -> PreparedStep:
    """Read the log once for a step: its identity, a halt without a sample, its options."""
    warn_about_server_tools(request, middleware=middleware, middleware_name=middleware.name)
    delegation_id = read_delegation_id(request.state)
    previous_records = find_monitor_records(
        read_step_records(request.state),
        monitor=middleware.label,
        agent=middleware.agent_name,
        delegation_id=delegation_id,
    )
    returns = read_subagent_returns(request.state)
    trajectory = load_step_trajectory(
        request,
        label=middleware.label,
        task_author=middleware.task_author,
    )
    identity = StepIdentity(
        monitor=middleware.label,
        agent=middleware.agent_name,
        step_number=len(previous_records) + 1,
        protocol=type(middleware.protocol).__name__,
        delegation_id=delegation_id,
    )
    halt = find_lost_write_halt(
        trajectory.lost_writes,
        monitor=middleware.label,
        middleware_name=middleware.name,
    ) or find_halt_decision(
        request.state,
        previous_records=previous_records,
        returns=returns,
        middleware_name=middleware.name,
        when_subagent_halts=middleware.when_subagent_halts,
    )
    options = PendingStepOptions(
        request=request,
        monitor=middleware.monitor,
        task_author=middleware.task_author,
        task_message_ids=read_message_ids(request.state, key=TASK_MESSAGES_KEY),
        run_inputs=read_current_run_inputs(request.state),
        rewritten_input_ids=read_message_ids(request.state, key=REWRITTEN_INPUTS_KEY),
        previous_records=previous_records,
        blocks_in_thread=count_blocks_in_thread(request.state, monitor=middleware.label),
        new_subagent_blocks=count_returned_blocks(returns, monitor=middleware.label),
    )
    return PreparedStep(identity=identity, halt=halt, options=options, trajectory=trajectory)
