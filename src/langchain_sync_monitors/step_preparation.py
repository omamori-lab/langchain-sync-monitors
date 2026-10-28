"""The state a step reads once, before the protocol draws any reply.

`prepare_step` names the step, finds the halt it gets without a sample, if any,
and builds the step the protocol drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from langchain_sync_monitors._langchain import AgentModelRequest
from langchain_sync_monitors.delegation import read_delegation_id
from langchain_sync_monitors.halts import find_halt_message, find_lost_write_halt
from langchain_sync_monitors.model_calls import resolve_chat_model
from langchain_sync_monitors.records import find_monitor_records, read_step_records
from langchain_sync_monitors.server_tools import warn_about_server_tools
from langchain_sync_monitors.spans import StepIdentity
from langchain_sync_monitors.steps import Step
from langchain_sync_monitors.subagent_returns import read_subagent_returns
from langchain_sync_monitors.trajectories import check_trajectory_kept, load_step_trajectory

if TYPE_CHECKING:
    # The middleware module imports this one.
    from langchain_sync_monitors.middleware import MonitorMiddleware


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedStep:
    """A step as the middleware reads it before drawing any reply.

    `identity` names the step, `halt` is the message of the halt it gets
    without a sample, if any, and `step` is the step the protocol drives, the
    monitor's trajectory synced with the step's request.
    """

    identity: StepIdentity
    halt: AIMessage | None
    step: Step


def prepare_step(request: AgentModelRequest, *, middleware: MonitorMiddleware) -> PreparedStep:
    """Read the state once for a step: its identity, a halt without a sample, the step to drive.

    A thread whose decided steps outlive the monitor's trajectory raises
    `MonitorError`, as `trajectories` says.
    """
    warn_about_server_tools(request, middleware=middleware, middleware_name=middleware.name)
    check_trajectory_kept(request.state, label=middleware.label, agent=middleware.agent_name)
    delegation_id = read_delegation_id(request.state)
    previous_records = find_monitor_records(
        read_step_records(request.state),
        monitor=middleware.label,
        agent=middleware.agent_name,
        delegation_id=delegation_id,
    )
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
    ) or find_halt_message(
        request.state,
        previous_records=previous_records,
        returns=read_subagent_returns(request.state),
        middleware_name=middleware.name,
        when_subagent_halts=middleware.when_subagent_halts,
    )
    trusted_model = middleware.trusted_model
    step = Step(
        request=request,
        trajectory=trajectory,
        # The middleware resolved the trusted model when it was built, so this returns it.
        trusted_model=None if trusted_model is None else resolve_chat_model(trusted_model),
        feedback=middleware.feedback,
    )
    return PreparedStep(identity=identity, halt=halt, step=step)
