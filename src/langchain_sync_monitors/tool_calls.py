"""How a monitor runs one of its agent's tool calls, and checks what the call writes.

Before the call, the monitor hands any subagent the call starts its
`Delegation`, through the state the tool reads. After it, the monitor checks
what the call writes, whatever the shape of the result or of a
`ParentCommand` the call raises: `task_authorship` relabels the messages and
drops the writes to the keys only the monitor writes, and `returned_records`
checks the records and stores the halts and blocks they hold.

Only the outermost monitor of an agent does this. A monitor inside it passes
the call on as it is: the outer monitor's guard would drop what the inner one
added to the result, and take it for a tool's write. A context variable
holds the request the outermost monitor is running, and each task and thread
the call starts copies it [@langchain2026]. A monitor further in knows the
call as the same tool call, or as a call with the same id in the same agent
state, should a middleware between the two have copied the tool call. A
subagent's own calls run in a state of their own, so the subagent's monitor
checks them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from langchain.agents.middleware.types import ToolCallRequest
from langgraph.errors import ParentCommand

from langchain_sync_monitors._langchain import (
    AsyncToolCallHandler,
    ToolCallHandler,
    ToolCallResult,
    ToolCallResults,
    cast_to_tool_call_result,
)
from langchain_sync_monitors.delegation import add_delegation
from langchain_sync_monitors.returned_records import (
    ToolCaller,
    check_parent_command_records,
    check_returned_records,
    read_tool_caller,
)
from langchain_sync_monitors.task_authorship import mark_tool_written_notes, relabel_parent_command

checked_tool_call: ContextVar[ToolCallRequest | None] = ContextVar(
    "monitor_checked_tool_call",
    default=None,
)
"""The tool request the outermost monitor of the running agent is checking, if any."""


def is_checked_further_out(request: ToolCallRequest) -> bool:
    """Tell whether a monitor further out in this agent is already checking this very call."""
    checked = checked_tool_call.get()
    if checked is None:
        return False
    return checked.tool_call is request.tool_call or (
        checked.state is request.state and checked.tool_call["id"] == request.tool_call["id"]
    )


@contextmanager
def check_tool_call(request: ToolCallRequest) -> Iterator[None]:
    """Mark the call as checked by this monitor while the rest of the stack runs it."""
    token = checked_tool_call.set(request)
    try:
        yield
    finally:
        checked_tool_call.reset(token)


def check_parent_command(bubble: ParentCommand, *, caller: ToolCaller) -> None:
    """Relabel and check, in place, what the command in a `ParentCommand` writes."""
    relabel_parent_command(bubble, tool_name=caller.tool_call["name"], state=caller.state)
    check_parent_command_records(bubble, caller=caller)


def check_tool_result(result: ToolCallResults, *, caller: ToolCaller) -> ToolCallResult:
    """Return a tool's result with its messages relabelled and its records checked."""
    written = mark_tool_written_notes(
        result,
        tool_name=caller.tool_call["name"],
        state=caller.state,
    )
    return cast_to_tool_call_result(check_returned_records(written, caller=caller))


def run_tool_call(
    request: ToolCallRequest,
    *,
    handler: ToolCallHandler,
    agent: str,
) -> ToolCallResult:
    """Run a tool call under `invoke()`, handing on the delegation and checking what it writes."""
    if is_checked_further_out(request):
        return handler(request)
    caller = read_tool_caller(request.state, agent=agent, tool_call=request.tool_call)
    with check_tool_call(request):
        try:
            result = handler(add_delegation(request, agent=agent))
        except ParentCommand as bubble:
            check_parent_command(bubble, caller=caller)
            raise
    return check_tool_result(result, caller=caller)


async def arun_tool_call(
    request: ToolCallRequest,
    *,
    handler: AsyncToolCallHandler,
    agent: str,
) -> ToolCallResult:
    """Run a tool call under `ainvoke()`, handing on the delegation and checking what it writes."""
    if is_checked_further_out(request):
        return await handler(request)
    caller = read_tool_caller(request.state, agent=agent, tool_call=request.tool_call)
    with check_tool_call(request):
        try:
            result = await handler(add_delegation(request, agent=agent))
        except ParentCommand as bubble:
            check_parent_command(bubble, caller=caller)
            raise
    return check_tool_result(result, caller=caller)
