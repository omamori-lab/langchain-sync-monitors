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
holds the request the outermost monitor handed on, and each task and thread
the call starts copies it [@langchain2026]. A monitor further in knows the
call by the tool runtime in that request, which a middleware between the two
keeps even when it copies the tool call or the state; failing that, as the
same tool call, or as a call with the same id in the same agent state. A
subagent's own calls run with a runtime and a state of their own, so the
subagent's monitor checks them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from langchain.agents.middleware.types import ToolCallRequest
from langgraph.errors import ParentCommand

from langchain_sync_monitors._langchain import (
    AsyncToolCallHandler,
    ToolCallHandler,
    ToolCallResult,
    ToolCallResults,
    cast_to_tool_call_result,
)
from langchain_sync_monitors.context_values import set_context_value
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
    if checked.runtime is not None and checked.runtime is request.runtime:
        return True
    return checked.tool_call is request.tool_call or (
        checked.state is request.state and checked.tool_call["id"] == request.tool_call["id"]
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckedToolCall:
    """A tool call as this monitor hands it on: the request to run, and who made the call.

    `caller` is None when a monitor further out in the agent checks the call,
    and the result then passes through as it is.
    """

    request: ToolCallRequest
    caller: ToolCaller | None = None

    def check_result(self, result: ToolCallResults) -> ToolCallResult:
        """Return the call's result with its messages relabelled and its records checked."""
        if self.caller is None:
            return cast_to_tool_call_result(result)
        written = mark_tool_written_notes(
            result,
            tool_name=self.caller.tool_call["name"],
            state=self.caller.state,
        )
        return cast_to_tool_call_result(check_returned_records(written, caller=self.caller))


@contextmanager
def check_tool_call(request: ToolCallRequest, *, agent: str) -> Iterator[CheckedToolCall]:
    """Hand on the call with its delegation, marked as checked while the rest of the stack runs it.

    A call a monitor further out already checks is handed on as it is. A
    `ParentCommand` the call raises is relabelled and checked, in place, on
    its way out.
    """
    if is_checked_further_out(request):
        yield CheckedToolCall(request=request)
        return
    caller = read_tool_caller(request.state, agent=agent, tool_call=request.tool_call)
    delegated = add_delegation(request, agent=agent)
    with set_context_value(checked_tool_call, value=delegated):
        try:
            yield CheckedToolCall(request=delegated, caller=caller)
        except ParentCommand as bubble:
            relabel_parent_command(bubble, tool_name=caller.tool_call["name"], state=caller.state)
            check_parent_command_records(bubble, caller=caller)
            raise


def run_tool_call(
    request: ToolCallRequest,
    *,
    handler: ToolCallHandler,
    agent: str,
) -> ToolCallResult:
    """Run a tool call under `invoke()`, handing on the delegation and checking what it writes."""
    with check_tool_call(request, agent=agent) as checked_call:
        result = handler(checked_call.request)
    return checked_call.check_result(result)


async def arun_tool_call(
    request: ToolCallRequest,
    *,
    handler: AsyncToolCallHandler,
    agent: str,
) -> ToolCallResult:
    """Run a tool call under `ainvoke()`, handing on the delegation and checking what it writes."""
    with check_tool_call(request, agent=agent) as checked_call:
        result = await handler(checked_call.request)
    return checked_call.check_result(result)
