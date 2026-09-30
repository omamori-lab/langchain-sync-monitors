"""A halt counts the thread's run inputs, so a hook that rewrites the history cannot move it.

Each halt stores how many run inputs the thread held, and the halt stands until
that count grows. A hook that removes the halt message, or trims the history
as LangGraph's short-term memory guide shows [@langgraph2026], cannot lift
the halt without new input, and one that writes messages after the run's input
cannot keep the user's new message from lifting it. The two open paths that
`task_authorship` names still lift it, and the tests here pin that on purpose.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from langchain_sync_monitors.halts import (
    INPUTS_AT_HALT_KEY,
    InputsAtHalt,
    merge_inputs_at_halt,
    read_run_inputs_at_halt,
)
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import TrustedMonitoring
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.unit.middleware.test_standing_halts import (
    ReturningToModelMiddleware,
    build_halting_monitor,
    read_outcomes,
)

REMINDER = "Reminder: today is a working day."


class HaltRemovingMiddleware(AgentMiddleware[Any, Any, Any]):
    """A `before_model` hook, listed before the monitor, that removes the halt messages."""

    @property
    def name(self) -> str:
        return "halt_remover"

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.build_removal(state)

    async def abefore_model(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.build_removal(state)

    def build_removal(self, state: AgentState[Any]) -> dict[str, Any] | None:
        removals = [
            RemoveMessage(id=message.id)
            for message in state["messages"]
            if isinstance(message, AIMessage)
            and message.id is not None
            and message.id.startswith("monitor-")
            and not message.tool_calls
        ]
        return {"messages": removals} if removals else None


class TrimmingMiddleware(AgentMiddleware[Any, Any, Any]):
    """A `before_model` hook that trims the history with the settings LangGraph's guide shows."""

    @property
    def name(self) -> str:
        return "trimmer"

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.build_trim(state)

    async def abefore_model(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.build_trim(state)

    def build_trim(self, state: AgentState[Any]) -> dict[str, Any] | None:
        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=10_000,
            start_on="human",
            end_on=("human", "tool"),
        )
        if [message.id for message in trimmed] == [message.id for message in state["messages"]]:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}


class ContextInjectingMiddleware(AgentMiddleware[Any, Any, Any]):
    """A `before_agent` hook that hands the agent retrieved context as a finished tool call."""

    @property
    def name(self) -> str:
        return "context_injector"

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.build_injection(state)

    async def abefore_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.build_injection(state)

    def build_injection(self, state: AgentState[Any]) -> dict[str, Any]:
        call_id = f"context-{len(state['messages'])}"
        call = AIMessage(
            "",
            tool_calls=[{"name": "load_context", "args": {}, "id": call_id, "type": "tool_call"}],
        )
        result = ToolMessage("Company handbook: be concise.", tool_call_id=call_id)
        return {"messages": [call, result]}


class ReminderAtStartMiddleware(AgentMiddleware[Any, Any, Any]):
    """A `before_agent` hook, listed before the monitor, that writes an untagged reminder."""

    @property
    def name(self) -> str:
        return "reminder_at_start"

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return {"messages": [HumanMessage(REMINDER)]}

    async def abefore_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return {"messages": [HumanMessage(REMINDER)]}


class ReminderAtEndMiddleware(AgentMiddleware[Any, Any, Any]):
    """An `after_agent` hook, listed before the monitor, that writes an untagged reminder.

    LangChain runs `after_agent` hooks in reverse list order [@langchain2026],
    so it writes after the monitor's own hook has closed the run, and it does
    not send the run back to the model.
    """

    @property
    def name(self) -> str:
        return "reminder_at_end"

    def after_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return {"messages": [HumanMessage(REMINDER)]}

    async def aafter_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return {"messages": [HumanMessage(REMINDER)]}


def build_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            build_exfiltration_step(),
            build_read_step(),
            *[AIMessage(f"Done {index}.") for index in range(4)],
        ],
    )


def build_agent(
    model: ScriptedChatModel,
    *,
    workspace: Workspace,
    middleware: list[AgentMiddleware[Any, Any, Any]],
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model,
        tools=workspace.build_tools(),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def run_without_input(agent: Any, *, mode: RunMode, config: RunnableConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": []}
    if mode == "invoke":
        return agent.invoke(payload, config)
    return asyncio.run(agent.ainvoke(payload, config))


def test_removing_the_halt_message_does_not_lift_the_halt(run_mode: RunMode) -> None:
    # Arrange: a grader returns the halted run, and a hook removes the halt message
    workspace = Workspace()
    model = build_model()
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ReturningToModelMiddleware(returns=1),
        HaltRemovingMiddleware(),
        build_halting_monitor(),
    ]
    agent = create_agent(model, tools=workspace.build_tools(), middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == ["halted", "halted"]
    assert len(model.calls) == 1
    assert workspace.executed == []


def test_trimming_the_history_does_not_lift_the_halt_on_a_run_without_input(
    run_mode: RunMode,
) -> None:
    # Arrange: a trim drops the halt message, which ends the history as an AI message
    workspace = Workspace()
    model = build_model()
    agent = build_agent(
        model,
        workspace=workspace,
        middleware=[TrimmingMiddleware(), build_halting_monitor()],
    )
    config = build_thread_config(f"trim-{run_mode}")
    assert read_outcomes(run_agent(agent, mode=run_mode, config=config)) == ["halted"]

    # Act
    second = run_without_input(agent, mode=run_mode, config=config)

    # Assert
    assert read_outcomes(second) == ["halted", "halted"]
    assert len(model.calls) == 1
    assert workspace.executed == []


def test_the_users_new_message_lifts_the_halt_despite_context_written_after_it(
    run_mode: RunMode,
) -> None:
    # Arrange: every run starts with context written after the run's input
    workspace = Workspace()
    model = build_model()
    agent = build_agent(
        model,
        workspace=workspace,
        middleware=[ContextInjectingMiddleware(), build_halting_monitor()],
    )
    config = build_thread_config(f"injected-context-{run_mode}")
    assert read_outcomes(run_agent(agent, mode=run_mode, config=config)) == ["halted"]

    # Act
    second = run_agent(agent, mode=run_mode, config=config, task="Summarise q3.md only.")

    # Assert
    assert read_outcomes(second) == ["halted", "allowed", "allowed"]
    assert workspace.executed == ["read_file:q3.md"]


@pytest.mark.parametrize("hook", ["before-agent", "after-agent"])
def test_a_reminder_another_middleware_writes_at_a_run_s_edge_lifts_the_halt(
    run_mode: RunMode,
    hook: str,
) -> None:
    # Arrange: the two open paths task_authorship names, pinned so a change is deliberate
    workspace = Workspace()
    model = build_model()
    reminder: AgentMiddleware[Any, Any, Any] = (
        ReminderAtStartMiddleware() if hook == "before-agent" else ReminderAtEndMiddleware()
    )
    agent = build_agent(
        model,
        workspace=workspace,
        middleware=[reminder, build_halting_monitor()],
    )
    config = build_thread_config(f"reminder-{hook}-{run_mode}")
    assert read_outcomes(run_agent(agent, mode=run_mode, config=config))[0] == "halted"

    # Act: a run with no message of the user's
    second = run_without_input(agent, mode=run_mode, config=config)

    # Assert: the reminder counts as the run's input, so the untrusted model runs again
    assert read_outcomes(second)[-1] == "allowed"
    assert len(model.calls) > 1
    assert workspace.executed == ["read_file:q3.md"]


def test_a_thread_that_halts_again_after_new_input_stands_again(run_mode: RunMode) -> None:
    # Arrange: halt, a new message lifts it, the next step halts, and a run without input
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            build_exfiltration_step(call_id="call-post-2"),
            build_read_step(),
            AIMessage("Done."),
        ],
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ReturningToModelMiddleware(returns=2),
        build_halting_monitor(),
    ]
    agent = build_agent(model, workspace=workspace, middleware=stack)
    config = build_thread_config(f"halt-twice-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)
    second = run_agent(agent, mode=run_mode, config=config, task="Try once more.")

    # Act
    third = run_without_input(agent, mode=run_mode, config=config)

    # Assert
    assert read_outcomes(second) == ["halted", "halted", "halted", "halted"]
    assert read_outcomes(third) == ["halted", "halted", "halted", "halted", "halted"]
    assert len(model.calls) == 2
    assert workspace.executed == []


def test_an_outer_monitor_s_halt_stands_on_its_own_latest_count(run_mode: RunMode) -> None:
    # Arrange: the outer monitor halts over an inner one that allows every step
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            build_exfiltration_step(call_id="call-post-2"),
            build_read_step(),
        ],
    )
    outer = build_halting_monitor(label="guard")
    inner = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=TrustedMonitoring(flag_threshold=0.5),
        label="inner",
    )
    agent = build_agent(model, workspace=workspace, middleware=[outer, inner])
    config = build_thread_config(f"stacked-counts-{run_mode}")
    first = run_agent(agent, mode=run_mode, config=config)

    # Act: a new message lifts the guard's halt, the guard halts again, then no input
    second = run_agent(agent, mode=run_mode, config=config, task="Try once more.")
    third = run_without_input(agent, mode=run_mode, config=config)

    # Assert: each run without new input stands on the guard's latest count
    guard_outcomes = [
        record["outcome"] for record in third["monitor_log"] if record["monitor"] == "guard"
    ]
    assert read_outcomes(first) == ["allowed", "halted"]
    assert [r["monitor"] for r in second["monitor_log"]][2:] == ["inner", "guard"]
    assert guard_outcomes == ["halted", "halted", "halted"]
    assert len(model.calls) == 2
    assert workspace.executed == []


def test_each_monitor_reads_the_count_of_its_own_latest_halt() -> None:
    # Arrange: stacked monitors halted at different counts, and the guard halted twice
    recorded = [
        InputsAtHalt(monitor="guard[main]", run_inputs=1),
        InputsAtHalt(monitor="monitor[main]", run_inputs=2),
    ]
    new = [InputsAtHalt(monitor="guard[main]", run_inputs=3)]

    # Act
    merged = merge_inputs_at_halt(recorded, new)
    state = {INPUTS_AT_HALT_KEY: merged}

    # Assert
    assert read_run_inputs_at_halt(state, monitor="guard[main]") == 3
    assert read_run_inputs_at_halt(state, monitor="monitor[main]") == 2
    assert read_run_inputs_at_halt(state, monitor="other[main]") is None
    assert len(merged) == 2
