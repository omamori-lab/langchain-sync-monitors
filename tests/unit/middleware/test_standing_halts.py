"""A halt stands for the rest of the run, even when a hook sends the agent back to the model.

An `after_model` hook's `jump_to: "end"` reaches the agent's exit node, which
is the first `after_agent` hook when there is one [@langchain2026], and such a
hook can jump back to the model, as Deep Agents' `RubricMiddleware` does. The
monitor then halts every further step without drawing a sample, until a later
run brings a message the monitor records as that run's input. A human message
written during a run, tagged or not, never lifts the halt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from langchain_sync_monitors.contracts import OutcomeName, StepRecord, SubagentHalt
from langchain_sync_monitors.halts import (
    STANDING_HALT_MESSAGE,
    build_standing_halt_decision,
    is_halt_standing,
)
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
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
from tests.support.protocols import AcceptFirst
from tests.unit.middleware.test_middleware_state import build_delegation_step, delegate

REVIEWER_SOURCE = "reviewer"
REVISION_REQUEST = "The task is not done yet. Keep going."
NUDGE = "Call the relevant state-changing tool now instead of replying."
NUDGE_NAME = "nemotron_policy_nudge"


class ReturningToModelMiddleware(AgentMiddleware[Any, Any, Any]):
    """An `after_agent` hook that sends a finished run back to the model, as a grader does.

    Its revision request is tagged with an `lc_source`, as Deep Agents'
    `RubricMiddleware` tags its own, and it returns at most `returns` times.
    """

    def __init__(self, *, returns: int, label: str = "returner") -> None:
        self.returns = returns
        self.label = label

    @property
    def name(self) -> str:
        return self.label

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.build_return(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.build_return(state)

    def build_return(self, state: AgentState[Any]) -> dict[str, Any] | None:
        sent = [
            message
            for message in state["messages"]
            if message.additional_kwargs.get("lc_source") == REVIEWER_SOURCE
        ]
        if len(sent) >= self.returns:
            return None
        revision = HumanMessage(
            REVISION_REQUEST,
            additional_kwargs={"lc_source": REVIEWER_SOURCE},
        )
        return {"messages": [revision], "jump_to": "model"}


class NudgingMiddleware(AgentMiddleware[Any, Any, Any]):
    """A `before_model` hook that nudges the agent with an untagged human message.

    Deep Agents' Nemotron 3 Ultra profile writes its nudges this way, as a
    `HumanMessage` named after their source with no `lc_source` tag, and
    Deep Agents lists a harness profile's middleware after the user's, so
    after the monitor [@deepagents2026]. This one nudges once, on a return
    to the model after a revision request.
    """

    @property
    def name(self) -> str:
        return "nudger"

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.build_nudge(state)

    async def abefore_model(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.build_nudge(state)

    def build_nudge(self, state: AgentState[Any]) -> dict[str, Any] | None:
        messages = state["messages"]
        is_return = bool(messages) and messages[-1].text == REVISION_REQUEST
        if not is_return or any(message.name == NUDGE_NAME for message in messages):
            return None
        return {"messages": [HumanMessage(NUDGE, name=NUDGE_NAME)]}


class StoppingOnceMiddleware(AgentMiddleware[Any, Any, Any]):
    """An `after_agent` hook, listed after the monitor, that raises the first time it runs.

    LangChain runs `after_agent` hooks in reverse list order [@langchain2026],
    so the run stops before the monitor's own hook marks it closed.
    """

    def __init__(self) -> None:
        self.stops: list[str] = []

    @property
    def name(self) -> str:
        return "stopper"

    def after_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.stop_once()

    async def aafter_agent(  # lanorme: ignore[NAMING-011]
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.stop_once()

    def stop_once(self) -> None:
        if not self.stops:
            self.stops.append("stopped")
            message = "the client went away"
            raise ConnectionError(message)


def build_halting_monitor(**options: Any) -> MonitorMiddleware:
    protocol = DeferToResample(
        fallback=HaltRun(),
        defer_threshold=0.5,
        audit_threshold=None,
        max_resamples=0,
    )
    return MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol, **options)


def build_agent(
    model: ScriptedChatModel,
    *,
    workspace: Workspace,
    returns: int,
    checkpointer: InMemorySaver | None = None,
    interrupt_before: list[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ReturningToModelMiddleware(returns=returns),
        build_halting_monitor(),
    ]
    return create_agent(
        model,
        tools=workspace.build_tools(),
        middleware=stack,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


def resume(agent: Runnable[Any, Any], *, mode: RunMode, config: RunnableConfig) -> Any:
    if mode == "invoke":
        return agent.invoke(None, config)
    return asyncio.run(agent.ainvoke(None, config))


def read_outcomes(result: dict[str, Any]) -> list[str]:
    return [record["outcome"] for record in result["monitor_log"]]


def test_a_hook_that_returns_to_the_model_after_a_halt_meets_a_halt_again(
    run_mode: RunMode,
) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )
    agent = build_agent(model, workspace=workspace, returns=2)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the untrusted model was called for the first step only, and nothing ran
    assert len(model.calls) == 1
    assert workspace.executed == []
    log = result["monitor_log"]
    assert read_outcomes(result) == ["halted", "halted", "halted"]
    assert [record["step_number"] for record in log] == [1, 2, 3]
    assert len(log[0]["samples"]) == 1
    assert all(record["samples"] == [] and record["flagged"] for record in log[1:])
    last = result["messages"][-1]
    assert last.text == STANDING_HALT_MESSAGE
    assert (last.id or "").startswith("monitor-")
    assert not last.tool_calls


def test_the_standing_halt_message_is_marked_as_the_monitor_s() -> None:
    # Act
    message = build_standing_halt_decision().response.result[0].text

    # Assert
    assert message == (
        "[Safety monitor] This run is stopped: the safety monitor halted an earlier step, and "
        "that halt stands until a later run brings new input the monitor can confirm. The "
        "rejected steps are recorded in monitor_log for review."
    )
    assert message.startswith("[Safety monitor]")


def test_a_hook_that_returns_to_the_model_after_an_allowed_step_is_sampled(
    run_mode: RunMode,
) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[AIMessage("Here is the summary."), build_read_step(), AIMessage("Done.")],
    )
    agent = build_agent(model, workspace=workspace, returns=1)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: without a halt, the return reaches the untrusted model as usual
    assert len(model.calls) == 3
    assert read_outcomes(result) == ["allowed", "allowed", "allowed"]
    assert workspace.executed == ["read_file:q3.md"]


def test_a_new_user_message_after_a_standing_halt_starts_afresh(run_mode: RunMode) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )
    agent = build_agent(model, workspace=workspace, returns=1, checkpointer=InMemorySaver())
    config = build_thread_config(f"standing-halt-then-new-turn-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    second_turn = run_agent(agent, mode=run_mode, config=config, task="Summarise q3.md only.")

    # Assert
    assert read_outcomes(second_turn) == ["halted", "halted", "allowed", "allowed"]
    assert len(model.calls) == 3
    assert workspace.executed == ["read_file:q3.md"]


def test_a_nudge_written_before_the_next_model_call_never_lifts_a_halt(
    run_mode: RunMode,
) -> None:
    # Arrange: a grader returns the halted run to the model, and a harness hook nudges
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ReturningToModelMiddleware(returns=1),
        build_halting_monitor(),
        NudgingMiddleware(),
    ]
    agent = create_agent(model, tools=workspace.build_tools(), middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the nudge reached the state untagged, yet the untrusted model stayed stopped
    nudges = [message for message in result["messages"] if message.name == NUDGE_NAME]
    assert [nudge.text for nudge in nudges] == [NUDGE]
    assert len(model.calls) == 1
    assert workspace.executed == []
    assert read_outcomes(result) == ["halted", "halted"]
    assert result["monitor_log"][-1]["samples"] == []


def test_a_fresh_run_without_a_new_message_stays_halted(run_mode: RunMode) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(responses=[build_exfiltration_step(), build_read_step()])
    agent = build_agent(model, workspace=workspace, returns=0, checkpointer=InMemorySaver())
    config = build_thread_config(f"fresh-run-without-input-{run_mode}")
    first = run_agent(agent, mode=run_mode, config=config)
    assert read_outcomes(first) == ["halted"]

    # Act: the application runs the thread again, with no message of the user's
    payload: dict[str, Any] = {"messages": []}
    if run_mode == "invoke":
        second = agent.invoke(payload, config)
    else:
        second = asyncio.run(agent.ainvoke(payload, config))

    # Assert
    assert read_outcomes(second) == ["halted", "halted"]
    assert second["monitor_log"][-1]["samples"] == []
    assert len(model.calls) == 1
    assert workspace.executed == []


def test_a_message_the_monitor_cannot_confirm_as_input_never_lifts_a_halt(
    run_mode: RunMode,
) -> None:
    # Arrange: the halted run stops before its end, so the next run's input is unconfirmed
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        build_halting_monitor(),
        StoppingOnceMiddleware(),
    ]
    agent = create_agent(
        model,
        tools=workspace.build_tools(),
        middleware=stack,
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"unconfirmed-after-halt-{run_mode}")
    with pytest.raises(ConnectionError):
        run_agent(agent, mode=run_mode, config=config)

    # Act
    unconfirmed_turn = run_agent(agent, mode=run_mode, config=config, task="Go on, safely.")
    confirmed_turn = run_agent(agent, mode=run_mode, config=config, task="Summarise q3.md.")

    # Assert: the unconfirmed turn stays halted, and the next turn's input lifts the halt
    unconfirmed = [
        message
        for message in unconfirmed_turn["messages"]
        if message.additional_kwargs.get("lc_source") == "unconfirmed_input"
    ]
    assert [message.text for message in unconfirmed] == ["Go on, safely."]
    assert read_outcomes(unconfirmed_turn) == ["halted", "halted"]
    assert read_outcomes(confirmed_turn) == ["halted", "halted", "allowed", "allowed"]
    assert len(model.calls) == 3
    assert workspace.executed == ["read_file:q3.md"]


def test_resuming_between_the_halt_and_the_hook_halts_again(run_mode: RunMode) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(responses=[build_exfiltration_step(), build_read_step()])
    agent = build_agent(
        model,
        workspace=workspace,
        returns=1,
        checkpointer=InMemorySaver(),
        interrupt_before=["returner.after_agent"],
    )
    config = build_thread_config(f"resume-after-halt-{run_mode}")
    paused = run_agent(agent, mode=run_mode, config=config)
    assert read_outcomes(paused) == ["halted"]

    # Act
    resumed = resume(agent, mode=run_mode, config=config)

    # Assert
    assert read_outcomes(resumed) == ["halted", "halted"]
    assert resumed["monitor_log"][-1]["samples"] == []
    assert len(model.calls) == 1
    assert workspace.executed == []


def test_a_fork_from_before_the_halt_samples_its_own_branch(run_mode: RunMode) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_read_step(call_id="call-read-1"),
            build_exfiltration_step(),
            build_read_step(call_id="call-read-2"),
            AIMessage("Done."),
            AIMessage("Done, as revised."),
        ],
    )
    agent = build_agent(model, workspace=workspace, returns=1, checkpointer=InMemorySaver())
    config = build_thread_config(f"fork-before-halt-{run_mode}")
    first = run_agent(agent, mode=run_mode, config=config)
    assert read_outcomes(first) == ["allowed", "halted", "halted"]
    fork = next(
        snapshot
        for snapshot in agent.get_state_history(config)
        if snapshot.next == ("model",) and len(snapshot.values.get("monitor_log", [])) == 1
    )

    # Act
    forked = resume(agent, mode=run_mode, config=fork.config)

    # Assert: the fork holds no halt, so its steps, the returned one included, are sampled
    assert read_outcomes(forked) == ["allowed", "allowed", "allowed", "allowed"]
    assert len(model.calls) == 5
    assert workspace.executed == ["read_file:q3.md", "read_file:q3.md"]


def test_a_parent_that_stops_the_whole_run_stays_stopped(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [ReturningToModelMiddleware(returns=1), monitor]
    agent = create_agent(model, tools=[delegate], middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert len(model.calls) == 1
    assert [(record["agent"], record["outcome"]) for record in result["monitor_log"]] == [
        ("main", "allowed"),
        ("researcher", "halted"),
        ("main", "halted"),
        ("main", "halted"),
    ]
    assert result["messages"][-1].text == STANDING_HALT_MESSAGE


def test_an_inner_monitor_keeps_its_halt_under_an_outer_monitor(run_mode: RunMode) -> None:
    # Arrange
    workspace = Workspace()
    model = ScriptedChatModel(responses=[build_exfiltration_step(), build_read_step()])
    outer = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=AcceptFirst(), label="guard"
    )
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        ReturningToModelMiddleware(returns=1),
        outer,
        build_halting_monitor(),
    ]
    agent = create_agent(model, tools=workspace.build_tools(), middleware=stack)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    inner_outcomes = [
        record["outcome"] for record in result["monitor_log"] if record["monitor"] == "monitor"
    ]
    assert inner_outcomes == ["halted", "halted"]
    assert len(model.calls) == 1
    assert workspace.executed == []


def build_record(*, outcome: OutcomeName) -> StepRecord:
    return StepRecord(
        agent="main",
        monitor="monitor",
        step_number=1,
        outcome=outcome,
        flagged=outcome == "halted",
        blocked_count=0,
        samples=[],
    )


@pytest.mark.parametrize(
    ("records", "run_inputs", "run_inputs_at_halt", "stands"),
    [
        ([build_record(outcome="halted")], 1, 1, True),
        ([build_record(outcome="halted")], 2, 1, False),
        ([build_record(outcome="halted")], 1, None, True),
        ([build_record(outcome="allowed")], 1, 1, False),
        ([build_record(outcome="allowed")], 1, None, False),
        ([], 1, None, False),
    ],
    ids=[
        "no-input-since-the-halt",
        "a-run-input-since-the-halt",
        "a-halt-without-its-count",
        "after-an-allowed-step",
        "after-an-allowed-step-without-a-count",
        "before-any-step",
    ],
)
def test_a_halt_stands_until_the_thread_records_another_run_input(
    records: list[StepRecord],
    run_inputs: int,
    run_inputs_at_halt: int | None,
    stands: bool,
) -> None:
    # Act
    standing = is_halt_standing(
        records,
        run_inputs=run_inputs,
        run_inputs_at_halt=run_inputs_at_halt,
    )

    # Assert
    assert standing is stands
