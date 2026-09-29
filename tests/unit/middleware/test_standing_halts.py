"""A halt stands for the rest of the run, even when a hook sends the agent back to the model.

An `after_model` hook's `jump_to: "end"` reaches the agent's exit node, which
is the first `after_agent` hook when there is one [@langchain2026], and such a
hook can jump back to the model, as Deep Agents' `RubricMiddleware` does. The
monitor then halts every further step without drawing a sample, until the task
author writes again.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from langchain_sync_monitors.contracts import OutcomeName, StepRecord, SubagentHalt
from langchain_sync_monitors.halts import STANDING_HALT_MESSAGE, is_halt_standing
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


TASK = HumanMessage("Summarise q3.md.")
HALT = AIMessage("[Safety monitor] Stopped.")
REVISION = HumanMessage(REVISION_REQUEST, additional_kwargs={"lc_source": REVIEWER_SOURCE})
NEW_TASK = HumanMessage("Summarise q4.md instead.")


@pytest.mark.parametrize(
    ("records", "messages", "stands"),
    [
        ([build_record(outcome="halted")], [TASK, HALT], True),
        ([build_record(outcome="halted")], [TASK, HALT, REVISION], True),
        ([build_record(outcome="halted")], [TASK, HALT, REVISION, NEW_TASK], False),
        ([build_record(outcome="halted")], [TASK, HALT, NEW_TASK, REVISION], False),
        ([build_record(outcome="allowed")], [TASK, HALT, REVISION], False),
        ([], [TASK], False),
    ],
    ids=[
        "right-after-the-halt",
        "after-a-tagged-message",
        "after-a-new-task",
        "after-a-new-task-then-a-tagged-message",
        "after-an-allowed-step",
        "before-any-step",
    ],
)
def test_a_halt_stands_until_the_task_author_writes_after_it(
    records: list[StepRecord],
    messages: list[BaseMessage],
    stands: bool,
) -> None:
    # Act
    standing = is_halt_standing(records, messages=messages)

    # Assert
    assert standing is stands
