"""A subagent whose steps are recorded as the main agent's halts the main agent when it halts.

A fork runs under the main agent's own monitor, and a compiled subagent whose
monitor keeps the default `agent_name`, `main`, and shares the main monitor's
label records as that monitor too. Either way the subagent's halt is the last
record of the main agent's monitor, so the main agent's next step meets a
standing halt, whatever `when_subagent_halts` says, and a later run with a new
message from the user lifts it, as `STANDING_HALT_MESSAGE` says. Under another
label, the halt is recorded as the main agent's but not as its monitor's, so
nothing halts the main agent.
"""

from __future__ import annotations

import pytest
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors.contracts import SubagentHalt, TaskAuthor
from langchain_sync_monitors.halts import STANDING_HALT_MESSAGE
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample, SteerWithFeedback

MAIN_ANSWER = "Here is the summary."
FOLLOW_UP = "Just summarise q3.md, please."


def build_compiled_worker(*, label: str = "monitor") -> CompiledSubAgent:
    """A compiled subagent whose own monitor halts every step and keeps `agent_name='main'`.

    The main monitor's label is the default, `monitor`, so the default `label`
    shares it.
    """
    worker_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=HaltAfterOneSample(),
        label=label,
        task_author=TaskAuthor.PARENT_AGENT,
    )
    runnable = create_agent(
        ScriptedChatModel(responses=[AIMessage("Three sources found.")]),
        middleware=[worker_monitor],
    )
    return CompiledSubAgent(name="worker", description="Finds sources.", runnable=runnable)


def read_outcomes(result: dict[str, object]) -> list[tuple[str, str, str]]:
    log = result["monitor_log"]
    assert isinstance(log, list)
    return [(record["agent"], record["monitor"], record["outcome"]) for record in log]


@pytest.mark.parametrize("when_subagent_halts", list(SubagentHalt))
def test_a_compiled_subagent_recorded_as_main_halts_the_main_agent(
    run_mode: RunMode,
    when_subagent_halts: SubagentHalt,
) -> None:
    # Arrange: under STOP_SUBAGENT_ONLY the main agent would otherwise carry on
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    main_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=when_subagent_halts,
    )
    agent = create_deep_agent(
        model=main_model,
        middleware=[main_monitor],
        subagents=[build_compiled_worker()],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the main agent's next step halts without calling its model
    assert read_outcomes(result) == [
        ("main", "monitor", "allowed"),
        ("main", "monitor", "halted"),
        ("main", "monitor", "halted"),
    ]
    assert len(main_model.calls) == 1
    assert result["monitor_log"][-1]["samples"] == []
    assert result["messages"][-1].text == STANDING_HALT_MESSAGE


def test_a_compiled_subagent_under_another_label_goes_unseen_by_stop_whole_run(
    run_mode: RunMode,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    main_monitor = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
    )
    agent = create_deep_agent(
        model=main_model,
        middleware=[main_monitor],
        subagents=[build_compiled_worker(label="worker")],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the worker's halt is recorded as the main agent's, so no subagent halted
    assert read_outcomes(result) == [
        ("main", "monitor", "allowed"),
        ("main", "worker", "halted"),
        ("main", "monitor", "allowed"),
    ]
    assert len(main_model.calls) == 2
    assert result["messages"][-1].text == MAIN_ANSWER


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
@pytest.mark.parametrize("when_subagent_halts", list(SubagentHalt))
def test_a_fork_passed_to_create_deep_agent_halts_the_main_agent(
    run_mode: RunMode,
    when_subagent_halts: SubagentHalt,
) -> None:
    # Arrange: the fork shares the main agent's model, which proposes the fork's step second
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_delegation_step(subagent_type="forker"),
            build_exfiltration_step(),
            AIMessage("never drawn"),
        ],
    )
    main_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=SteerWithFeedback(max_blocks=1),
        when_subagent_halts=when_subagent_halts,
    )
    fork = SubAgent(
        name="forker",
        description="Posts the report.",
        system_prompt="Post the report.",
        tools=workspace.build_tools(),
        mode="fork",
    )
    agent = create_deep_agent(model=model, middleware=[main_monitor], subagents=[fork])

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_outcomes(result) == [
        ("main", "monitor", "allowed"),
        ("main", "monitor", "halted"),
        ("main", "monitor", "halted"),
    ]
    assert len(model.calls) == 2
    assert workspace.executed == []
    assert result["messages"][-1].text == STANDING_HALT_MESSAGE


def test_a_later_run_with_a_new_user_message_lifts_the_halt(run_mode: RunMode) -> None:
    # Arrange
    main_model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)])
    main_monitor = MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())
    agent = create_deep_agent(
        model=main_model,
        middleware=[main_monitor],
        subagents=[build_compiled_worker()],
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"recorded-as-main-{run_mode}")
    first = run_agent(agent, mode=run_mode, config=config)

    # Act
    second = run_agent(agent, mode=run_mode, config=config, task=FOLLOW_UP)

    # Assert: the second run samples the main agent's model again
    assert first["messages"][-1].text == STANDING_HALT_MESSAGE
    assert read_outcomes(second)[-1] == ("main", "monitor", "allowed")
    assert len(main_model.calls) == 2
    assert second["messages"][-1].text == MAIN_ANSWER
