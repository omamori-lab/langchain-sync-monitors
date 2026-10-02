"""A subagent that shares its parent's name keeps its own delegation, so its halts are answered.

A fork runs under the main agent's monitor, and a compiled subagent whose
monitor keeps the default `agent_name` records under `main` too. Each still
records the id of the call that started it, so the parent never takes the
subagent's steps for its own, and answers its halts as `SubagentHalt` says,
on a fresh thread and after a halt of its own.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.contracts import SubagentHalt, TaskAuthor
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_thread_config,
    run_messages,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import HaltAfterOneSample, SteerWithFeedback

type Thread = Literal["fresh", "after-a-halt-of-its-own"]
MAIN_ANSWER = "MAIN CARRIES ON"
CALL_ID = "call-task"


@dataclass(frozen=True, kw_only=True)
class SubagentKind:
    """How to build a parent whose subagent halts, and how the halt message names it."""

    build: Callable[[SubagentHalt, list[AIMessage]], tuple[Any, ScriptedChatModel]]
    subagent_type: str
    shares_the_model: bool
    halted_name: str


def build_main_monitor(when_subagent_halts: SubagentHalt) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=SteerWithFeedback(max_blocks=1),
        when_subagent_halts=when_subagent_halts,
    )


def build_fork_parent(
    when_subagent_halts: SubagentHalt,
    responses: list[AIMessage],
) -> tuple[CompiledStateGraph[Any, Any, Any, Any], ScriptedChatModel]:
    model = ScriptedChatModel(responses=responses)
    fork = SubAgent(
        name="forker",
        description="Posts.",
        system_prompt="Post.",
        tools=Workspace().build_tools(),
        mode="fork",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = create_deep_agent(
            model=model,
            middleware=[build_main_monitor(when_subagent_halts)],
            subagents=[fork],
            checkpointer=InMemorySaver(),
        )
    return agent, model


def build_compiled_parent_factory(
    *,
    agent_name: str | None,
    label: str,
) -> Callable[[SubagentHalt, list[AIMessage]], tuple[Any, ScriptedChatModel]]:
    def build(
        when_subagent_halts: SubagentHalt,
        responses: list[AIMessage],
    ) -> tuple[CompiledStateGraph[Any, Any, Any, Any], ScriptedChatModel]:
        worker_monitor = MonitorMiddleware(
            monitor=KeywordMonitor(),
            protocol=HaltAfterOneSample(),
            label=label,
            agent_name=agent_name or "main",
            task_author=TaskAuthor.PARENT_AGENT,
        )
        runnable = create_agent(
            ScriptedChatModel(responses=[AIMessage("Three sources.")]),
            middleware=[worker_monitor],
        )
        worker = CompiledSubAgent(name="worker", description="Finds.", runnable=runnable)
        model = ScriptedChatModel(responses=responses)
        agent = create_deep_agent(
            model=model,
            middleware=[build_main_monitor(when_subagent_halts)],
            subagents=[worker],
            checkpointer=InMemorySaver(),
        )
        return agent, model

    return build


def build_declarative_main_parent(
    when_subagent_halts: SubagentHalt,
    responses: list[AIMessage],
) -> tuple[CompiledStateGraph[Any, Any, Any, Any], ScriptedChatModel]:
    main_monitor = build_main_monitor(when_subagent_halts)
    spec = SubAgent(
        name="main",
        description="Posts.",
        system_prompt="Post.",
        tools=Workspace().build_tools(),
        model=ScriptedChatModel(responses=[build_exfiltration_step(call_id="call-post-sub")]),
    )
    model = ScriptedChatModel(responses=responses)
    agent = create_deep_agent(
        model=model,
        middleware=[main_monitor],
        subagents=monitor_subagents(middleware=main_monitor, subagents=[spec]),
        checkpointer=InMemorySaver(),
    )
    return agent, model


KINDS = {
    "fork": SubagentKind(
        build=build_fork_parent,
        subagent_type="forker",
        shares_the_model=True,
        halted_name="forker",
    ),
    "compiled-default-name": SubagentKind(
        build=build_compiled_parent_factory(agent_name=None, label="monitor"),
        subagent_type="worker",
        shares_the_model=False,
        halted_name="worker",
    ),
    "compiled-default-name-own-label": SubagentKind(
        build=build_compiled_parent_factory(agent_name=None, label="worker"),
        subagent_type="worker",
        shares_the_model=False,
        halted_name="worker",
    ),
    "compiled-named-control": SubagentKind(
        build=build_compiled_parent_factory(agent_name="worker", label="monitor"),
        subagent_type="worker",
        shares_the_model=False,
        halted_name="worker",
    ),
    "declarative-named-main": SubagentKind(
        build=build_declarative_main_parent,
        subagent_type="main",
        shares_the_model=False,
        halted_name="main",
    ),
}


def build_main_responses(kind: SubagentKind, *, thread: Thread) -> list[AIMessage]:
    """Script the main agent: a halt of its own on the first turn, if any, then the delegation."""
    responses: list[AIMessage] = []
    if thread == "after-a-halt-of-its-own":
        responses.append(build_exfiltration_step(call_id="call-post-main"))
    responses.append(build_delegation_step(call_id=CALL_ID, subagent_type=kind.subagent_type))
    if kind.shares_the_model:
        responses.append(build_exfiltration_step(call_id="call-post-fork"))
    return [*responses, AIMessage(MAIN_ANSWER)]


def run_thread(agent: Any, *, mode: RunMode, thread: Thread) -> dict[str, Any]:
    """Run the turns of the thread, and return the result of the last one."""
    config = build_thread_config(f"{thread}-{mode}")
    texts = ["Post the report."] if thread == "after-a-halt-of-its-own" else []
    result: dict[str, Any] = {}
    for text in [*texts, "Summarise q3.md."]:
        result = run_messages(agent, [{"role": "user", "content": text}], mode=mode, config=config)
    return result


def read_main_steps(result: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (record["step_number"], record["outcome"])
        for record in result["monitor_log"]
        if record["agent"] == "main" and "delegation_id" not in record
    ]


THREADS: list[Thread] = ["fresh", "after-a-halt-of-its-own"]


@pytest.mark.parametrize("thread", THREADS)
@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_stop_whole_run_halts_the_parent_when_a_same_named_subagent_halts(
    run_mode: RunMode,
    kind: SubagentKind,
    thread: Thread,
) -> None:
    # Arrange
    responses = build_main_responses(kind, thread=thread)
    agent, model = kind.build(SubagentHalt.STOP_WHOLE_RUN, responses)

    # Act
    result = run_thread(agent, mode=run_mode, thread=thread)

    # Assert: the parent's next step halts without drawing the answer it was scripted
    assert result["messages"][-1].text == (
        f"[Safety monitor] Stopped: the safety monitor halted the subagent {kind.halted_name}, "
        "so this agent stops too."
    )
    assert len(model.calls) == len(responses) - 1
    main_steps = read_main_steps(result)
    assert main_steps[-1] == (len(main_steps), "halted")


@pytest.mark.parametrize("thread", THREADS)
@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_stop_subagent_only_lets_the_parent_carry_on_after_a_same_named_subagent_halts(
    run_mode: RunMode,
    kind: SubagentKind,
    thread: Thread,
) -> None:
    # Arrange
    responses = build_main_responses(kind, thread=thread)
    agent, model = kind.build(SubagentHalt.STOP_SUBAGENT_ONLY, responses)

    # Act
    result = run_thread(agent, mode=run_mode, thread=thread)

    # Assert
    assert result["messages"][-1].text == MAIN_ANSWER
    assert len(model.calls) == len(responses)
    assert read_main_steps(result)[-1] == (len(read_main_steps(result)), "allowed")


@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_the_parent_numbers_its_steps_apart_from_a_same_named_subagent_s(
    run_mode: RunMode,
    kind: SubagentKind,
) -> None:
    # Arrange
    responses = build_main_responses(kind, thread="after-a-halt-of-its-own")
    agent, _model = kind.build(SubagentHalt.STOP_SUBAGENT_ONLY, responses)

    # Act
    result = run_thread(agent, mode=run_mode, thread="after-a-halt-of-its-own")

    # Assert: main's third step is numbered 3, after the subagent's own step 1
    assert read_main_steps(result) == [(1, "halted"), (2, "allowed"), (3, "allowed")]
    subagent_steps = [
        (record["step_number"], record["outcome"], record["delegation_id"])
        for record in result["monitor_log"]
        if "delegation_id" in record
    ]
    assert subagent_steps == [(1, "halted", CALL_ID)]
