"""A step held for a person to audit in a run without a checkpointer warns, once per process.

LangGraph saves a paused run in a checkpointer, so a run without one stops at
the audit and cannot be resumed. The warning names the two ways out: a
checkpointer for the agent, or `AuditResponse.FLAG`, which runs the step and
flags it. It reads the checkpointer from the config LangGraph gives the
node, which holds the saver a parent lends to its subgraph, and it stays
silent when the config does not say.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest
from hyoshigi import AuditResponse
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.constants import CONFIG_KEY_CHECKPOINTER
from langgraph.graph import END, START, StateGraph, add_messages

from langchain_sync_monitors import DeferToResample, MissingCheckpointerWarning
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import ProtocolConfiguration
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel

pytestmark = pytest.mark.usefixtures("fresh_checkpointer_warning")

THREAD = build_thread_config("checkpointer")
HOLD_SUSPICIOUS = DeferToResample(defer_threshold=0.5, audit_threshold=0.5, max_resamples=0)
"""Holds a step at 0.5 or above for a person, drawing no resample."""
FLAG_SUSPICIOUS = DeferToResample(
    defer_threshold=0.5,
    audit_threshold=0.5,
    max_resamples=0,
    on_audit=AuditResponse.FLAG,
)
"""Runs and flags a step at 0.5 or above."""


@dataclass
class ParentState:
    """The messages a parent graph passes to the agent it holds."""

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


def build_agent(
    *,
    checkpointer: InMemorySaver | None,
    protocol: ProtocolConfiguration = HOLD_SUSPICIOUS,
    middleware: MonitorMiddleware | None = None,
) -> Any:
    """Build an agent whose first step is suspicious, so that the protocol holds or flags it."""
    monitor = middleware or MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    return create_agent(
        ScriptedChatModel(responses=[build_exfiltration_step(), AIMessage("Done.")]),
        tools=Workspace().build_tools(),
        middleware=[monitor],
        checkpointer=checkpointer,
    )


@contextmanager
def collect_every_warning() -> Generator[list[warnings.WarningMessage]]:
    """Record each warning raised inside the block, however often it was raised before."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def read_checkpointer_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in caught if item.category is MissingCheckpointerWarning]


@pytest.mark.xfail(strict=True, reason="the audit's pause does not warn yet")
@pytest.mark.parametrize(
    "config",
    [None, THREAD],
    ids=["without a thread id", "with a thread id"],
)
def test_a_held_audit_without_a_checkpointer_warns_once(
    run_mode: RunMode,
    config: RunnableConfig | None,
) -> None:
    # Arrange
    agent = build_agent(checkpointer=None)

    # Act
    with collect_every_warning() as caught:
        result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    assert len(read_checkpointer_warnings(caught)) == 1
    assert len(result["__interrupt__"]) == 1


@pytest.mark.xfail(strict=True, reason="the audit's pause does not warn yet")
def test_the_warning_names_the_checkpointer_and_the_flag_response(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(checkpointer=None)

    # Act
    with collect_every_warning() as caught:
        run_agent(agent, mode=run_mode)

    # Assert
    messages = read_checkpointer_warnings(caught)
    assert len(messages) == 1
    assert "create_agent(checkpointer=...)" in messages[0]
    assert "AuditResponse.FLAG" in messages[0]


def test_a_held_audit_with_a_checkpointer_does_not_warn(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(checkpointer=InMemorySaver())

    # Act
    with collect_every_warning() as caught:
        result = run_agent(agent, mode=run_mode, config=THREAD)

    # Assert
    assert read_checkpointer_warnings(caught) == []
    assert len(result["__interrupt__"]) == 1


@pytest.mark.xfail(strict=True, reason="the audit's pause does not warn yet")
def test_a_second_held_audit_in_the_process_does_not_warn_again(run_mode: RunMode) -> None:
    # Arrange
    first = build_agent(checkpointer=None)
    second = build_agent(checkpointer=None)

    # Act
    with collect_every_warning() as caught:
        run_agent(first, mode=run_mode)
        run_agent(second, mode=run_mode)

    # Assert
    assert len(read_checkpointer_warnings(caught)) == 1


@pytest.mark.xfail(strict=True, reason="the audit's pause does not warn yet")
def test_an_audit_with_a_checkpointer_leaves_the_warning_for_the_next_audit_without_one(
    run_mode: RunMode,
) -> None:
    # Arrange
    saved = build_agent(checkpointer=InMemorySaver())
    unsaved = build_agent(checkpointer=None)

    # Act
    with collect_every_warning() as caught:
        run_agent(saved, mode=run_mode, config=THREAD)
        run_agent(unsaved, mode=run_mode)

    # Assert
    assert len(read_checkpointer_warnings(caught)) == 1


def test_an_agent_that_its_parent_checkpoints_does_not_warn(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(checkpointer=None)
    parent = (
        StateGraph(ParentState)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=InMemorySaver())
    )

    # Act
    with collect_every_warning() as caught:
        result = run_agent(parent, mode=run_mode, config=THREAD)

    # Assert
    assert read_checkpointer_warnings(caught) == []
    assert len(result["__interrupt__"]) == 1


def test_a_flagged_audit_runs_without_a_checkpointer_and_does_not_warn(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(checkpointer=None, protocol=FLAG_SUSPICIOUS)

    # Act
    with collect_every_warning() as caught:
        result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_checkpointer_warnings(caught) == []
    assert "__interrupt__" not in result


@contextmanager
def lack_checkpointer_key() -> Generator[None]:
    """Run the block in a node config that holds no checkpointer key, as an unknown LangGraph's."""
    config = get_config()
    configurable = dict(config["configurable"])
    del configurable[CONFIG_KEY_CHECKPOINTER]
    token = var_child_runnable_config.set({**config, "configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


class MiddlewareWithoutCheckpointerKey(MonitorMiddleware):
    """The monitor middleware, whose pause runs in a config that holds no checkpointer key."""

    def pause_without_key(self, state: Any, runtime: Any) -> Any:
        with lack_checkpointer_key():
            return MonitorMiddleware.after_model(self, state, runtime)

    async def pause_without_key_async(self, state: Any, runtime: Any) -> Any:
        with lack_checkpointer_key():
            return await MonitorMiddleware.aafter_model(self, state, runtime)

    after_model = pause_without_key
    aafter_model = pause_without_key_async


def test_a_config_without_the_checkpointer_key_does_not_warn(run_mode: RunMode) -> None:
    # Arrange
    middleware = MiddlewareWithoutCheckpointerKey(
        monitor=build_keyword_monitor(),
        protocol=HOLD_SUSPICIOUS,
    )
    agent = build_agent(checkpointer=None, middleware=middleware)

    # Act
    with collect_every_warning() as caught:
        result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_checkpointer_warnings(caught) == []
    assert len(result["__interrupt__"]) == 1


class RecordCheckpointerKey(AgentMiddleware[Any, Any, Any]):
    """Records what the node config holds under LangGraph's checkpointer key after a model call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []

    def record_key(self, state: Any, runtime: Any) -> None:
        self.seen.append(get_config()["configurable"].get(CONFIG_KEY_CHECKPOINTER, "no key"))

    async def record_key_async(self, state: Any, runtime: Any) -> None:
        self.record_key(state, runtime)

    after_model = record_key
    aafter_model = record_key_async


def test_a_run_without_a_checkpointer_sees_the_key_as_none(run_mode: RunMode) -> None:
    # Arrange
    probe = RecordCheckpointerKey()
    agent = create_agent(
        ScriptedChatModel(responses=[AIMessage("Done.")]),
        middleware=[probe],
    )

    # Act
    run_agent(agent, mode=run_mode)

    # Assert
    assert probe.seen == [None]
