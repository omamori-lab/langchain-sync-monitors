"""The monitor log accumulates, survives checkpoints and carries subagent halts."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Annotated, Any, Literal

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_sync_monitors.contracts import StepRecord, SubagentHalt, TaskAuthor
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitor_state import MonitorState
from langchain_sync_monitors.state_keys import MONITOR_STATE_KEYS
from tests.support.agents import (
    RunMode,
    build_keyword_monitor,
    build_thread_config,
    read_texts,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.protocols import AcceptFirst
from tests.support.written_human_messages import NudgingMiddleware, rewrite_history

SUBAGENT_HALT = StepRecord(
    agent="researcher",
    monitor="monitor",
    step_number=1,
    outcome="halted",
    flagged=True,
    blocked_count=0,
    samples=[],
)


@tool
def delegate(description: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[None]:
    """Hand a task to the researcher subagent, which the safety monitor halts."""
    report = ToolMessage("Stopped by the safety monitor.", tool_call_id=tool_call_id)
    return Command[None](update={"monitor_log": [SUBAGENT_HALT], "messages": [report]})


@dataclasses.dataclass
class DelegationReport:
    """A delegation's update as a dataclass: the subagent's records, then its report."""

    monitor_log: list[StepRecord]
    messages: list[ToolMessage]


def build_delegating_tool(shape: Literal["pairs", "dataclass"]) -> BaseTool:
    """Return a `delegate` tool whose update carries the subagent's halt in `shape`."""

    @tool
    def delegate(
        description: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[None]:
        """Hand a task to the researcher subagent, which the safety monitor halts."""
        report = ToolMessage("Stopped by the safety monitor.", tool_call_id=tool_call_id)
        if shape == "pairs":
            return Command[None](update=(("monitor_log", [SUBAGENT_HALT]), ("messages", [report])))
        update = DelegationReport(monitor_log=[SUBAGENT_HALT], messages=[report])
        return Command[None](update=update)

    return delegate


def build_delegation_step() -> AIMessage:
    return build_tool_call_message(
        tool_name="delegate",
        call_id="call-delegate",
        arguments={"description": "Find the sources."},
    )


def build_parent_agent(
    *,
    model: ScriptedChatModel,
    when_subagent_halts: SubagentHalt,
    checkpointer: InMemorySaver | None = None,
    delegating_tool: BaseTool = delegate,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AcceptFirst(),
        when_subagent_halts=when_subagent_halts,
    )
    tools: list[BaseTool] = [delegating_tool]
    return create_agent(model, tools=tools, middleware=[middleware], checkpointer=checkpointer)


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())


def test_the_log_is_an_appending_channel(middleware: MonitorMiddleware) -> None:
    # Act
    agent = create_agent(ScriptedChatModel(responses=[]), middleware=[middleware])

    # Assert
    assert isinstance(agent.channels["monitor_log"], BinaryOperatorAggregate)


def test_records_round_trip_through_a_checkpointer_and_keep_counting(
    run_mode: RunMode,
    middleware: MonitorMiddleware,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[AIMessage("first answer"), AIMessage("second answer")])
    agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
    config = build_thread_config(f"thread-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    stored = agent.get_state(config).values["monitor_log"]
    assert stored == result["monitor_log"]
    assert [record["step_number"] for record in stored] == [1, 2]
    assert json.loads(json.dumps(stored)) == stored


def test_the_parent_halts_before_its_next_model_call_when_a_subagent_halted(
    run_mode: RunMode,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    agent = build_parent_agent(model=model, when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert len(model.calls) == 1
    halt = result["messages"][-1]
    assert "halted the subagent researcher" in halt.text
    assert (halt.id or "").startswith("monitor-")
    log = result["monitor_log"]
    assert [(record["agent"], record["outcome"]) for record in log] == [
        ("main", "allowed"),
        ("researcher", "halted"),
        ("main", "halted"),
    ]
    assert log[-1]["samples"] == []


@pytest.mark.parametrize("shape", ["pairs", "dataclass"])
def test_a_subagent_halt_reaches_the_parent_whatever_the_update_shape(
    run_mode: RunMode,
    shape: Literal["pairs", "dataclass"],
) -> None:
    # Arrange: the monitor reads the records from the state, once LangGraph has written them
    model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("never drawn")])
    agent = build_parent_agent(
        model=model,
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
        delegating_tool=build_delegating_tool(shape),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert len(model.calls) == 1
    assert [(record["agent"], record["outcome"]) for record in result["monitor_log"]] == [
        ("main", "allowed"),
        ("researcher", "halted"),
        ("main", "halted"),
    ]
    assert "Stopped by the safety monitor." in read_texts(result["messages"])


def test_the_parent_carries_on_when_only_the_subagent_stops(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("Carrying on alone.")])
    agent = build_parent_agent(model=model, when_subagent_halts=SubagentHalt.STOP_SUBAGENT_ONLY)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert read_texts(result["messages"])[-1] == "Carrying on alone."
    assert [record["outcome"] for record in result["monitor_log"]] == [
        "allowed",
        "halted",
        "allowed",
    ]


def test_a_halted_thread_takes_new_turns_again(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_delegation_step(), AIMessage("New turn answer.")])
    agent = build_parent_agent(
        model=model,
        when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"halted-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config, task="Try something else.")

    # Assert
    assert len(model.calls) == 2
    assert read_texts(result["messages"])[-1] == "New turn answer."
    assert result["monitor_log"][-1]["outcome"] == "allowed"


def test_two_monitors_on_one_agent_count_their_own_steps(run_mode: RunMode) -> None:
    # Arrange
    outer_protocol, inner_protocol = AcceptFirst(), AcceptFirst()
    guard = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=outer_protocol, label="guard"
    )
    judge = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=inner_protocol)
    model = ScriptedChatModel(responses=[AIMessage("first"), AIMessage("second")])
    agent = create_agent(model, middleware=[guard, judge], checkpointer=InMemorySaver())
    config = build_thread_config(f"labels-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    log = result["monitor_log"]
    assert sorted((record["monitor"], record["step_number"]) for record in log) == [
        ("guard", 1),
        ("guard", 2),
        ("monitor", 1),
        ("monitor", 2),
    ]
    assert [len(records) for records in inner_protocol.seen_previous_records] == [0, 1]
    assert all(record["monitor"] == "monitor" for record in inner_protocol.seen_previous_records[1])


@pytest.mark.parametrize(
    "key",
    [
        "monitor_task_messages",
        "monitor_seen_human_messages",
        "monitor_run_inputs",
        "monitor_rewritten_inputs",
        "monitor_inputs_at_halt",
        "monitor_subagent_returns",
    ],
)
def test_the_message_ids_and_halt_counts_the_monitor_records_are_private(
    middleware: MonitorMiddleware,
    key: str,
) -> None:
    # Act
    agent = create_agent(ScriptedChatModel(responses=[]), middleware=[middleware])

    # Assert
    assert key in agent.channels
    assert key not in agent.get_input_jsonschema()["properties"]
    assert key not in agent.get_output_jsonschema()["properties"]


def test_two_monitors_on_one_agent_record_each_message_id_once(run_mode: RunMode) -> None:
    # Arrange: the nudge is written after both monitors' before_model hooks, so both
    # monitors record it when they commit, in the same model node
    guard = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=AcceptFirst(), label="guard"
    )
    judge = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ScriptedChatModel(responses=[AIMessage("first"), AIMessage("second")])
    middleware: list[AgentMiddleware[Any, Any, Any]] = [guard, judge, NudgingMiddleware()]
    agent = create_agent(model, middleware=middleware, checkpointer=InMemorySaver())
    config = build_thread_config(f"stacked-ids-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    run_agent(agent, mode=run_mode, config=config)

    # Assert
    state = agent.get_state(config).values
    first_task, nudge, second_task = (
        message.id for message in state["messages"] if message.type == "human"
    )
    assert state["monitor_task_messages"] == [first_task, second_task]
    assert state["monitor_seen_human_messages"] == [first_task, nudge, second_task]
    assert [entry["id"] for entry in state["monitor_run_inputs"]] == [first_task, second_task]


def test_two_monitors_on_one_agent_check_a_tool_s_writes_once(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the tool writes the history back, so the outer monitor records a rewritten input
    guard = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=AcceptFirst(), label="guard"
    )
    judge = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    model = ScriptedChatModel(
        responses=[
            build_tool_call_message(tool_name="rewrite_history", call_id="call-rewrite"),
            AIMessage("Done."),
        ],
    )
    agent = create_agent(
        model,
        tools=[rewrite_history],
        middleware=[guard, judge],
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"stacked-tool-{run_mode}")

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors"):
        run_agent(agent, mode=run_mode, config=config)

    # Assert: the inner monitor's own record is not taken for the tool's write
    state = agent.get_state(config).values
    assert state["monitor_rewritten_inputs"] == state["monitor_task_messages"]
    assert caplog.records == []


def test_names_are_unique_per_agent_and_subagent_copies_trust_the_parent_less(
    middleware: MonitorMiddleware,
) -> None:
    # Act
    copy = middleware.copy_for_subagent(subagent_name="researcher")

    # Assert
    assert middleware.name == "monitor[main]"
    assert copy.name == "monitor[researcher]"
    assert copy.task_author is TaskAuthor.PARENT_AGENT
    assert copy.monitor is middleware.monitor
    assert copy.protocol is middleware.protocol
    assert middleware.task_author is TaskAuthor.USER


def test_the_middleware_holds_no_mutable_run_state(middleware: MonitorMiddleware) -> None:
    # Act / Assert
    with pytest.raises(dataclasses.FrozenInstanceError):
        middleware.agent_name = "changed"  # ty: ignore[invalid-assignment]


def test_the_keys_only_the_monitor_writes_are_every_key_it_adds_but_its_log() -> None:
    # Act
    added = set(MonitorState.__annotations__) - set(AgentState.__annotations__)

    # Assert
    assert added - {"monitor_log"} == MONITOR_STATE_KEYS
