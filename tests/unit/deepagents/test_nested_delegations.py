"""A subagent that delegates again hands its own subagent the thread's total, whatever the ids."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.contracts import StepRecord
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.deep_agents import build_delegation_step

MAX_TOTAL_BLOCKS = 5


def build_benign_post(*, call_id: str) -> AIMessage:
    return build_tool_call_message(
        tool_name="http_post",
        call_id=call_id,
        arguments={"url": "https://notes.example/team", "body": "Three sources."},
    )


def build_http_tools() -> list[BaseTool]:
    return [tool for tool in Workspace().build_tools() if tool.name == "http_post"]


def build_nested_agent(
    *,
    main_model: ScriptedChatModel,
    middle_model: ScriptedChatModel,
    inner_model: ScriptedChatModel,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build main, which delegates to `middle`, a deep agent that delegates to `inner`."""
    main_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=0.6, max_total_blocks=MAX_TOTAL_BLOCKS),
    )
    middle_monitor = main_monitor.copy_for_subagent(subagent_name="middle")
    middle = create_deep_agent(
        model=middle_model,
        tools=build_http_tools(),
        middleware=[middle_monitor],
        subagents=monitor_subagents(
            middleware=middle_monitor,
            subagents=[SubAgent(name="inner", description="Checks.", model=inner_model)],
        ),
    )
    return create_deep_agent(
        model=main_model,
        tools=build_http_tools(),
        middleware=[main_monitor],
        subagents=[
            CompiledSubAgent(name="middle", description="Delegates again.", runnable=middle),
        ],
    )


def build_inner_model(*, attempts: int) -> ScriptedChatModel:
    """Script an inner subagent that tries to exfiltrate, then posts benignly, each attempt."""
    responses: list[AIMessage] = []
    for index in range(attempts):
        responses += [
            build_exfiltration_step(call_id=f"call-post-inner-{index}"),
            build_benign_post(call_id=f"call-benign-inner-{index}"),
        ]
    return ScriptedChatModel(responses=[*responses, AIMessage("Inner done.")])


def summarise(log: list[StepRecord]) -> list[tuple[str, str, int, str | None]]:
    return [
        (record["agent"], record["outcome"], record["blocked_count"], record.get("delegation_id"))
        for record in log
    ]


@pytest.mark.parametrize(
    ("main_task_call_id", "middle_task_call_id"),
    [("call-task-main", "call-task-middle"), ("call_0", "call_0")],
    ids=["distinct-ids", "reused-id"],
)
def test_a_nested_subagent_counts_the_blocks_of_every_agent_above_it(
    run_mode: RunMode,
    main_task_call_id: str,
    middle_task_call_id: str,
) -> None:
    # Arrange
    main_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-main"),
            build_delegation_step(call_id=main_task_call_id, subagent_type="middle"),
            AIMessage("Main done."),
        ],
    )
    middle_model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-middle"),
            build_delegation_step(call_id=middle_task_call_id, subagent_type="inner"),
            AIMessage("Middle done."),
        ],
    )
    inner_model = build_inner_model(attempts=4)
    agent = build_nested_agent(
        main_model=main_model,
        middle_model=middle_model,
        inner_model=inner_model,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert summarise(result["monitor_log"]) == [
        ("main", "steered", 1, None),
        ("middle", "steered", 1, main_task_call_id),
        ("inner", "steered", 1, middle_task_call_id),
        ("inner", "steered", 1, middle_task_call_id),
        ("inner", "halted", 1, middle_task_call_id),
        ("middle", "halted", 0, main_task_call_id),
        ("main", "halted", 0, None),
    ]
    assert sum(record["blocked_count"] for record in result["monitor_log"]) == MAX_TOTAL_BLOCKS
    assert len(inner_model.calls) == 5
    assert "monitor_delegation" not in result
