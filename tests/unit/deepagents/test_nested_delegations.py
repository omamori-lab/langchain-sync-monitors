"""A subagent that delegates again hands its own subagent the run's total, whatever the ids."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import BlockAndFeedBack
from tests.support.agents import (
    RunMode,
    Workspace,
    build_benign_post,
    build_exfiltration_step,
    build_keyword_monitor,
    build_task_input,
    build_thread_config,
    run_stopping_audits,
    summarise_records,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step

MAX_TOTAL_BLOCKS = 5


def build_nested_agent(
    *,
    main_model: ScriptedChatModel,
    middle_model: ScriptedChatModel,
    inner_model: ScriptedChatModel,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build main, which delegates to `middle`, a deep agent that delegates to `inner`.

    Main keeps a checkpointer, which its subagents share, so an audit can pause.
    """
    main_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.6, blocks_in_total=MAX_TOTAL_BLOCKS),
    )
    middle_monitor = main_monitor.copy_for_subagent(subagent_name="middle")
    middle = create_deep_agent(
        model=middle_model,
        tools=Workspace().build_http_tools(),
        middleware=[middle_monitor],
        subagents=monitor_subagents(
            middleware=middle_monitor,
            subagents=[SubAgent(name="inner", description="Checks.", model=inner_model)],
        ),
    )
    return create_deep_agent(
        model=main_model,
        tools=Workspace().build_http_tools(),
        middleware=[main_monitor],
        subagents=[
            CompiledSubAgent(name="middle", description="Delegates again.", runnable=middle),
        ],
        checkpointer=InMemorySaver(),
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
    result = run_stopping_audits(
        agent,
        build_task_input(),
        mode=run_mode,
        config=build_thread_config("nested"),
    )

    # Assert: the inner's third block is the run's fifth, so its next proposal is audited and
    # stopped; the audit resets the total, so middle and main run on.
    assert summarise_records(result["monitor_log"]) == [
        ("main", "steered", 1, None),
        ("middle", "steered", 1, main_task_call_id),
        ("inner", "steered", 1, middle_task_call_id),
        ("inner", "steered", 1, middle_task_call_id),
        ("inner", "halted", 1, middle_task_call_id),
        ("middle", "allowed", 0, main_task_call_id),
        ("main", "allowed", 0, None),
    ]
    assert sum(record["blocked_count"] for record in result["monitor_log"]) == MAX_TOTAL_BLOCKS
    assert len(inner_model.calls) == 6
    assert "monitor_delegation" not in result
