"""A subagent's halt stands for its own delegation only.

The count of run inputs that a halt stores is private state, so a subagent
starts every delegation without its parent's, and its own never flows back.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    build_task_input,
    build_thread_config,
    run_stopping_audits,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import HALT_EVERY_STEP, RUN_EVERY_STEP


def test_a_subagent_halted_in_one_delegation_is_sampled_in_the_next(run_mode: RunMode) -> None:
    # Arrange: the worker's monitor audits every step, each stopped, and the parent delegates
    # twice
    main_model = ScriptedChatModel(
        responses=[
            build_delegation_step(call_id="call-task-1"),
            build_delegation_step(call_id="call-task-2"),
            AIMessage("Here is the summary."),
        ],
    )
    worker_model = ScriptedChatModel(
        responses=[AIMessage("First report."), AIMessage("Second report.")],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=RUN_EVERY_STEP),
        worker_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=HALT_EVERY_STEP),
        checkpointer=InMemorySaver(),
    )

    # Act
    result = run_stopping_audits(
        agent,
        build_task_input(),
        mode=run_mode,
        config=build_thread_config("halted-twice"),
    )

    # Assert: each delegation drew a sample of its own
    worker_records = [record for record in result["monitor_log"] if record["agent"] == "worker"]
    assert len(worker_model.calls) == 2
    assert [(record["outcome"], len(record["samples"])) for record in worker_records] == [
        ("halted", 1),
        ("halted", 1),
    ]
    assert [record["delegation_id"] for record in worker_records] == [
        "call-task-1",
        "call-task-2",
    ]
