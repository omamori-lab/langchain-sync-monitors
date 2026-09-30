"""A subagent's halt stands for its own delegation only.

The count of run inputs that a halt stores is private state, so a subagent
starts every delegation without its parent's, and its own never flows back.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample


def test_a_subagent_halted_in_one_delegation_is_sampled_in_the_next(run_mode: RunMode) -> None:
    # Arrange: the worker's monitor halts every step, and the parent delegates twice
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
        main_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst()),
        worker_monitor=MonitorMiddleware(monitor=KeywordMonitor(), protocol=HaltAfterOneSample()),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

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
