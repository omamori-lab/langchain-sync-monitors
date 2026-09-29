"""A subagent's steps are spans under the `task` call that started it, named by that call's id.

Deep Agents runs a subagent inside its `task` tool, so a tracer nests the
subagent's step spans under the call, and every monitor span of a subagent
step carries the call's id as `monitor_delegation_id`. A main agent's spans
carry no delegation id.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import SubagentHalt
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst, HaltAfterOneSample
from tests.support.tracing import RecordedRun, run_traced_agent

WORKER_REPORT = "Three sources found."
MAIN_ANSWER = "Here is the summary."
DELEGATION_ID = "call-task"


@pytest.fixture
def monitor() -> KeywordMonitor:
    return KeywordMonitor()


def read_step_names(steps: list[RecordedRun]) -> list[tuple[str, int, str | None]]:
    return [
        (
            step.metadata["monitor_agent"],
            step.metadata["monitor_step_number"],
            step.metadata.get("monitor_delegation_id"),
        )
        for step in steps
    ]


def test_a_subagent_s_spans_nest_under_its_task_call_and_carry_its_delegation_id(
    run_mode: RunMode,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    agent = build_deep_agent(
        main_model=ScriptedChatModel(responses=[build_delegation_step(), AIMessage(MAIN_ANSWER)]),
        worker_model=ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)]),
        main_monitor=MonitorMiddleware(monitor=monitor, protocol=AcceptFirst()),
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    steps = tracer.find_runs("monitor step")
    assert read_step_names(steps) == [
        ("main", 1, None),
        ("worker", 1, DELEGATION_ID),
        ("main", 2, None),
    ]
    worker_step = steps[1]
    assert "task" in tracer.find_ancestor_names(worker_step)
    assert "task" not in tracer.find_ancestor_names(steps[0])
    worker_spans = [child for child in worker_step.children if child.is_monitor_span]
    assert [span.name for span in worker_spans] == ["monitor judgement", "monitor decision"]
    assert all(span.metadata["monitor_delegation_id"] == DELEGATION_ID for span in worker_spans)
    assert all(span.metadata["monitor_agent"] == "worker" for span in worker_spans)
    assert tracer.find_unknown_parents() == []
    assert tracer.find_open_runs() == []


def test_a_halt_after_a_subagent_halt_is_a_step_span_with_only_its_decision(
    run_mode: RunMode,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    agent = build_deep_agent(
        main_model=ScriptedChatModel(responses=[build_delegation_step()]),
        worker_model=ScriptedChatModel(responses=[AIMessage(WORKER_REPORT)]),
        main_monitor=MonitorMiddleware(
            monitor=monitor,
            protocol=AcceptFirst(),
            when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN,
        ),
        worker_monitor=MonitorMiddleware(monitor=monitor, protocol=HaltAfterOneSample()),
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    steps = tracer.find_runs("monitor step")
    assert read_step_names(steps) == [
        ("main", 1, None),
        ("worker", 1, DELEGATION_ID),
        ("main", 2, None),
    ]
    halt = steps[-1]
    [decision] = halt.children
    assert decision.name == "monitor decision"
    assert halt.inputs == {"step_number": 2, "proposed_step": None}
    assert (halt.outputs["outcome"], halt.outputs["max_suspicion"], halt.outputs["samples"]) == (
        "halted",
        None,
        [],
    )
    assert decision.tags == ["monitor", "monitor:halted", "monitor:flagged"]
    assert "monitor_max_suspicion" not in decision.metadata
    assert all(step.error is None for step in steps)
