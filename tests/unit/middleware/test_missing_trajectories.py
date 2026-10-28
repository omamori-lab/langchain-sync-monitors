"""A thread whose own decided steps outlive its trajectory raises rather than lose them.

A checkpointer that prunes without walking `monitor_trajectory`'s delta chain
rebuilds it empty, and a thread checkpointed before the key existed has
none. Either would let the protocols count from nothing, so a monitor whose
agent has steps with samples in `monitor_log` and no trajectory raises
`MonitorError`. Records of another agent, and halts without a sample, which
record no decision, do not count.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors import MonitorError, MonitorMiddleware, SampleRecord, StepRecord
from tests.support.agents import RunMode, build_keyword_monitor, build_thread_config, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.protocols import RUN_EVERY_STEP

SAMPLE = SampleRecord(
    suspicion=0.1,
    reason="nothing suspicious",
    proposal={
        "kind": "action",
        "id": "step-1",
        "thought": [],
        "parts": [{"kind": "text", "text": "Done."}],
        "tool_calls": [],
    },
    executed=True,
)
OWN_STEP = StepRecord(
    agent="main",
    monitor="monitor",
    step_number=1,
    outcome="allowed",
    flagged=False,
    blocked_count=0,
    samples=[SAMPLE],
)
SUBAGENT_STEP = StepRecord(
    agent="worker",
    monitor="monitor",
    step_number=1,
    outcome="allowed",
    flagged=False,
    blocked_count=0,
    samples=[SAMPLE],
    delegation_id="call-task",
)
OWN_HALT = StepRecord(
    agent="main",
    monitor="monitor",
    step_number=1,
    outcome="halted",
    flagged=True,
    blocked_count=0,
    samples=[],
)


def build_thread_with_log(record: StepRecord, *, thread_id: str) -> tuple[Any, RunnableConfig]:
    """Return an agent and the config of a thread whose log holds the record and no trajectory."""
    agent = create_agent(
        ScriptedChatModel(responses=[AIMessage("Again.")]),
        middleware=[MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP)],
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(thread_id)
    update = {
        "messages": [HumanMessage("Summarise q3.md."), AIMessage("Done.", id="step-1")],
        "monitor_log": [record],
    }
    agent.update_state(config, update, as_node="model")
    return agent, config


def test_a_thread_whose_own_steps_outlive_its_trajectory_raises(run_mode: RunMode) -> None:
    # Arrange
    agent, config = build_thread_with_log(OWN_STEP, thread_id=f"own-step-{run_mode}")

    # Act / Assert
    with pytest.raises(MonitorError, match="monitor_trajectory"):
        run_agent(agent, mode=run_mode, config=config, task="Summarise it again.")


@pytest.mark.parametrize(
    ("record", "outcome"),
    [(SUBAGENT_STEP, "allowed"), (OWN_HALT, "halted")],
    ids=["a-subagent-s-step", "an-own-halt-without-a-sample"],
)
def test_records_that_hold_no_own_decision_need_no_trajectory(
    run_mode: RunMode,
    record: StepRecord,
    outcome: str,
) -> None:
    # Arrange
    agent, config = build_thread_with_log(record, thread_id=f"no-decision-{run_mode}")

    # Act
    result = run_agent(agent, mode=run_mode, config=config, task="Summarise it again.")

    # Assert: an own halt stands, since the log alone records no run input at it
    assert result["monitor_log"][-1]["outcome"] == outcome
