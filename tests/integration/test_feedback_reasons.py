"""The reason Auto Mode shows the agent carries no scores, however the monitor is wrapped.

Auto Mode sends the monitor's reason to the agent word for word. A score in it
would tell an attacker how close its step came to the threshold, so the
wrappers keep numbers in the record, for the auditor, and out of the reason.
"""

from __future__ import annotations

import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitors import CalibratedMonitor, CascadeMonitor, RepeatedMonitor
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel


def build_wrapped_monitor() -> CalibratedMonitor:
    cascade = CascadeMonitor(
        fast=build_keyword_monitor(),
        careful=build_keyword_monitor(),
        escalate_at=0.5,
    )
    return CalibratedMonitor(
        monitor=RepeatedMonitor(monitor=cascade, repeats=2),
        honest_scores=[0.2] * 20,
        random_seed=1,
    )


def test_the_feedback_on_a_block_carries_the_reason_without_numbers(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )
    middleware = MonitorMiddleware(
        monitor=build_wrapped_monitor(),
        protocol=AutoMode(block_threshold=0.9),
    )
    agent = create_agent(model, tools=Workspace().build_tools(), middleware=[middleware])

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the agent reads the reason, and only the record holds the score
    [feedback] = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert "Reason: mentions attacker.example" in feedback.text
    assert not re.search(r"\d", feedback.text)
    first_record = result["monitor_log"][0]
    assert first_record["outcome"] == "steered"
    assert first_record["samples"][0]["suspicion"] > 0.9
    assert first_record["samples"][0]["reason"] == "mentions attacker.example"
