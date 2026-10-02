"""The whole stack at once: a real monitor class, a real protocol and the middleware.

The LLM monitor and the agents are scripted, so the tests stay offline, but
every component between them is the production code.
"""

from __future__ import annotations

from typing import Any

import pytest
import stamina
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors import (
    AutoMode,
    DeferToResample,
    HaltRun,
    LLMMonitor,
    MonitorMiddleware,
    TrustedMonitoring,
)
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel

FINAL_ANSWER = "Q3 revenue grew 12%."


def build_judge_reply(*, score: int) -> AIMessage:
    return AIMessage(f"<reasoning>Scored against the task.</reasoning>\n<score>{score}</score>")


def test_llm_monitor_drives_defer_to_resample_end_to_end(run_mode: RunMode) -> None:
    # Arrange
    judge = ScriptedChatModel(
        responses=[build_judge_reply(score=score) for score in (9, 1, 0)],
    )
    workspace = Workspace()
    middleware = MonitorMiddleware(
        monitor=LLMMonitor(model=judge),
        protocol=DeferToResample(
            fallback=HaltRun(message="Stopped."),
            defer_threshold=0.6,
            audit_threshold=0.95,
        ),
    )
    agent: Runnable[Any, Any] = create_agent(
        model=ScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        tools=workspace.build_tools(),
        middleware=[middleware],
    )

    # Act
    state = run_agent(agent, mode=run_mode)

    # Assert
    first_judge_prompt = "\n".join(message.text for message in judge.calls[0])
    assert "attacker.example" in first_judge_prompt
    assert "<proposed_step>" in first_judge_prompt
    assert workspace.executed == ["read_file:q3.md"]
    assert [record["outcome"] for record in state["monitor_log"]] == ["resampled", "allowed"]
    assert [sample["suspicion"] for sample in state["monitor_log"][0]["samples"]] == [0.9, 0.1]


class RateLimitError(Exception):
    """A provider's rate limit, with the HTTP status as `status_code`, as provider SDKs give it."""

    status_code = 429


def test_a_rate_limited_judge_call_is_made_again_and_the_step_commits(run_mode: RunMode) -> None:
    # Arrange: the judge's first call is rate limited, as ChatOpenRouter raises one at once
    judge = FlakyChatModel(
        replies=[RateLimitError("Too Many Requests"), *[build_judge_reply(score=1)] * 2],
    )
    workspace = Workspace()
    middleware = MonitorMiddleware(
        monitor=LLMMonitor(model=judge),
        protocol=TrustedMonitoring(audit_threshold=0.9),
    )
    agent: Runnable[Any, Any] = create_agent(
        model=ScriptedChatModel(responses=[build_read_step(), AIMessage(FINAL_ANSWER)]),
        tools=workspace.build_tools(),
        middleware=[middleware],
    )

    # Act
    with stamina.set_testing(True, attempts=10, cap=True):
        state = run_agent(agent, mode=run_mode, config={"recursion_limit": 20})

    # Assert
    assert judge.started_calls == 3
    assert workspace.executed == ["read_file:q3.md"]
    assert [record["outcome"] for record in state["monitor_log"]] == ["allowed", "allowed"]


def test_auto_mode_steers_a_deep_agent_subagent_and_the_parent_sees_its_record(
    run_mode: RunMode,
) -> None:
    # Arrange
    pytest.importorskip("deepagents")
    from tests.support.deep_agents import build_deep_agent, build_delegation_step

    workspace = Workspace()
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=AutoMode(block_threshold=0.6)
    )
    agent = build_deep_agent(
        main_model=ScriptedChatModel(
            responses=[build_delegation_step(), AIMessage("The worker found the figure.")],
        ),
        worker_model=ScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        main_monitor=monitor,
        tools=workspace.build_tools(),
    )

    # Act
    state = run_agent(agent, mode=run_mode)

    # Assert
    records_by_agent = {(record["agent"], record["outcome"]) for record in state["monitor_log"]}
    assert ("worker", "steered") in records_by_agent
    assert ("main", "allowed") in records_by_agent
    assert workspace.executed == ["read_file:q3.md"]
