"""Committed steps stream as custom events; the monitor's own model calls stay internal."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.internal_call_transformer import INTERNAL_CALL_METADATA_KEY
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    SUSPICION_BY_KEYWORD,
    RunMode,
    Workspace,
    build_read_step,
    build_task_input,
    stream_custom_events,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import JUDGE_SOURCE, KeywordMonitor
from tests.support.protocols import AcceptFirst

JUDGE_TEXT = "judge verdict text"
AGENT_ANSWER = "Q3 revenue grew 12%."


@pytest.fixture
def judged_middleware() -> MonitorMiddleware:
    judge_model = GenericFakeChatModel(messages=iter([AIMessage(JUDGE_TEXT)] * 4))
    monitor = KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_KEYWORD, judge_model=judge_model)
    return MonitorMiddleware(monitor=monitor, protocol=AcceptFirst())


def test_each_committed_step_is_streamed_as_a_custom_event(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_read_step(), AIMessage(AGENT_ANSWER)])
    middleware = MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())
    agent = create_agent(model, tools=Workspace().build_tools(), middleware=[middleware])

    # Act
    events = stream_custom_events(agent, mode=run_mode)

    # Assert
    assert [event["type"] for event in events] == ["monitor_step", "monitor_step"]
    assert [event["record"]["step_number"] for event in events] == [1, 2]
    assert all(event["record"]["agent"] == "main" for event in events)


def stream_message_metadata(agent: Runnable[Any, Any], *, mode: RunMode) -> list[dict[str, Any]]:
    payload = build_task_input()
    if mode == "invoke":
        return [metadata for _chunk, metadata in agent.stream(payload, stream_mode="messages")]

    async def collect() -> list[dict[str, Any]]:
        stream = agent.astream(payload, stream_mode="messages")
        return [metadata async for _chunk, metadata in stream]

    return asyncio.run(collect())


def test_monitor_calls_in_the_messages_stream_carry_the_internal_call_tag(
    run_mode: RunMode,
    judged_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[AIMessage(AGENT_ANSWER)])
    agent = create_agent(model, middleware=[judged_middleware])

    # Act
    chunks = stream_message_metadata(agent, mode=run_mode)

    # Assert
    judge_chunks = [metadata for metadata in chunks if metadata.get("lc_source") == JUDGE_SOURCE]
    agent_chunks = [metadata for metadata in chunks if metadata.get("lc_source") is None]
    assert judge_chunks
    assert agent_chunks
    assert all(INTERNAL_CALL_METADATA_KEY in metadata for metadata in judge_chunks)
    assert all(INTERNAL_CALL_METADATA_KEY not in metadata for metadata in agent_chunks)


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
def test_monitor_calls_are_dropped_from_the_run_messages_projection(
    judged_middleware: MonitorMiddleware,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[AIMessage(AGENT_ANSWER)])
    agent = create_agent(model, middleware=[judged_middleware])

    # Act
    run = agent.stream_events(build_task_input(), version="v3")
    texts = [str(stream.output.content) for stream in run.messages]

    # Assert
    assert any(AGENT_ANSWER in text for text in texts)
    assert all(JUDGE_TEXT not in text for text in texts)
