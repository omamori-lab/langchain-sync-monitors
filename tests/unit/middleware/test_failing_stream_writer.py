"""A stream writer that fails drops the monitor's event and changes nothing else.

The step it was written for stays committed, and a failed step's own error is
still raised. The log line says the event is dropped, which holds for both
events: `monitor_step`, whose step commits, and `monitor_step_failed`, whose
step never does.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    build_keyword_monitor,
    run_agent,
    stream_custom_events,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel
from tests.support.protocols import AcceptFirst

STREAM_WRITER_LOGGER = "langchain_sync_monitors._langchain"
DROPPED_EVENT_TEXT = "The stream writer failed on a monitor event; the event is dropped."
ANSWER = "Q3 revenue grew 12%."


class StreamConsumerGoneError(RuntimeError):
    """What the failing writer raises."""


def fail_to_write(_event: object) -> None:
    message = "the stream consumer went away"
    raise StreamConsumerGoneError(message)


class FailingStreamWriterMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hands the monitor inside it a request whose stream writer always raises."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self.break_stream_writer(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self.break_stream_writer(request))

    def break_stream_writer(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        return replace(request, runtime=replace(request.runtime, stream_writer=fail_to_write))


def build_agent(model: ScriptedChatModel | FlakyChatModel) -> Runnable[Any, Any]:
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    stack: list[AgentMiddleware[Any, Any, Any]] = [FailingStreamWriterMiddleware(), monitor]
    return create_agent(model, middleware=stack)


def read_stream_writer_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == STREAM_WRITER_LOGGER and record.levelno == logging.ERROR
    ]


def test_a_committed_step_stays_committed_when_its_event_is_dropped(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    agent = build_agent(ScriptedChatModel(responses=[AIMessage(ANSWER)]))
    caplog.set_level(logging.ERROR, logger=STREAM_WRITER_LOGGER)

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert [record["outcome"] for record in result["monitor_log"]] == ["allowed"]
    assert result["messages"][-1].text == ANSWER
    assert read_stream_writer_errors(caplog) == [DROPPED_EVENT_TEXT]


def test_a_failed_step_still_raises_its_own_error_when_its_event_is_dropped(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    agent = build_agent(FlakyChatModel(replies=[TimeoutError("the provider timed out")]))
    caplog.set_level(logging.ERROR, logger=STREAM_WRITER_LOGGER)

    # Act
    with pytest.raises(TimeoutError, match="the provider timed out"):
        run_agent(agent, mode=run_mode)

    # Assert
    assert read_stream_writer_errors(caplog) == [DROPPED_EVENT_TEXT]


def test_a_working_stream_writer_logs_no_error(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    monitor = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=AcceptFirst())
    agent = create_agent(ScriptedChatModel(responses=[AIMessage(ANSWER)]), middleware=[monitor])
    caplog.set_level(logging.ERROR, logger=STREAM_WRITER_LOGGER)

    # Act
    events = stream_custom_events(agent, mode=run_mode)

    # Assert
    assert [event["type"] for event in events] == ["monitor_step"]
    assert read_stream_writer_errors(caplog) == []
