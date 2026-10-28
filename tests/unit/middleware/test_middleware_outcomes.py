"""How a step's draws run under each entry point, and what a committed step logs.

`test_step_commits` checks what each decision commits; these tests check
that resamples drawn at once overlap only under `ainvoke()`, that `invoke()`
decides inside a running event loop and outside a graph, and the debug line
each committed step logs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sync_monitors import BlockAndFeedBack, DeferToResample, Resampling
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_task_input,
    read_texts,
    run_agent,
)
from tests.support.chat_models import OverlapCountingChatModel, ScriptedChatModel
from tests.support.protocols import RUN_EVERY_STEP

FINAL_ANSWER = "Q3 revenue grew 12%."
MIDDLEWARE_LOGGER = "langchain_sync_monitors.middleware"


@pytest.mark.parametrize(("run_mode", "most_at_once"), [("invoke", 1), ("ainvoke", 2)])
def test_resamples_drawn_at_once_overlap_only_under_ainvoke(
    run_mode: RunMode,
    most_at_once: int,
) -> None:
    # Arrange: a suspicious step and two benign resamples, each call sleeping under ainvoke
    workspace = Workspace()
    responses = [build_exfiltration_step(), build_read_step(call_id="call-read-a")]
    responses += [build_read_step(call_id="call-read-b"), AIMessage(FINAL_ANSWER)]
    untrusted_model = OverlapCountingChatModel(responses=responses, delay=0.01)
    protocol = DeferToResample(
        defer_threshold=0.5,
        audit_threshold=1.0,
        max_resamples=2,
        resampling=Resampling.PARALLEL,
    )
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    agent = create_agent(untrusted_model, tools=workspace.build_tools(), middleware=[middleware])

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert untrusted_model.most_at_once == most_at_once
    assert workspace.executed == ["read_file:q3.md"]
    first = result["monitor_log"][0]
    assert len(first["samples"]) == 3
    assert sum(sample["executed"] for sample in first["samples"]) == 1
    assert read_texts(result["messages"])[-1] == FINAL_ANSWER


async def test_invoke_works_inside_a_running_event_loop() -> None:
    # Arrange
    model = ScriptedChatModel(responses=[AIMessage(FINAL_ANSWER)])
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP)
    agent = create_agent(model, middleware=[middleware])

    # Act
    result = agent.invoke(build_task_input())

    # Assert
    assert read_texts(result["messages"])[-1] == FINAL_ANSWER
    assert len(result["monitor_log"]) == 1


def decide_outside_a_graph(
    middleware: MonitorMiddleware,
    request: ModelRequest[Any],
    *,
    mode: RunMode,
) -> ExtendedModelResponse[Any]:
    """Run the hook `mode` calls on a request built by hand, as no graph is running."""
    if mode == "invoke":
        return middleware.wrap_model_call(
            request,
            lambda inner: ModelResponse(result=[inner.model.invoke(inner.messages)]),
        )

    async def handle(inner: ModelRequest[Any]) -> ModelResponse[Any]:
        return ModelResponse(result=[await inner.model.ainvoke(inner.messages)])

    return asyncio.run(middleware.awrap_model_call(request, handle))


def test_a_request_outside_a_graph_commits_without_a_stream_writer(run_mode: RunMode) -> None:
    # Arrange
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP)
    request: ModelRequest[Any] = ModelRequest(
        model=ScriptedChatModel(responses=[AIMessage(FINAL_ANSWER)]),
        messages=[HumanMessage("Summarise.")],
    )

    # Act
    result = decide_outside_a_graph(middleware, request, mode=run_mode)

    # Assert
    assert isinstance(result, ExtendedModelResponse)
    assert read_texts(result.model_response.result) == [FINAL_ANSWER]
    assert result.command is not None
    assert isinstance(result.command.update, dict)
    [record] = result.command.update["monitor_log"]
    assert record["step_number"] == 1


def test_each_committed_step_logs_its_outcome_at_debug_level(
    run_mode: RunMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the block and feed back run that the how-to Read the monitor log shows
    caplog.set_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER)
    untrusted_model = ScriptedChatModel(
        responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
    )
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.6),
    )
    agent = create_agent(untrusted_model, tools=Workspace().build_tools(), middleware=[middleware])

    # Act
    run_agent(agent, mode=run_mode)

    # Assert: the lines the how-to quotes, word for word
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == MIDDLEWARE_LOGGER and record.levelno == logging.DEBUG
    ] == ["monitor[main] committed step 1: steered", "monitor[main] committed step 2: allowed"]
