"""A draw's reply is its last AI message, and one that repeats an id is read under a fresh one.

A retry after a block may reuse the blocked call's id, as may a cached or
scripted model. The trajectory refuses an id it holds, and LangGraph merges
messages that share one, so the draw's repeated ids are replaced with fresh
uuid4s in the committed message, the tool message of its draw that answers
it, and the action recorded for it alike.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from hyoshigi import Trajectory
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from langchain_sync_monitors import BlockAndFeedBack, MonitorError, MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_benign_post,
    build_exfiltration_step,
    build_keyword_monitor,
    build_thread_config,
    read_stored_trajectory,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.protocols import RUN_EVERY_STEP


class Report(BaseModel):
    summary: str


class EmptyResponseMiddleware(AgentMiddleware[Any, Any, Any]):
    """Answers every model call inside the monitor with a response that holds no message."""

    def wrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        return ModelResponse(result=[])

    async def awrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        return ModelResponse(result=[])


def test_a_retry_that_reuses_the_blocked_call_s_id_runs_under_a_fresh_one(
    run_mode: RunMode,
) -> None:
    # Arrange: the retry after the block reuses call-post
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(),
            build_benign_post(call_id="call-post"),
            AIMessage("Posted the notes."),
        ],
    )
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.5),
    )
    agent = create_agent(model, tools=workspace.build_tools(), middleware=[middleware])

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert workspace.executed == ["http_post:https://notes.example/team"]
    blocked, retried = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and message.tool_calls
    ]
    retried_call_id = retried.tool_calls[0]["id"]
    assert blocked.tool_calls[0]["id"] == "call-post"
    assert retried_call_id != "call-post"
    assert UUID(retried_call_id).version == 4
    answers = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [answer.tool_call_id for answer in answers] == ["call-post", retried_call_id]
    recorded = Trajectory.load(read_stored_trajectory(result)).records[1]
    assert [call.id for call in recorded.proposals[0].action.tool_calls] == [retried_call_id]


def test_a_structured_answer_that_reuses_a_call_id_keeps_its_tool_message_paired(
    run_mode: RunMode,
) -> None:
    # Arrange: the second run's structured answer reuses the first's call id
    report = build_tool_call_message(
        tool_name="Report",
        call_id="call-report",
        arguments={"summary": "Q3 revenue grew 12%."},
    )
    agent = create_agent(
        ScriptedChatModel(responses=[report, report]),
        response_format=ToolStrategy(Report),
        middleware=[MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP)],
        checkpointer=InMemorySaver(),
    )
    config = build_thread_config(f"structured-{run_mode}")
    run_agent(agent, mode=run_mode, config=config)

    # Act
    result = run_agent(agent, mode=run_mode, config=config, task="Summarise it again.")

    # Assert
    *_, answer, answer_result = result["messages"]
    assert isinstance(answer, AIMessage)
    assert isinstance(answer_result, ToolMessage)
    assert answer.tool_calls[0]["id"] != "call-report"
    assert answer_result.tool_call_id == answer.tool_calls[0]["id"]


def test_a_draw_without_an_ai_message_fails_the_step(run_mode: RunMode) -> None:
    # Arrange: a middleware inside the monitor answers with no message at all
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP),
        EmptyResponseMiddleware(),
    ]
    agent = create_agent(ScriptedChatModel(responses=[]), middleware=stack)

    # Act / Assert
    with pytest.raises(MonitorError, match="no AI message"):
        run_agent(agent, mode=run_mode)
