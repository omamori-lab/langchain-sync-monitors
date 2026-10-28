"""A draw's reply is its last AI message, and one that repeats an id is read under a fresh one.

A retry after a block may reuse the blocked call's id, as may a cached or
scripted model, and a reply may give two of its calls one id. The trajectory
refuses an id it holds, and LangGraph merges messages that share one, so the
draw's repeated ids are replaced with fresh uuid4s in the committed message,
the tool message of its draw that answers it, and the action recorded for it
alike: the first call with an id keeps it unless the run holds it, and each
later call with it takes a fresh one.
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


READS_UNDER_ONE_ID = AIMessage(
    content="",
    tool_calls=[
        {"name": "read_file", "args": {"path": "q3.md"}, "id": "call-dup", "type": "tool_call"},
        {"name": "read_file", "args": {"path": "q2.md"}, "id": "call-dup", "type": "tool_call"},
    ],
)
"""A reply whose two calls share one id."""

TOOL_USE_BLOCKS_UNDER_ONE_ID = AIMessage(
    content=[
        {"type": "tool_use", "id": "call-dup", "name": "read_file", "input": {"path": "q3.md"}},
        {"type": "tool_use", "id": "call-dup", "name": "read_file", "input": {"path": "q2.md"}},
    ],
    tool_calls=[
        {"name": "read_file", "args": {"path": "q3.md"}, "id": "call-dup", "type": "tool_call"},
        {"name": "read_file", "args": {"path": "q2.md"}, "id": "call-dup", "type": "tool_call"},
    ],
)
"""A reply in Anthropic's shape, whose `tool_use` blocks repeat its two calls under one id."""


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


@pytest.mark.parametrize(
    ("responses", "keeps_first_id"),
    [
        ([READS_UNDER_ONE_ID, AIMessage("Done.")], True),
        (
            [
                build_exfiltration_step(call_id="call-dup"),
                READS_UNDER_ONE_ID,
                AIMessage("Done."),
            ],
            False,
        ),
    ],
    ids=["in-the-first-reply", "in-a-retry-after-a-block-under-the-blocked-id"],
)
def test_two_calls_of_a_reply_that_share_an_id_each_run_under_an_id_of_their_own(
    run_mode: RunMode,
    responses: list[AIMessage],
    keeps_first_id: bool,
) -> None:
    # Arrange
    workspace = Workspace()
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.5),
    )
    agent = create_agent(
        ScriptedChatModel(responses=responses),
        tools=workspace.build_tools(),
        middleware=[middleware],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert sorted(workspace.executed) == ["read_file:q2.md", "read_file:q3.md"]
    [reads] = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and len(message.tool_calls) == 2
    ]
    call_ids = [call["id"] for call in reads.tool_calls]
    assert len(set(call_ids)) == 2
    assert ("call-dup" in call_ids) is keeps_first_id
    answered = [
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.status == "success"
    ]
    assert answered == call_ids


def test_tool_use_blocks_that_share_an_id_take_their_calls_ids_in_order(
    run_mode: RunMode,
) -> None:
    # Arrange
    workspace = Workspace()
    middleware = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.5),
    )
    agent = create_agent(
        ScriptedChatModel(responses=[TOOL_USE_BLOCKS_UNDER_ONE_ID, AIMessage("Done.")]),
        tools=workspace.build_tools(),
        middleware=[middleware],
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    [reads] = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and len(message.tool_calls) == 2
    ]
    block_ids = [block["id"] for block in reads.content if isinstance(block, dict)]
    call_ids = [call["id"] for call in reads.tool_calls]
    assert block_ids == call_ids
    assert len(set(block_ids)) == 2
