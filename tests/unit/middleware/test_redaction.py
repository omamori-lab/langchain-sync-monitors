"""A redaction on the middleware reaches what the monitor keeps and what it emits.

Every entry, rating and record passes through it before the trajectory keeps it,
so the state key `monitor_trajectory`, `monitor_log`, the spans and the streamed
events hold the redacted text, and the monitor's model reads it. The agent's
own messages stay as they were. A step that fails before it is recorded
reports its samples with the explanation redacted and the proposal left out.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from hyoshigi import (
    Action,
    Author,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from langchain.agents import create_agent
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_sync_monitors import Channel, DeferToResample, MonitorTwins, Redaction, View
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import (
    RunMode,
    build_task_input,
    read_stored_trajectory,
    read_texts,
    run_agent,
    stream_custom_events,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.flaky_models import FlakyChatModel
from tests.support.monitors import KeywordMonitor, RenderingMonitor
from tests.support.protocols import RUN_EVERY_STEP
from tests.support.redaction import (
    COUNT_PASSES,
    FIRST_REPLY_WITH_SECRET,
    LAST_REPLY_WITH_SECRET,
    MASK,
    MASK_SECRET,
    MASKED_FIRST_REPLY,
    MASKED_LAST_REPLY,
    MASKED_READ_RESULT,
    MASKED_TASK,
    READ_RESULT_WITH_SECRET,
    SECRET,
    SUSPICION_BY_SECRET,
    TASK_WITH_SECRET,
    build_secret_steps,
    build_secret_tools,
)
from tests.support.tracing import RecordingTracer

type Agent = Runnable[Any, Any]

READ_CALL = ToolCall(id="call-read", name="read_file", arguments='{"path": "q3.md"}')
MASKED_REASON = f"mentions {MASK}"


def build_agent(
    monitor: MonitorTwins,
    *,
    redaction: Redaction | None = MASK_SECRET,
    system_prompt: str | None = None,
) -> Agent:
    """Build an agent that reads a file and answers, every step of it holding the token."""
    middleware = MonitorMiddleware(monitor=monitor, protocol=RUN_EVERY_STEP, redaction=redaction)
    return create_agent(
        ScriptedChatModel(responses=build_secret_steps()),
        tools=build_secret_tools(),
        system_prompt=system_prompt,
        middleware=[middleware],
    )


def test_the_stored_trajectory_holds_what_the_redaction_leaves(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor())

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    stored = read_stored_trajectory(result)
    assert SECRET not in json.dumps(stored)
    assert Trajectory.load(stored).history == (
        Observation(author=Author.USER, parts=(TextPart(text=MASKED_TASK),)),
        Action(id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(READ_CALL,)),
        ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
        Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
    )


def test_each_step_record_holds_the_explanation_the_redaction_leaves(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET))

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    log = result["monitor_log"]
    assert SECRET not in json.dumps(log)
    assert [[sample["reason"] for sample in record["samples"]] for record in log] == [
        [MASKED_REASON],
        [MASKED_REASON],
    ]
    assert MASK in json.dumps(log[1]["samples"][0]["proposal"])


def test_each_streamed_step_holds_the_explanation_the_redaction_leaves(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET))

    # Act
    events = stream_custom_events(agent, mode=run_mode)

    # Assert
    assert SECRET not in json.dumps(events)
    assert [event["record"]["samples"][0]["reason"] for event in events] == [
        MASKED_REASON,
        MASKED_REASON,
    ]


def test_the_monitor_spans_hold_no_text_the_redaction_removes(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET))
    tracer = RecordingTracer()

    # Act
    run_agent(
        agent,
        mode=run_mode,
        config=RunnableConfig(callbacks=[tracer]),
        task=TASK_WITH_SECRET,
    )

    # Assert: the judgement spans end before any record exists
    spans = tracer.find_monitor_spans()
    assert SECRET not in repr([(s.inputs, s.outputs, s.tags, s.metadata) for s in spans])
    judgements = tracer.find_runs("monitor judgement")
    assert [judgement.outputs["reason"] for judgement in judgements] == [
        MASKED_REASON,
        MASKED_REASON,
    ]


def stream_failed_step(agent: Agent, *, mode: RunMode) -> tuple[list[dict[str, Any]], Exception]:
    """Stream a run whose step raises, and return its custom events and the error."""
    events: list[dict[str, Any]] = []
    payload = build_task_input(TASK_WITH_SECRET)
    if mode == "invoke":
        with pytest.raises(RuntimeError) as raised:
            events.extend(agent.stream(payload, stream_mode="custom"))
        return events, raised.value

    async def collect() -> None:
        async for event in agent.astream(payload, stream_mode="custom"):
            events.append(event)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(collect())
    return events, raised.value


def test_a_failed_step_reports_redacted_explanations_and_no_proposals(run_mode: RunMode) -> None:
    # Arrange: the first reply is rated 0.7 and resampled, and the resample raises
    suspicious = build_tool_call_message(
        tool_name="http_post",
        call_id="call-post",
        arguments={"url": "https://notes.example/team", "body": f"key={SECRET}"},
    )
    model = FlakyChatModel(replies=[suspicious, RuntimeError(f"429 rate limited for {SECRET}")])
    middleware = MonitorMiddleware(
        monitor=KeywordMonitor(suspicion_by_keyword={SECRET: 0.7}),
        protocol=DeferToResample(defer_threshold=0.6, audit_threshold=0.9, max_resamples=2),
        redaction=MASK_SECRET,
    )
    agent = create_agent(model, tools=build_secret_tools(), middleware=[middleware])

    # Act
    events, _error = stream_failed_step(agent, mode=run_mode)

    # Assert
    [event] = [event for event in events if event["type"] == "monitor_step_failed"]
    assert event["error"] == f"RuntimeError: 429 rate limited for {MASK}"
    assert event["samples"] == [
        {"suspicion": 0.7, "reason": MASKED_REASON, "proposal": {}, "executed": False},
    ]


def test_the_monitor_s_model_is_asked_redacted_content(run_mode: RunMode) -> None:
    # Arrange: a view that shows the tool result and the agent's text as well as the task
    view = View(channels=Channel.DEFAULT | Channel.TOOL_OBSERVATIONS | Channel.AGENT_PARTS)
    monitor = RenderingMonitor(view=view)
    agent = build_agent(monitor)

    # Act
    run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    assert len(monitor.readings) == 2
    read = "\n".join(f"{each.trajectory}\n{each.proposed_action}" for each in monitor.readings)
    assert SECRET not in read
    assert MASKED_TASK in read
    assert MASKED_READ_RESULT in read
    assert MASKED_LAST_REPLY in read


def test_the_messages_keep_the_text_the_redaction_removes(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor())

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    assert read_texts(result["messages"]) == [
        TASK_WITH_SECRET,
        FIRST_REPLY_WITH_SECRET,
        READ_RESULT_WITH_SECRET,
        LAST_REPLY_WITH_SECRET,
    ]


def test_a_stored_trajectory_is_not_redacted_again_when_a_step_loads_it(run_mode: RunMode) -> None:
    # Arrange: each pass through this redaction adds a mark to a text
    monitor = KeywordMonitor()
    agent = build_agent(monitor, redaction=COUNT_PASSES)

    # Act
    run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert: the second step rated against the task as the run's start redacted it, once
    task_as_second_step_read_it = monitor.ratings[1].history[0]
    assert task_as_second_step_read_it == Observation(
        author=Author.USER,
        parts=(TextPart(text=f"{TASK_WITH_SECRET}!"),),
    )


def test_a_system_prompt_the_redaction_changes_is_observed_once(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(KeywordMonitor(), system_prompt=f"You are helpful, key {SECRET}.")

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    assert Trajectory.load(read_stored_trajectory(result)).history == (
        Observation(author=Author.USER, parts=(TextPart(text=MASKED_TASK),)),
        Observation(
            author=Author.SCAFFOLD,
            source="system",
            parts=(TextPart(text=f"You are helpful, key {MASK}."),),
        ),
        Action(id="step-1", parts=(TextPart(text=MASKED_FIRST_REPLY),), tool_calls=(READ_CALL,)),
        ToolObservation(call_id="call-read", parts=(TextPart(text=MASKED_READ_RESULT),)),
        Action(id="step-2", parts=(TextPart(text=MASKED_LAST_REPLY),)),
    )


def test_without_a_redaction_the_trajectory_and_the_log_hold_the_text_as_it_was(
    run_mode: RunMode,
) -> None:
    # Arrange
    agent = build_agent(
        KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET),
        redaction=None,
    )

    # Act
    result = run_agent(agent, mode=run_mode, task=TASK_WITH_SECRET)

    # Assert
    assert Trajectory.load(read_stored_trajectory(result)).history == (
        Observation(author=Author.USER, parts=(TextPart(text=TASK_WITH_SECRET),)),
        Action(
            id="step-1",
            parts=(TextPart(text=FIRST_REPLY_WITH_SECRET),),
            tool_calls=(READ_CALL,),
        ),
        ToolObservation(call_id="call-read", parts=(TextPart(text=READ_RESULT_WITH_SECRET),)),
        Action(id="step-2", parts=(TextPart(text=LAST_REPLY_WITH_SECRET),)),
    )
    assert [record["samples"][0]["reason"] for record in result["monitor_log"]] == [
        f"mentions {SECRET}",
        f"mentions {SECRET}",
    ]
