"""The monitor's spans leave every stream as it was; only `astream_events` grows.

The spans reach callback handlers alone. Each stream is compared part by part,
as JSON, leaving out only the ids and checkpoint names that differ between
any two runs, across three runs: with the spans switched off, as a monitor
without them would stream; with the spans on and no tracer, where LangGraph's
own handler for `stream_mode="messages"` still receives them; and with the
spans on and a tracer attached.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Literal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langchain_sync_monitors import _langchain
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode
from tests.support.agents import (
    SUSPICION_BY_KEYWORD,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    build_task_input,
)
from tests.support.chat_models import StreamingScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.tracing import RecordingTracer, build_judge_model

type StreamMode = Literal["messages", "updates", "custom"]

FINAL_ANSWER = "Q3 revenue grew 12%."
RUN_SPECIFIC_KEYS = frozenset(
    {
        "id",
        "run_id",
        "thread_id",
        "checkpoint_ns",
        "langgraph_checkpoint_ns",
        "monitor_task_messages",
        "monitor_seen_human_messages",
    },
)
"""Keys whose values differ between any two runs: random message ids and checkpoint names.

The monitor's lists of message ids, which `stream_mode="updates"` carries, hold ids too.
"""

MONITOR_SPAN_NAMES = {"monitor step", "monitor judgement", "monitor decision"}


def build_agent() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build an agent whose first step Auto Mode blocks, and whose steps stream token by token."""
    monitor = KeywordMonitor(
        suspicion_by_keyword=SUSPICION_BY_KEYWORD,
        judge_model=build_judge_model(),
    )
    return create_agent(
        model=StreamingScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=AutoMode(block_threshold=0.6))],
    )


def build_config(tracer: RecordingTracer | None) -> RunnableConfig | None:
    return None if tracer is None else RunnableConfig(callbacks=[tracer])


def normalise(value: object) -> object:
    """Return the value as plain data, without the keys that differ between runs."""
    if isinstance(value, BaseMessage):
        return normalise(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): normalise(item) for key, item in value.items() if key not in RUN_SPECIFIC_KEYS
        }
    if isinstance(value, list | tuple):
        return [normalise(item) for item in value]
    return value


def render_parts(parts: list[Any]) -> list[str]:
    """Render every streamed part as the JSON a consumer would compare byte for byte."""
    return [json.dumps(normalise(part), sort_keys=True, default=str) for part in parts]


def stream_parts(
    *,
    mode: RunMode,
    stream_mode: StreamMode,
    tracer: RecordingTracer | None,
) -> list[str]:
    """Stream a fresh agent in one stream mode, and return every part rendered."""
    agent, payload, config = build_agent(), build_task_input(), build_config(tracer)
    if mode == "invoke":
        return render_parts(list(agent.stream(payload, config, stream_mode=stream_mode)))

    async def collect() -> list[Any]:
        return [part async for part in agent.astream(payload, config, stream_mode=stream_mode)]

    return render_parts(asyncio.run(collect()))


def stream_v3_messages(*, tracer: RecordingTracer | None) -> list[str]:
    """Return every message of the v3 event stream's `run.messages`, rendered."""
    run = build_agent().stream_events(build_task_input(), build_config(tracer), version="v3")
    return render_parts([message_stream.output for message_stream in run.messages])


def switch_spans_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every span helper find no handler, so the run opens no span at all."""
    monkeypatch.setattr(_langchain, "build_span_manager", lambda *_, **__: None)


def collect_events(*, exclude_tags: list[str] | None = None) -> list[dict[str, Any]]:
    """Return every `astream_events` event of a run, as the v2 event stream reports it."""

    async def collect() -> list[dict[str, Any]]:
        stream = build_agent().astream_events(
            build_task_input(),
            version="v2",
            exclude_tags=exclude_tags,
        )
        return [dict(event) async for event in stream]

    return asyncio.run(collect())


@pytest.mark.parametrize("stream_mode", ["messages", "updates", "custom"])
def test_the_spans_leave_the_stream_byte_for_byte_the_same(
    run_mode: RunMode,
    stream_mode: StreamMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    tracer = RecordingTracer()

    # Act
    with monkeypatch.context() as patch:
        switch_spans_off(patch)
        without_spans = stream_parts(mode=run_mode, stream_mode=stream_mode, tracer=None)
    untraced = stream_parts(mode=run_mode, stream_mode=stream_mode, tracer=None)
    traced = stream_parts(mode=run_mode, stream_mode=stream_mode, tracer=tracer)

    # Assert
    assert without_spans
    assert untraced == without_spans
    assert traced == without_spans
    assert {span.name for span in tracer.find_monitor_spans()} == MONITOR_SPAN_NAMES


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
def test_the_spans_leave_the_v3_event_stream_s_messages_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    tracer = RecordingTracer()

    # Act
    with monkeypatch.context() as patch:
        switch_spans_off(patch)
        without_spans = stream_v3_messages(tracer=None)
    untraced = stream_v3_messages(tracer=None)
    traced = stream_v3_messages(tracer=tracer)

    # Assert
    assert len(without_spans) == 2
    assert untraced == without_spans
    assert traced == without_spans
    assert tracer.find_monitor_spans()


def test_astream_events_reports_the_spans_and_drops_them_by_their_tag() -> None:
    # Act
    events = collect_events()
    without_monitor = collect_events(exclude_tags=["monitor"])

    # Assert
    started_spans = {
        event["name"]
        for event in events
        if event["event"] == "on_chain_start" and event["name"].startswith("monitor ")
    }
    assert started_spans == MONITOR_SPAN_NAMES
    assert not any(event["name"].startswith("monitor ") for event in without_monitor)
    model_starts = [event for event in events if event["event"] == "on_chat_model_start"]
    kept_model_starts = [
        event for event in without_monitor if event["event"] == "on_chat_model_start"
    ]
    assert len(kept_model_starts) == len(model_starts)
