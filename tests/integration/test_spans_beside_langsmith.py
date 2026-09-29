"""With LangSmith and another tracer attached together, the spans still nest under the model node.

`create_agent` wraps each middleware hook in a run only LangSmith sees, and a
callback manager configured inside the hook takes that run as its parent. The
spans are opened from the node's own callback manager instead, so a second
handler, such as Langfuse's, never receives a step span whose parent it has not
seen. LangSmith's client is a mock, so nothing leaves the machine.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer

from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
from tests.support.agents import (
    SUSPICION_BY_KEYWORD,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.tracing import RecordingTracer, build_judge_model

MONITOR_SPAN_NAMES = {"monitor step", "monitor judgement", "monitor decision"}


def build_agent() -> Runnable[Any, Any]:
    monitor = KeywordMonitor(
        suspicion_by_keyword=SUSPICION_BY_KEYWORD,
        judge_model=build_judge_model(),
    )
    protocol = DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=0.95)
    return create_agent(
        model=ScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
        ),
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )


def build_langsmith_tracer(client: MagicMock) -> LangChainTracer:
    """Return LangSmith's tracer on a mock client, which records runs and sends nothing."""
    client.otel_exporter = None
    return LangChainTracer(client=client, project_name="monitor-spans")


def test_the_spans_nest_under_the_model_node_for_a_handler_beside_langsmith(
    run_mode: RunMode,
) -> None:
    # Arrange
    client = MagicMock()
    tracer = RecordingTracer()
    config = RunnableConfig(callbacks=[build_langsmith_tracer(client), tracer])

    # Act
    run_agent(build_agent(), mode=run_mode, config=config)

    # Assert
    steps = tracer.find_runs("monitor step")
    assert [tracer.find_parent(step).name for step in steps] == ["model", "model"]
    assert not any(name.startswith("monitor ") for name in tracer.find_unknown_parents())
    assert all(
        sample.parent_run_id == steps[0].run_id
        for sample in tracer.find_runs("ScriptedChatModel")[:2]
    )
    sent_to_langsmith = {call.kwargs["name"] for call in client.create_run.call_args_list}
    assert sent_to_langsmith >= MONITOR_SPAN_NAMES
