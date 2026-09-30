"""The spans with LangSmith tracing on, alone and beside another tracer.

`create_agent` wraps each middleware hook in a `traceable` run only LangSmith
sees, and a callback manager configured inside the hook takes that run as its
parent. The spans are opened from the node's own callback manager instead, so
they nest under the model node in LangSmith, and a second handler, such as
Langfuse's, never receives a step span whose parent it has not seen.

LangSmith builds each `traceable` run opened inside a span from the span's run
and copies the span's metadata into it, so such runs carry the step's labels
in LangSmith. The library cannot prevent that without dropping the labels, so
the tests pin it. LangSmith's client is a mock, so nothing leaves the machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import traceable, tracing_context

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun, TrustedMonitoring
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
from tests.support.tracing import JUDGE_MODEL_NAME, RecordingTracer, build_judge_model

MONITOR_SPAN_NAMES = {"monitor step", "monitor judgement", "monitor decision"}
MONITOR_CALL = "monitor call"
TRACED_CHECK_NAME = "traced check"
HOOK_NAMES = {"invoke": "wrap_model_call", "ainvoke": "awrap_model_call"}


@traceable(name=TRACED_CHECK_NAME)
def run_traced_check() -> str:
    return "fine"


@traceable(name=TRACED_CHECK_NAME)
async def arun_traced_check() -> str:
    return "fine"


@dataclass(kw_only=True)
class TracedMonitor(Monitor):
    """Traces part of its own work with LangSmith's `traceable`, and finds every step benign."""

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        await arun_traced_check()
        return Verdict(suspicion=0.1, reason="nothing suspicious")

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        run_traced_check()
        return Verdict(suspicion=0.1, reason="nothing suspicious")


@dataclass(frozen=True, kw_only=True)
class SentRun:
    """One run as LangSmith's client received it."""

    run_id: str
    parent_run_id: str | None
    name: str
    metadata: dict[str, Any]

    def read_labels(self) -> dict[str, Any]:
        return {key: value for key, value in self.metadata.items() if key.startswith("monitor_")}


def read_sent_runs(client: MagicMock) -> dict[str, SentRun]:
    """Return every run the mock client was asked to create, by id."""
    runs: dict[str, SentRun] = {}
    for call in client.create_run.call_args_list:
        keywords = call.kwargs
        parent_run_id = keywords.get("parent_run_id")
        runs[str(keywords["id"])] = SentRun(
            run_id=str(keywords["id"]),
            parent_run_id=None if parent_run_id is None else str(parent_run_id),
            name=keywords["name"],
            metadata=dict((keywords.get("extra") or {}).get("metadata") or {}),
        )
    return runs


def find_sent_runs(runs: dict[str, SentRun], name: str) -> list[SentRun]:
    return [run for run in runs.values() if run.name == name]


def find_sent_parent(runs: dict[str, SentRun], run: SentRun) -> SentRun:
    assert run.parent_run_id is not None, f"{run.name} has no parent"
    return runs[run.parent_run_id]


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


def build_stacked_agent() -> Runnable[Any, Any]:
    """Build an agent with two monitors whose judges trace themselves with `traceable`."""
    return create_agent(
        model=ScriptedChatModel(responses=[AIMessage("Done.")]),
        tools=[],
        middleware=[
            MonitorMiddleware(
                monitor=TracedMonitor(),
                protocol=TrustedMonitoring(flag_threshold=0.8),
                label="outer",
            ),
            MonitorMiddleware(
                monitor=TracedMonitor(),
                protocol=TrustedMonitoring(flag_threshold=0.8),
                label="inner",
            ),
        ],
    )


@pytest.fixture
def langsmith_client() -> Iterator[MagicMock]:
    """Turn LangSmith tracing on with a mock client, which records runs and sends nothing."""
    client = MagicMock()
    client.otel_exporter = None
    with tracing_context(enabled=True, client=client):
        yield client


def build_langsmith_tracer(client: MagicMock) -> LangChainTracer:
    """Return LangSmith's tracer on the mock client."""
    return LangChainTracer(client=client, project_name="monitor-spans")


def test_the_spans_nest_under_the_model_node_for_a_handler_beside_langsmith(
    run_mode: RunMode,
    langsmith_client: MagicMock,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    config = RunnableConfig(callbacks=[build_langsmith_tracer(langsmith_client), tracer])

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
    sent_runs = read_sent_runs(langsmith_client)
    assert {run.name for run in sent_runs.values()} >= MONITOR_SPAN_NAMES
    assert f"monitor[main].{HOOK_NAMES[run_mode]}" in {run.name for run in sent_runs.values()}


def test_the_spans_nest_the_same_way_in_langsmith(
    run_mode: RunMode,
    langsmith_client: MagicMock,
) -> None:
    # Act
    run_agent(
        build_agent(),
        mode=run_mode,
        config=RunnableConfig(callbacks=[build_langsmith_tracer(langsmith_client)]),
    )

    # Assert
    sent_runs = read_sent_runs(langsmith_client)
    steps = find_sent_runs(sent_runs, "monitor step")
    spans_below_the_step = [
        *find_sent_runs(sent_runs, "monitor judgement"),
        *find_sent_runs(sent_runs, "monitor decision"),
    ]
    judge_calls = find_sent_runs(sent_runs, MONITOR_CALL)
    samples = find_sent_runs(sent_runs, "ScriptedChatModel")
    assert [find_sent_parent(sent_runs, step).name for step in steps] == ["model", "model"]
    assert {find_sent_parent(sent_runs, span).name for span in spans_below_the_step} == {
        "monitor step"
    }
    assert {find_sent_parent(sent_runs, call).name for call in judge_calls} == {"monitor judgement"}
    assert [find_sent_parent(sent_runs, sample) for sample in samples[:2]] == [steps[0]] * 2


def test_langsmith_traceable_runs_inside_a_span_carry_its_labels(
    run_mode: RunMode,
    langsmith_client: MagicMock,
) -> None:
    # Act
    run_agent(
        build_stacked_agent(),
        mode=run_mode,
        config=RunnableConfig(callbacks=[build_langsmith_tracer(langsmith_client)]),
    )

    # Assert
    sent_runs = read_sent_runs(langsmith_client)
    outer_step, inner_step = find_sent_runs(sent_runs, "monitor step")
    assert (outer_step.metadata["monitor_name"], inner_step.metadata["monitor_name"]) == (
        "outer",
        "inner",
    )
    [inner_hook] = find_sent_runs(sent_runs, f"inner[main].{HOOK_NAMES[run_mode]}")
    assert find_sent_parent(sent_runs, inner_hook) == outer_step
    assert inner_hook.read_labels() == outer_step.read_labels()
    traced_checks = find_sent_runs(sent_runs, TRACED_CHECK_NAME)
    assert len(traced_checks) == 2
    for traced_check in traced_checks:
        judgement = find_sent_parent(sent_runs, traced_check)
        assert judgement.name == "monitor judgement"
        assert traced_check.read_labels() == judgement.read_labels()
        assert traced_check.metadata["ls_agent_type"] == "middleware"
    [sample] = find_sent_runs(sent_runs, "ScriptedChatModel")
    assert find_sent_parent(sent_runs, sample) == inner_step
    assert sample.read_labels() == {}
    assert "ls_agent_type" not in sample.metadata


def test_langsmith_receives_the_monitor_calls_by_their_fixed_name_with_their_model(
    run_mode: RunMode,
    langsmith_client: MagicMock,
) -> None:
    # Act
    run_agent(
        build_agent(),
        mode=run_mode,
        config=RunnableConfig(callbacks=[build_langsmith_tracer(langsmith_client)]),
    )

    # Assert: three verdicts, each one judge call named for the monitor, not for its model.
    sent_runs = read_sent_runs(langsmith_client)
    judge_calls = find_sent_runs(sent_runs, MONITOR_CALL)
    assert len(judge_calls) == 3
    assert find_sent_runs(sent_runs, "NamedFakeChatModel") == []
    for call in judge_calls:
        assert call.metadata["ls_model_name"] == JUDGE_MODEL_NAME
        assert call.metadata["ls_message_view_exclude"] is True
    samples = find_sent_runs(sent_runs, "ScriptedChatModel")
    assert len(samples) == 3
