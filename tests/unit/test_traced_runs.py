"""The helpers that open the monitor's spans through LangChain's callback managers.

Each test runs a span inside a stand-in graph node, a `RunnableLambda` whose
config carries the node's own callback manager, once through the sync helper
under `invoke()` and once through the async helper under `ainvoke()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, cast
from uuid import UUID

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManager,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import ensure_config, var_child_runnable_config

from langchain_sync_monitors._langchain import (
    TracedRun,
    TraceSpan,
    label_monitor_spans,
    open_traced_run,
    open_traced_run_sync,
)
from tests.support.tracing import RecordingTracer

type CallPath = Literal["sync", "async"]
type SpanBody = Callable[[TracedRun], None]

SPAN = TraceSpan(
    name="monitor step",
    inputs={"step_number": 1},
    metadata={"monitor_agent": "main"},
    tags=["monitor"],
)
NODE_TAG = "node-tag"
NODE_METADATA = {"langgraph_node": "model"}


@pytest.fixture(params=["sync", "async"])
def call_path(request: pytest.FixtureRequest) -> CallPath:
    """Open each span once through the sync helper and once through the async one."""
    return cast("CallPath", request.param)


def run_span_in_node(
    span: TraceSpan,
    *,
    call_path: CallPath,
    body: SpanBody,
    callbacks: list[BaseCallbackHandler] | None,
) -> None:
    """Run `body` inside the span, inside a node that has the callbacks, as a graph would."""

    def node(_: object) -> None:
        with open_traced_run_sync(span) as traced_run:
            body(traced_run)

    async def async_node(_: object) -> None:
        async with open_traced_run(span) as traced_run:
            body(traced_run)

    runnable = RunnableLambda(node, afunc=async_node, name="node")
    config = RunnableConfig(callbacks=callbacks, tags=[NODE_TAG], metadata=NODE_METADATA)
    if call_path == "sync":
        runnable.invoke(None, config)
    else:
        asyncio.run(runnable.ainvoke(None, config))


def call_model(_: TracedRun) -> None:
    FakeListChatModel(responses=["ok"]).invoke("hello")


def test_the_span_nests_under_the_node_and_the_calls_inside_it_nest_under_the_span(
    call_path: CallPath,
) -> None:
    # Arrange
    tracer = RecordingTracer()

    # Act
    run_span_in_node(SPAN, call_path=call_path, body=call_model, callbacks=[tracer])

    # Assert
    [span] = tracer.find_runs("monitor step")
    [model_call] = tracer.find_runs("FakeListChatModel")
    assert tracer.find_parent(span).name == "node"
    assert tracer.find_parent(model_call) is span
    assert (span.run_type, span.inputs, span.error) == ("chain", SPAN.inputs, None)
    assert tracer.find_unknown_parents() == []
    assert tracer.find_open_runs() == []


def test_the_span_s_own_tags_and_metadata_stay_off_the_calls_inside_it(
    call_path: CallPath,
) -> None:
    # Arrange
    tracer = RecordingTracer()

    # Act
    run_span_in_node(SPAN, call_path=call_path, body=call_model, callbacks=[tracer])

    # Assert
    [span] = tracer.find_runs("monitor step")
    [model_call] = tracer.find_runs("FakeListChatModel")
    assert span.tags == [NODE_TAG, "monitor"]
    assert span.metadata["monitor_agent"] == "main"
    assert span.metadata["langgraph_node"] == "model"
    assert "monitor" not in model_call.tags
    assert NODE_TAG in model_call.tags
    assert "monitor_agent" not in model_call.metadata
    assert model_call.metadata["langgraph_node"] == "model"


def test_the_outputs_and_the_late_inputs_end_the_span(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()

    def report(traced_run: TracedRun) -> None:
        traced_run.outputs = {"outcome": "allowed", "samples": [{"suspicion": 0.1}]}
        traced_run.inputs_at_end = {"step_number": 1, "proposed_step": "<proposed_step/>"}

    # Act
    run_span_in_node(SPAN, call_path=call_path, body=report, callbacks=[tracer])

    # Assert
    [span] = tracer.find_runs("monitor step")
    assert span.outputs == {"outcome": "allowed", "samples": [{"suspicion": 0.1}]}
    assert span.inputs == {"step_number": 1, "proposed_step": "<proposed_step/>"}


def test_a_span_without_late_inputs_keeps_the_inputs_it_started_with(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()

    # Act
    run_span_in_node(SPAN, call_path=call_path, body=lambda _: None, callbacks=[tracer])

    # Assert
    [span] = tracer.find_runs("monitor step")
    assert (span.inputs, span.outputs) == ({"step_number": 1}, {})


def test_an_exception_ends_the_span_with_it_and_still_propagates(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()

    def fail(traced_run: TracedRun) -> None:
        traced_run.inputs_at_end = {"step_number": 1, "proposed_step": "<proposed_step/>"}
        message = "judge provider timed out"
        raise TimeoutError(message)

    # Act
    with pytest.raises(TimeoutError, match="judge provider timed out"):
        run_span_in_node(SPAN, call_path=call_path, body=fail, callbacks=[tracer])

    # Assert
    [span] = tracer.find_runs("monitor step")
    assert isinstance(span.error, TimeoutError)
    assert span.inputs == {"step_number": 1, "proposed_step": "<proposed_step/>"}
    assert tracer.find_open_runs() == []


def test_the_given_run_id_names_the_span(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()
    run_id = UUID("01a0ee00-0000-7000-8000-000000000001")
    span = TraceSpan(name="monitor step", run_id=run_id)

    # Act
    run_span_in_node(span, call_path=call_path, body=lambda _: None, callbacks=[tracer])

    # Assert
    [recorded] = tracer.find_runs("monitor step")
    assert recorded.run_id == run_id


def test_labels_reach_every_span_in_the_block_but_no_model_call(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()
    nested = TraceSpan(name="monitor judgement", metadata={"ls_agent_type": "middleware"})

    def open_nested_span(_: TracedRun) -> None:
        with label_monitor_spans({"monitor_step_id": "step-1"}), open_traced_run_sync(nested):
            FakeListChatModel(responses=["ok"]).invoke("hello")

    # Act
    run_span_in_node(SPAN, call_path=call_path, body=open_nested_span, callbacks=[tracer])

    # Assert
    [step] = tracer.find_runs("monitor step")
    [judgement] = tracer.find_runs("monitor judgement")
    [model_call] = tracer.find_runs("FakeListChatModel")
    assert judgement.metadata["monitor_step_id"] == "step-1"
    assert judgement.metadata["ls_agent_type"] == "middleware"
    assert "monitor_step_id" not in step.metadata
    assert "monitor_step_id" not in model_call.metadata


def test_the_node_s_callback_manager_is_left_unchanged(call_path: CallPath) -> None:
    # Arrange
    tracer = RecordingTracer()
    seen: list[tuple[list[str], dict[str, Any]]] = []

    def read_node_manager() -> BaseCallbackManager:
        manager = ensure_config()["callbacks"]
        assert isinstance(manager, BaseCallbackManager)
        return manager

    def node(_: object) -> None:
        manager = read_node_manager()
        with open_traced_run_sync(SPAN):
            pass
        seen.append((list(manager.tags), dict(manager.metadata)))

    async def async_node(_: object) -> None:
        manager = read_node_manager()
        async with open_traced_run(SPAN):
            pass
        seen.append((list(manager.tags), dict(manager.metadata)))

    runnable = RunnableLambda(node, afunc=async_node, name="node")
    config = RunnableConfig(callbacks=[tracer])

    # Act
    if call_path == "sync":
        runnable.invoke(None, config)
    else:
        asyncio.run(runnable.ainvoke(None, config))

    # Assert
    [(tags, metadata)] = seen
    assert "monitor" not in tags
    assert "monitor_agent" not in metadata


def test_callbacks_given_as_a_list_outside_a_graph_still_reach_the_span(
    call_path: CallPath,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    token = var_child_runnable_config.set(RunnableConfig(callbacks=[tracer]))

    # Act
    try:
        if call_path == "sync":
            with open_traced_run_sync(SPAN):
                pass
        else:

            async def open_span() -> None:
                async with open_traced_run(SPAN):
                    pass

            asyncio.run(open_span())
    finally:
        var_child_runnable_config.reset(token)

    # Assert
    [span] = tracer.find_runs("monitor step")
    assert (span.parent_run_id, span.tags, span.ended) == (None, ["monitor"], True)


@pytest.fixture
def started_run_names(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of every chain run any callback manager starts, with or without handlers."""
    names: list[str] = []
    sync_start = CallbackManager.on_chain_start
    async_start = AsyncCallbackManager.on_chain_start

    def record_sync(manager: CallbackManager, *arguments: Any, **keywords: Any) -> Any:
        names.append(str(keywords.get("name")))
        return sync_start(manager, *arguments, **keywords)

    async def record_async(manager: AsyncCallbackManager, *arguments: Any, **keywords: Any) -> Any:
        names.append(str(keywords.get("name")))
        return await async_start(manager, *arguments, **keywords)

    monkeypatch.setattr(CallbackManager, "on_chain_start", record_sync)
    monkeypatch.setattr(AsyncCallbackManager, "on_chain_start", record_async)
    return names


def test_without_a_tracer_no_span_starts_and_the_block_sees_the_node_s_config(
    call_path: CallPath,
    started_run_names: list[str],
) -> None:
    # Arrange
    configs: list[RunnableConfig | None] = []

    def read_config(_: TracedRun) -> None:
        configs.append(var_child_runnable_config.get())

    def node(_: object) -> None:
        configs.append(var_child_runnable_config.get())
        with open_traced_run_sync(SPAN) as traced_run:
            read_config(traced_run)

    async def async_node(_: object) -> None:
        configs.append(var_child_runnable_config.get())
        async with open_traced_run(SPAN) as traced_run:
            read_config(traced_run)

    runnable = RunnableLambda(node, afunc=async_node, name="node")

    # Act
    if call_path == "sync":
        runnable.invoke(None)
    else:
        asyncio.run(runnable.ainvoke(None))

    # Assert
    outside, inside = configs
    assert inside is outside
    assert "monitor step" not in started_run_names
    assert "node" in started_run_names


async def test_a_cancelled_block_ends_its_span_with_the_cancellation() -> None:
    # Arrange
    tracer = RecordingTracer()
    entered = asyncio.Event()

    async def async_node(_: object) -> None:
        async with open_traced_run(SPAN):
            entered.set()
            await asyncio.sleep(10)

    runnable = RunnableLambda(async_node, name="node")
    task = asyncio.create_task(runnable.ainvoke(None, RunnableConfig(callbacks=[tracer])))
    await entered.wait()

    # Act
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Assert
    [span] = tracer.find_runs("monitor step")
    assert isinstance(span.error, asyncio.CancelledError)
    assert span.ended
