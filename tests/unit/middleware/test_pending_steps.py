"""Pending steps draw samples through the handler and have the monitor judge each one."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_sync_monitors.contracts import BlockedAttempt, StepRecord, TaskAuthor
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.pending_steps import (
    AsyncPendingStep,
    SyncPendingStep,
    run_synchronously,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel
from tests.support.monitors import GatedMonitor, KeywordMonitor

TASK = HumanMessage("Summarise the report.", id="task")


@dataclass
class RecordingHandler:
    """Runs the request's model, recording each request and how many ran at once."""

    delay: float = 0.0
    requests: list[ModelRequest[Any]] = field(default_factory=list)
    running: int = 0
    most_at_once: int = 0

    def __call__(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        self.requests.append(request)
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def run_async(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        self.requests.append(request)
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        await asyncio.sleep(self.delay)
        self.running -= 1
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])


def build_request(model: ScriptedChatModel) -> ModelRequest[Any]:
    return ModelRequest(model=model, messages=[TASK], state={"messages": [TASK]})


@pytest.fixture
def monitor() -> KeywordMonitor:
    return KeywordMonitor(suspicion_by_keyword={".env": 0.9})


@pytest.fixture
def untrusted_model() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[AIMessage(f"sample {index}") for index in range(3)])


@pytest.fixture
def blocked_attempt() -> BlockedAttempt:
    return BlockedAttempt(
        proposal=AIMessage("Read .env first.", id="blocked"),
        feedback="[Safety monitor] Blocked: reads credentials.",
    )


def test_without_recorded_input_the_monitor_reads_no_human_message_as_the_task_author(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange: a request whose state holds no run input, as when no before_agent hook ran
    step = SyncPendingStep(
        request=build_request(untrusted_model),
        handler=RecordingHandler(),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    run_synchronously(step.sample(count=1))

    # Assert
    [task] = monitor.inputs[0].history
    assert task.additional_kwargs == {"lc_source": "application"}
    assert TASK.additional_kwargs == {}


async def test_async_samples_see_each_blocked_attempt_and_its_feedback(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    blocked_attempt: BlockedAttempt,
) -> None:
    # Arrange
    handler = RecordingHandler()
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.PARENT_AGENT,
        task_message_ids=frozenset({"task"}),
    )

    # Act
    [sample] = await step.sample(count=1, blocked_attempts=(blocked_attempt,))

    # Assert
    seen = untrusted_model.calls[0]
    assert [message.text for message in seen] == [
        "Summarise the report.",
        "Read .env first.",
        "[Safety monitor] Blocked: reads credentials.",
    ]
    assert monitor.inputs[0].history == tuple(seen)
    assert monitor.inputs[0].task_author is TaskAuthor.PARENT_AGENT
    assert sample.proposal.text == "sample 0"
    assert sample.verdict.suspicion == pytest.approx(0.1)


@pytest.mark.parametrize(("concurrently", "expected_overlap"), [(True, 3), (False, 1)])
async def test_async_samples_overlap_only_when_drawn_concurrently(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    concurrently: bool,
    expected_overlap: int,
) -> None:
    # Arrange
    handler = RecordingHandler(delay=0.01)
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    samples = await step.sample(count=3, concurrently=concurrently)

    # Assert
    assert handler.most_at_once == expected_overlap
    assert sorted(sample.proposal.text for sample in samples) == [
        "sample 0",
        "sample 1",
        "sample 2",
    ]


def test_sync_step_draws_concurrent_samples_one_after_another(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    handler = RecordingHandler()
    step = SyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    samples = run_synchronously(step.sample(count=3, concurrently=True))

    # Assert
    assert [sample.proposal.text for sample in samples] == ["sample 0", "sample 1", "sample 2"]
    assert len(monitor.inputs) == 3


def test_sync_samples_see_blocked_attempts_too(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    blocked_attempt: BlockedAttempt,
) -> None:
    # Arrange
    step = SyncPendingStep(
        request=build_request(untrusted_model),
        handler=RecordingHandler(),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    run_synchronously(step.sample(count=1, blocked_attempts=(blocked_attempt,)))

    # Assert
    feedback = untrusted_model.calls[0][-1]
    assert isinstance(feedback, HumanMessage)
    assert feedback.additional_kwargs["lc_source"] == "monitor"


async def test_async_trusted_step_uses_the_trusted_model_through_the_handler(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    trusted_model = ScriptedChatModel(responses=[AIMessage("trusted step")])
    handler = RecordingHandler()
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    response = await step.request_trusted_step(trusted_model=trusted_model)

    # Assert
    assert response.result[0].text == "trusted step"
    assert handler.requests[0].model is trusted_model
    assert handler.requests[0].messages == [TASK]
    assert untrusted_model.calls == []


def test_sync_trusted_step_uses_the_trusted_model_through_the_handler(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    trusted_model = ScriptedChatModel(responses=[AIMessage("trusted step")])
    handler = RecordingHandler()
    step = SyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    response = run_synchronously(step.request_trusted_step(trusted_model=trusted_model))

    # Assert
    assert response.result[0].text == "trusted step"
    assert handler.requests[0].model is trusted_model


def test_the_proposal_is_the_first_ai_message_of_the_response(monitor: KeywordMonitor) -> None:
    # Arrange
    proposal = AIMessage("structured answer", id="answer")
    extra: list[BaseMessage] = [proposal, ToolMessage("{}", tool_call_id="c"), AIMessage("later")]
    step = SyncPendingStep(
        request=build_request(ScriptedChatModel(responses=[])),
        handler=lambda _request: ModelResponse(result=extra),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    [sample] = run_synchronously(step.sample(count=1))

    # Assert
    assert sample.proposal is proposal


def test_a_response_without_an_ai_message_raises(monitor: KeywordMonitor) -> None:
    # Arrange
    step = SyncPendingStep(
        request=build_request(ScriptedChatModel(responses=[])),
        handler=lambda _request: ModelResponse(result=[ToolMessage("{}", tool_call_id="c")]),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act / Assert
    with pytest.raises(MonitorError, match="no AI message"):
        run_synchronously(step.sample(count=1))


def test_previous_records_are_the_ones_given(monitor: KeywordMonitor) -> None:
    # Arrange
    record = StepRecord(
        agent="main",
        monitor="monitor",
        step_number=1,
        outcome="allowed",
        flagged=False,
        blocked_count=0,
        samples=[],
    )

    # Act
    step = SyncPendingStep(
        request=build_request(ScriptedChatModel(responses=[])),
        handler=RecordingHandler(),
        monitor=monitor,
        task_author=TaskAuthor.USER,
        previous_records=(record,),
    )

    # Assert
    assert step.previous_records == (record,)


async def test_a_failed_concurrent_draw_cancels_the_others_and_raises_its_own_error(
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    model = FlakyChatModel(
        replies=[RuntimeError("429 rate limited"), AIMessage("slow"), AIMessage("slow")],
        delays=[0.0, 0.2, 0.2],
    )
    step = AsyncPendingStep(
        request=ModelRequest(model=model, messages=[TASK], state={"messages": [TASK]}),
        handler=RecordingHandler().run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    with pytest.raises(RuntimeError, match="429 rate limited"):
        await step.sample(count=3, concurrently=True)
    await asyncio.sleep(0.3)

    # Assert
    assert model.started_calls == 3
    assert model.finished_calls == 0
    assert step.judged_samples == []


async def test_each_judged_sample_is_kept_on_the_step(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=RecordingHandler().run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    first = await step.sample(count=1)
    second = await step.sample(count=2)

    # Assert
    assert step.judged_samples == [*first, *second]


async def test_an_async_draw_the_step_outlasts_is_never_judged(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange: the model call is still running when the step is closed
    handler = RecordingHandler(delay=0.05)
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )
    draw = asyncio.create_task(step.sample(count=1))
    await asyncio.sleep(0.01)

    # Act
    step.close()
    with pytest.raises(MonitorError, match="after its step was over"):
        await draw

    # Assert
    assert len(handler.requests) == 1
    assert monitor.inputs == []
    assert step.judged_samples == []


async def test_an_async_draw_after_its_step_closed_never_calls_the_model(
    untrusted_model: ScriptedChatModel,
) -> None:
    # Arrange: the step is closed while the monitor judges the first of two samples
    handler = RecordingHandler()
    monitor = GatedMonitor()
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )
    draws = asyncio.create_task(step.sample(count=2))
    await monitor.judging.wait()

    # Act
    step.close()
    monitor.release.set()
    with pytest.raises(MonitorError, match="after its step was over"):
        await draws

    # Assert: the first sample was judged, and the second was refused before its model call
    assert len(handler.requests) == 1
    assert len(step.judged_samples) == 1


async def test_a_closed_async_step_refuses_every_call(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    handler = RecordingHandler()
    step = AsyncPendingStep(
        request=build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )
    step.close()

    # Act
    with pytest.raises(MonitorError, match="after its step was over"):
        await step.sample(count=2, concurrently=True)
    with pytest.raises(MonitorError, match="after its step was over"):
        await step.request_trusted_step(trusted_model=untrusted_model)

    # Assert
    assert handler.requests == []
