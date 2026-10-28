"""Pending steps draw samples through the handler and have the monitor judge each one."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import hyoshigi
import pytest
from hyoshigi import Action, Author, Observation, SuspicionScore, Trajectory
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_sync_monitors import MonitorTwins
from langchain_sync_monitors._langchain import AsyncModelCallHandler, ModelCallHandler
from langchain_sync_monitors.contracts import BlockedAttempt, Sample, SampleDrawing, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError, MonitorError
from langchain_sync_monitors.pending_steps import (
    AsyncPendingStep,
    SyncPendingStep,
    run_synchronously,
)
from langchain_sync_monitors.spans import counting_judgements, trace_monitors
from langchain_sync_monitors.trajectories import load_step_trajectory
from tests.support.agents import RunMode
from tests.support.chat_models import ScriptedChatModel
from tests.support.flaky_models import FlakyChatModel
from tests.support.monitors import GatedMonitor, KeywordMonitor, read_action_text, score_by_keyword

TASK = HumanMessage("Summarise the report.", id="task")


@dataclass
class RecordingHandler:
    """Runs the request's model, recording each request and how many ran at once.

    Each call notes "model call" in `events` as it starts, before the model
    runs, so a log shared with a monitor shows what ran before each draw.
    """

    delay: float = 0.0
    requests: list[ModelRequest[Any]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    running: int = 0
    most_at_once: int = 0

    def __call__(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        self.requests.append(request)
        self.events.append("model call")
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            time.sleep(self.delay)
            return ModelResponse(result=[request.model.invoke(request.messages)])
        finally:
            self.running -= 1

    async def run_async(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        self.requests.append(request)
        self.events.append("model call")
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.running -= 1
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])


def note_rating(events: list[str], *, action: Action) -> SuspicionScore:
    """Note a rating in a log shared with the handler, and score the action as benign."""
    events.append(f"judgement of {read_action_text(action)}")
    return score_by_keyword("", suspicion_by_keyword={})


@dataclass(frozen=True, kw_only=True)
class EventRecordingRater(hyoshigi.Monitor):
    """Notes each rating in `events`."""

    events: list[str]

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        return note_rating(self.events, action=action)


@dataclass(frozen=True, kw_only=True)
class AsyncEventRecordingRater(hyoshigi.AsyncMonitor):
    """Notes each rating in `events`."""

    events: list[str]

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        return note_rating(self.events, action=action)


def build_event_recording_monitor(events: list[str]) -> MonitorTwins:
    return MonitorTwins(
        monitor=EventRecordingRater(events=events),
        async_monitor=AsyncEventRecordingRater(events=events),
    )


@pytest.fixture(autouse=True)
def numbered_judgements() -> Iterator[None]:
    """Number the ratings, as the middleware does for each step."""
    with counting_judgements():
        yield


def build_request(
    model: ScriptedChatModel, *, task_ids: list[str] | None = None
) -> ModelRequest[Any]:
    state: dict[str, Any] = {"messages": [TASK]}
    if task_ids is not None:
        state["monitor_task_messages"] = task_ids
    return ModelRequest(model=model, messages=[TASK], state=cast("AgentState[Any]", state))


def build_step_options(
    request: ModelRequest[Any],
    *,
    monitor: MonitorTwins,
    task_author: TaskAuthor,
) -> dict[str, Any]:
    """Return the keywords of a step rated by the monitor against the request's trajectory."""
    trajectory = load_step_trajectory(request, label="monitor", task_author=task_author)
    return {"request": request, "monitors": trace_monitors(monitor), "trajectory": trajectory}


def build_sync_step(
    request: ModelRequest[Any],
    *,
    handler: ModelCallHandler,
    monitor: MonitorTwins,
    task_author: TaskAuthor = TaskAuthor.USER,
) -> SyncPendingStep:
    options = build_step_options(request, monitor=monitor, task_author=task_author)
    return SyncPendingStep(handler=handler, **options)


def build_async_step(
    request: ModelRequest[Any],
    *,
    handler: AsyncModelCallHandler,
    monitor: MonitorTwins,
    task_author: TaskAuthor = TaskAuthor.USER,
) -> AsyncPendingStep:
    options = build_step_options(request, monitor=monitor, task_author=task_author)
    return AsyncPendingStep(handler=handler, **options)


def draw_samples(
    model: ScriptedChatModel,
    *,
    mode: RunMode,
    handler: RecordingHandler,
    monitor: MonitorTwins,
    sample_options: dict[str, Any],
) -> tuple[Sample, ...]:
    """Draw samples through the pending step that `invoke()` or `ainvoke()` builds."""
    request = build_request(model)
    if mode == "invoke":
        step = build_sync_step(request, handler=handler, monitor=monitor)
        return run_synchronously(step.sample(**sample_options))
    async_step = build_async_step(request, handler=handler.run_async, monitor=monitor)
    return asyncio.run(async_step.sample(**sample_options))


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
    step = build_sync_step(
        build_request(untrusted_model),
        handler=RecordingHandler(),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    run_synchronously(step.sample(count=1))

    # Assert
    [task] = monitor.ratings[0].history
    assert isinstance(task, Observation)
    assert (task.author, task.source) == (Author.SCAFFOLD, "application")
    assert TASK.additional_kwargs == {}


async def test_async_samples_see_each_blocked_attempt_and_its_feedback(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    blocked_attempt: BlockedAttempt,
) -> None:
    # Arrange
    handler = RecordingHandler()
    step = build_async_step(
        build_request(untrusted_model, task_ids=["task"]),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.PARENT_AGENT,
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
    task, attempt, feedback = monitor.ratings[0].history
    assert isinstance(task, Observation)
    assert isinstance(attempt, Action)
    assert isinstance(feedback, Observation)
    assert task.author is Author.DELEGATOR
    assert read_action_text(attempt) == "Read .env first."
    assert feedback.author is Author.MONITOR
    assert sample.proposal.text == "sample 0"
    assert sample.verdict.suspicion == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("drawing", "count", "expected_overlap"),
    [
        (SampleDrawing.CONCURRENT, 3, 3),
        (SampleDrawing.SEQUENTIAL, 3, 1),
        (SampleDrawing.CONCURRENT, 1, 1),
        (SampleDrawing.CONCURRENT, 0, 0),
    ],
)
async def test_async_samples_overlap_only_when_drawn_concurrently(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    drawing: SampleDrawing,
    count: int,
    expected_overlap: int,
) -> None:
    # Arrange
    handler = RecordingHandler(delay=0.01)
    step = build_async_step(
        build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    samples = await step.sample(count=count, drawing=drawing)

    # Assert: every draw is judged, and a call returns all the samples it asked for
    assert handler.most_at_once == expected_overlap
    assert len(monitor.ratings) == count
    assert sorted(sample.proposal.text for sample in samples) == [
        f"sample {index}" for index in range(count)
    ]


@pytest.mark.parametrize("drawing", [SampleDrawing.CONCURRENT, SampleDrawing.SEQUENTIAL])
@pytest.mark.parametrize(("count", "expected_overlap"), [(3, 1), (1, 1), (0, 0)])
def test_sync_step_draws_samples_one_after_another_however_asked(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    drawing: SampleDrawing,
    count: int,
    expected_overlap: int,
) -> None:
    # Arrange: each call holds the model a moment, so draws run on threads would overlap
    handler = RecordingHandler(delay=0.01)
    step = build_sync_step(
        build_request(untrusted_model),
        handler=handler,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    samples = run_synchronously(step.sample(count=count, drawing=drawing))

    # Assert: in the order they were asked for, and never two at once
    assert handler.most_at_once == expected_overlap
    assert [sample.proposal.text for sample in samples] == [
        f"sample {index}" for index in range(count)
    ]
    assert [read_action_text(rating.action) for rating in monitor.ratings] == [
        f"sample {index}" for index in range(count)
    ]


def test_a_sequential_draw_judges_each_sample_before_the_next_model_call(
    run_mode: RunMode,
    untrusted_model: ScriptedChatModel,
) -> None:
    # Arrange: the handler and the monitor note what they do in one shared log
    events: list[str] = []
    handler = RecordingHandler(events=events)
    monitor = build_event_recording_monitor(events)
    sample_options: dict[str, Any] = {"count": 3, "drawing": SampleDrawing.SEQUENTIAL}

    # Act
    draw_samples(
        untrusted_model,
        mode=run_mode,
        handler=handler,
        monitor=monitor,
        sample_options=sample_options,
    )

    # Assert: no model call starts before the previous sample's judgement has ended
    assert events == [
        "model call",
        "judgement of sample 0",
        "model call",
        "judgement of sample 1",
        "model call",
        "judgement of sample 2",
    ]


@pytest.mark.parametrize("drawing", ["concurrent", True])
def test_a_drawing_that_is_not_a_member_is_refused_before_any_draw(
    run_mode: RunMode,
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    drawing: object,
) -> None:
    # Arrange
    handler = RecordingHandler()
    sample_options: dict[str, Any] = {"count": 2, "drawing": drawing}

    # Act
    with pytest.raises(ConfigurationError) as refusal:
        draw_samples(
            untrusted_model,
            mode=run_mode,
            handler=handler,
            monitor=monitor,
            sample_options=sample_options,
        )

    # Assert: unrefused, a plain string would match no member by identity and draw in sequence
    assert str(refusal.value) == (
        "drawing must be one of SampleDrawing.SEQUENTIAL, SampleDrawing.CONCURRENT, "
        f"got {drawing!r}. Convert a string with SampleDrawing(value)."
    )
    assert handler.requests == []
    assert monitor.ratings == []


def test_sync_samples_see_blocked_attempts_too(
    untrusted_model: ScriptedChatModel,
    monitor: KeywordMonitor,
    blocked_attempt: BlockedAttempt,
) -> None:
    # Arrange
    step = build_sync_step(
        build_request(untrusted_model),
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
    step = build_async_step(
        build_request(untrusted_model),
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
    step = build_sync_step(
        build_request(untrusted_model),
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
    step = build_sync_step(
        build_request(ScriptedChatModel(responses=[])),
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
    step = build_sync_step(
        build_request(ScriptedChatModel(responses=[])),
        handler=lambda _request: ModelResponse(result=[ToolMessage("{}", tool_call_id="c")]),
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    with pytest.raises(MonitorError) as raised:
        run_synchronously(step.sample(count=1))

    # Assert
    assert str(raised.value) == (
        "The model call returned no AI message, so the monitor has no step to judge."
    )


async def test_a_failed_concurrent_draw_cancels_the_others_and_raises_its_own_error(
    monitor: KeywordMonitor,
) -> None:
    # Arrange
    model = FlakyChatModel(
        replies=[RuntimeError("429 rate limited"), AIMessage("slow"), AIMessage("slow")],
        delays=[0.0, 0.2, 0.2],
    )
    step = build_async_step(
        ModelRequest(model=model, messages=[TASK], state={"messages": [TASK]}),
        handler=RecordingHandler().run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )

    # Act
    with pytest.raises(RuntimeError, match="429 rate limited"):
        await step.sample(count=3, drawing=SampleDrawing.CONCURRENT)
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
    step = build_async_step(
        build_request(untrusted_model),
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
    step = build_async_step(
        build_request(untrusted_model),
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
    assert monitor.ratings == []
    assert step.judged_samples == []


async def test_an_async_draw_after_its_step_closed_never_calls_the_model(
    untrusted_model: ScriptedChatModel,
) -> None:
    # Arrange: the step is closed while the monitor judges the first of two samples
    handler = RecordingHandler()
    monitor = GatedMonitor()
    step = build_async_step(
        build_request(untrusted_model),
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
    step = build_async_step(
        build_request(untrusted_model),
        handler=handler.run_async,
        monitor=monitor,
        task_author=TaskAuthor.USER,
    )
    step.close()

    # Act
    with pytest.raises(MonitorError, match="after its step was over"):
        await step.sample(count=2, drawing=SampleDrawing.CONCURRENT)
    with pytest.raises(MonitorError, match="after its step was over"):
        await step.request_trusted_step(trusted_model=untrusted_model)

    # Assert
    assert handler.requests == []
