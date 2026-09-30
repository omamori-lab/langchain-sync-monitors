"""The score worker sends in windows, honours pauses, gives up on old scores and drains at exit.

The worker runs here on a clock the test moves, and its senders are scripted,
so each rule is checked at its exact boundary: a pause ends exactly when
`Retry-After` says, a score is dropped exactly when it has waited
`give_up_seconds`, and the drain stops exactly at its deadline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.score_worker import ScoreWorker, WorkerTimings
from langchain_sync_monitors.scores import DeliveryReport, PendingScore, ScoreSender, Tracer
from tests.support.score_services import FakeClock, build_step_id

type Answer = Callable[[Sequence[PendingScore]], DeliveryReport]


def write_all(scores: Sequence[PendingScore]) -> DeliveryReport:
    return DeliveryReport(written=list(scores))


def keep_waiting(scores: Sequence[PendingScore]) -> DeliveryReport:
    return DeliveryReport(waiting=list(scores))


def reject_all(scores: Sequence[PendingScore]) -> DeliveryReport:
    return DeliveryReport(refused=list(scores), refusal="HTTP 403")


def pause_for(seconds: float) -> Answer:
    def answer(scores: Sequence[PendingScore]) -> DeliveryReport:
        return DeliveryReport(waiting=list(scores), pause_seconds=seconds)

    return answer


def fail(scores: Sequence[PendingScore]) -> DeliveryReport:
    message = "the sender broke"
    raise RuntimeError(message)


@dataclass
class ScriptedSender:
    """Answers each send with the next scripted answer, the last one repeated."""

    answers: list[Answer]
    calls: list[list[PendingScore]] = field(default_factory=list)
    closed: bool = False

    def send(self, scores: Sequence[PendingScore]) -> DeliveryReport:
        self.calls.append(list(scores))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer(scores)

    def close(self) -> None:
        self.closed = True


@dataclass
class WorkerCase:
    worker: ScoreWorker
    clock: FakeClock
    senders: dict[Tracer, ScriptedSender]

    def put(self, *, tracer: Tracer = Tracer.LANGSMITH, count: int = 1) -> list[PendingScore]:
        scores = [
            PendingScore(
                step_id=build_step_id(),
                name="monitor_suspicion",
                value=0.9,
                tracer=tracer,
                project="monitor-scores" if tracer is Tracer.LANGSMITH else None,
                queued_at=self.clock.now,
            )
            for _ in range(count)
        ]
        for score in scores:
            self.worker.put(score)
        return scores

    def read_waiting(self, tracer: Tracer) -> list[PendingScore]:
        return self.worker.waiting.by_tracer.get(tracer, [])


def build_worker_case(
    *,
    langsmith: list[Answer] | None = None,
    langfuse: list[Answer] | None = None,
    timings: WorkerTimings | None = None,
) -> WorkerCase:
    clock = FakeClock()
    senders = {
        Tracer.LANGSMITH: ScriptedSender(answers=langsmith or [write_all]),
        Tracer.LANGFUSE: ScriptedSender(answers=langfuse or [write_all]),
    }

    def build_sender(tracer: Tracer) -> ScoreSender | None:
        return senders[tracer]

    worker = ScoreWorker(
        build_sender=build_sender,
        timings=timings or WorkerTimings(window_seconds=10.0, give_up_seconds=300.0),
        clock=clock.read,
        sleep=clock.sleep,
    )
    return WorkerCase(worker=worker, clock=clock, senders=senders)


def test_one_window_hands_each_tool_all_its_waiting_scores_in_one_call() -> None:
    # Arrange
    case = build_worker_case()
    langsmith_scores = case.put(count=50)
    langfuse_scores = case.put(tracer=Tracer.LANGFUSE, count=3)

    # Act
    case.worker.send_window()

    # Assert
    assert case.senders[Tracer.LANGSMITH].calls == [langsmith_scores]
    assert case.senders[Tracer.LANGFUSE].calls == [langfuse_scores]
    assert case.worker.waiting.count() == 0


def test_a_tool_with_nothing_waiting_is_not_called() -> None:
    # Arrange
    case = build_worker_case()
    case.put(count=2)

    # Act
    case.worker.send_window()
    case.worker.send_window()

    # Assert
    assert len(case.senders[Tracer.LANGSMITH].calls) == 1
    assert case.senders[Tracer.LANGFUSE].calls == []


def test_a_score_that_must_wait_is_sent_again_in_the_next_window() -> None:
    # Arrange
    case = build_worker_case(langfuse=[keep_waiting, write_all])
    [score] = case.put(tracer=Tracer.LANGFUSE)

    # Act
    case.worker.send_window()
    waiting_after_first = list(case.read_waiting(Tracer.LANGFUSE))
    case.clock.now += 10
    case.worker.send_window()

    # Assert
    assert waiting_after_first == [score]
    assert case.senders[Tracer.LANGFUSE].calls == [[score], [score]]
    assert case.worker.waiting.count() == 0


def test_a_pause_holds_the_tool_until_exactly_the_time_it_asked_for() -> None:
    # Arrange
    case = build_worker_case(langsmith=[pause_for(30.0), write_all])
    [score] = case.put()
    case.put(tracer=Tracer.LANGFUSE)
    case.worker.send_window()
    started = case.clock.now

    # Act
    case.clock.now = started + 29.999
    case.worker.send_window()
    calls_during_pause = len(case.senders[Tracer.LANGSMITH].calls)
    case.clock.now = started + 30.0
    case.worker.send_window()

    # Assert: the paused tool waited, and the other tool was never held
    assert calls_during_pause == 1
    assert case.senders[Tracer.LANGSMITH].calls == [[score], [score]]
    assert len(case.senders[Tracer.LANGFUSE].calls) == 1
    assert case.worker.waiting.count() == 0


def test_a_score_is_given_up_exactly_when_it_has_waited_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case(langfuse=[keep_waiting])
    [score] = case.put(tracer=Tracer.LANGFUSE)
    queued = case.clock.now

    # Act
    case.clock.now = queued + 299.999
    case.worker.send_window()
    waiting_before_limit = list(case.read_waiting(Tracer.LANGFUSE))
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.clock.now = queued + 300.0
        case.worker.send_window()

    # Assert
    assert waiting_before_limit == [score]
    assert case.read_waiting(Tracer.LANGFUSE) == []
    assert "gave up on 1 langfuse score(s) after 300 seconds" in caplog.text


def test_refused_scores_are_dropped_with_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    case = build_worker_case(langsmith=[reject_all])
    case.put(count=2)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.send_window()

    # Assert
    assert case.worker.waiting.count() == 0
    assert "langsmith refused 2 score(s): HTTP 403" in caplog.text


def test_a_sender_that_raises_leaves_its_scores_waiting_and_the_other_tool_is_served(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case(langsmith=[fail])
    [score] = case.put()
    case.put(tracer=Tracer.LANGFUSE)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.send_window()

    # Assert
    assert case.read_waiting(Tracer.LANGSMITH) == [score]
    assert case.read_waiting(Tracer.LANGFUSE) == []
    assert "sending to langsmith failed" in caplog.text


def test_a_tool_without_credentials_has_its_scores_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case()
    case.worker.build_sender = lambda tracer: None
    case.put(count=2)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.send_window()

    # Assert
    assert case.worker.waiting.count() == 0
    assert "2 langsmith score(s) dropped: its credentials are missing" in caplog.text


def test_a_sender_that_cannot_be_built_is_logged_once_and_its_scores_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case()
    builds: list[Tracer] = []

    def refuse_to_build(tracer: Tracer) -> ScoreSender | None:
        builds.append(tracer)
        message = "LANGSMITH_API_KEY holds a control or non-ASCII character"
        raise ConfigurationError(message)

    case.worker.build_sender = refuse_to_build
    case.put()

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.send_window()
        case.put()
        case.worker.send_window()

    # Assert
    assert builds == [Tracer.LANGSMITH]
    assert case.worker.waiting.count() == 0
    assert "cannot write to langsmith" in caplog.text


def test_scores_past_the_waiting_limit_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    case = build_worker_case(timings=WorkerTimings(max_waiting=2))
    kept = case.put(count=2)
    case.put(count=1)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.waiting.take_incoming()

    # Assert
    assert case.read_waiting(Tracer.LANGSMITH) == kept
    assert "1 score(s) dropped, since 2 already wait to be sent" in caplog.text


def test_the_drain_sends_what_waits_and_stops_once_nothing_is_left() -> None:
    # Arrange
    case = build_worker_case(langfuse=[keep_waiting, write_all])
    case.put(tracer=Tracer.LANGFUSE)
    case.put()

    # Act
    case.worker.drain()

    # Assert: one window found nothing for Langfuse, one pause later it was written
    assert case.clock.sleeps == [10.0]
    assert len(case.senders[Tracer.LANGFUSE].calls) == 2
    assert case.worker.waiting.count() == 0
    assert all(sender.closed for sender in case.senders.values())


def test_the_drain_stops_exactly_at_its_deadline_and_logs_what_it_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case(
        langfuse=[keep_waiting],
        timings=WorkerTimings(window_seconds=10.0, drain_seconds=25.0),
    )
    case.put(tracer=Tracer.LANGFUSE, count=3)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.drain()

    # Assert: windows at 0, 10, 20 and 25 seconds, then nothing is left waiting
    assert case.clock.sleeps == [10.0, 10.0, 5.0]
    assert len(case.senders[Tracer.LANGFUSE].calls) == 4
    assert case.worker.waiting.count() == 0
    assert "3 langfuse score(s) dropped: the exit drain ran out of time" in caplog.text


def test_a_drain_with_no_time_left_drops_what_waits_without_pausing() -> None:
    # Arrange
    case = build_worker_case(
        langsmith=[keep_waiting],
        timings=WorkerTimings(window_seconds=10.0, drain_seconds=0.0),
    )
    case.put()

    # Act
    case.worker.drain()

    # Assert
    assert case.clock.sleeps == []
    assert case.worker.waiting.count() == 0


def test_stop_drains_a_worker_whose_thread_never_started() -> None:
    # Arrange
    case = build_worker_case()
    case.put()

    # Act
    case.worker.stop()

    # Assert
    assert len(case.senders[Tracer.LANGSMITH].calls) == 1
    assert case.worker.waiting.count() == 0


def test_stop_waits_for_the_thread_to_drain_what_waits() -> None:
    # Arrange: a real thread and clock, with a window far longer than the test
    senders: dict[Tracer, ScriptedSender] = {Tracer.LANGSMITH: ScriptedSender(answers=[write_all])}
    worker = ScoreWorker(
        build_sender=senders.get,
        timings=WorkerTimings(window_seconds=60.0, drain_seconds=5.0),
    )
    worker.start()
    worker.put(
        PendingScore(
            step_id=build_step_id(),
            name="monitor_suspicion",
            value=0.5,
            tracer=Tracer.LANGSMITH,
            project="monitor-scores",
            queued_at=time.monotonic(),
        ),
    )

    # Act
    worker.stop()

    # Assert
    assert worker.thread is not None
    assert not worker.thread.is_alive()
    assert len(senders[Tracer.LANGSMITH].calls) == 1
    assert senders[Tracer.LANGSMITH].closed


def test_a_window_that_fails_is_logged_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case()
    case.put()

    def break_the_window() -> None:
        message = "the queue broke"
        raise RuntimeError(message)

    monkeypatch.setattr(case.worker.waiting, "take_incoming", break_the_window)

    # Act
    with caplog.at_level(logging.WARNING, logger="langchain_sync_monitors.score_worker"):
        case.worker.send_window()

    # Assert
    assert "a send window failed" in caplog.text
