"""The score worker sends in windows, honours pauses, gives up on old scores and drains at exit.

The worker runs here on a clock the test moves, and its senders are scripted,
so each rule is checked at its exact boundary: a pause ends exactly when
`Retry-After` says, a score is dropped exactly when it has waited
`give_up_seconds`, and the drain stops exactly at its deadline. A few tests
start the real thread, to check that it sends once a window, is a daemon,
and never holds the exit longer than the drain allows.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from langchain_sync_monitors import score_worker
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.score_worker import ScoreWorker, WorkerTimings
from langchain_sync_monitors.scores import DeliveryReport, PendingScore, ScoreSender, Tracer
from tests.support.score_services import FakeClock, build_step_id

type Answer = Callable[[Sequence[PendingScore]], DeliveryReport]

WORKER_LOGGER = "langchain_sync_monitors.score_worker"


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
    fails_to_close: bool = False
    sent: threading.Event = field(default_factory=threading.Event)

    def send(self, scores: Sequence[PendingScore]) -> DeliveryReport:
        self.calls.append(list(scores))
        self.sent.set()
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer(scores)

    def close(self) -> None:
        if self.fails_to_close:
            message = "the connection pool broke"
            raise RuntimeError(message)
        self.closed = True


def build_score(*, tracer: Tracer = Tracer.LANGSMITH, queued_at: float) -> PendingScore:
    return PendingScore(
        step_id=build_step_id(),
        name="monitor_suspicion",
        value=0.9,
        tracer=tracer,
        project="monitor-scores" if tracer is Tracer.LANGSMITH else None,
        queued_at=queued_at,
    )


@dataclass
class WorkerCase:
    worker: ScoreWorker
    clock: FakeClock
    senders: dict[Tracer, ScriptedSender]

    def put(self, *, tracer: Tracer = Tracer.LANGSMITH, count: int = 1) -> list[PendingScore]:
        scores = [build_score(tracer=tracer, queued_at=self.clock.now) for _ in range(count)]
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


def read_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == WORKER_LOGGER]


def read_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    return next(record for record in caplog.records if record.getMessage() == message)


def test_one_window_hands_each_tool_all_its_waiting_scores_in_one_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case()
    langsmith_scores = case.put(count=50)
    langfuse_scores = case.put(tracer=Tracer.LANGFUSE, count=3)

    # Act
    with caplog.at_level(logging.DEBUG, logger=WORKER_LOGGER):
        case.worker.send_window()

    # Assert
    assert case.senders[Tracer.LANGSMITH].calls == [langsmith_scores]
    assert case.senders[Tracer.LANGFUSE].calls == [langfuse_scores]
    assert case.worker.waiting.count() == 0
    assert read_messages(caplog) == [
        "score export: 50 score(s) written to langsmith",
        "score export: 3 score(s) written to langfuse",
    ]


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


def test_the_count_includes_scores_still_on_the_queue() -> None:
    # Arrange
    case = build_worker_case(langsmith=[keep_waiting])
    case.put(count=2)
    case.worker.send_window()

    # Act
    case.put(count=3)

    # Assert
    assert case.worker.waiting.count() == 5


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
    [old] = case.put(tracer=Tracer.LANGFUSE)
    queued = case.clock.now

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.clock.now = queued + 299.999
        case.worker.send_window()
        messages_before_limit = read_messages(caplog)
        [recent] = case.put(tracer=Tracer.LANGFUSE)
        case.clock.now = queued + 300.0
        case.worker.send_window()

    # Assert
    assert messages_before_limit == []
    assert case.read_waiting(Tracer.LANGFUSE) == [recent]
    assert old not in case.read_waiting(Tracer.LANGFUSE)
    assert read_messages(caplog) == [
        "score export: gave up on 1 langfuse score(s) after 300 seconds, since their steps "
        "were not found or not accepted",
    ]


def test_refused_scores_are_dropped_with_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    case = build_worker_case(langsmith=[reject_all])
    case.put(count=2)

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.send_window()

    # Assert
    assert case.worker.waiting.count() == 0
    assert read_messages(caplog) == ["score export: langsmith refused 2 score(s): HTTP 403"]


def test_a_sender_that_raises_leaves_its_scores_waiting_and_the_other_tool_is_served(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case(langsmith=[fail])
    [score] = case.put()
    case.put(tracer=Tracer.LANGFUSE)

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.send_window()

    # Assert
    assert case.read_waiting(Tracer.LANGSMITH) == [score]
    assert case.read_waiting(Tracer.LANGFUSE) == []
    record = read_record(caplog, "score export: sending to langsmith failed")
    exception = record.exc_info
    assert exception
    assert exception[0] is RuntimeError


def test_a_tool_without_credentials_has_its_scores_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    case = build_worker_case()
    case.worker.build_sender = lambda tracer: None
    case.put(count=2)

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.send_window()

    # Assert
    assert case.worker.waiting.count() == 0
    assert read_messages(caplog) == [
        "score export: 2 langsmith score(s) dropped: its credentials are missing"
    ]


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
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.send_window()
        case.put()
        case.worker.send_window()

    # Assert
    assert builds == [Tracer.LANGSMITH]
    assert case.worker.waiting.count() == 0
    assert read_messages(caplog)[0] == (
        "score export: cannot write to langsmith: "
        "LANGSMITH_API_KEY holds a control or non-ASCII character"
    )


def test_scores_past_the_waiting_limit_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    case = build_worker_case(timings=WorkerTimings(max_waiting=2))
    kept = case.put(count=2)
    case.put(count=3)

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.waiting.take_incoming()

    # Assert
    assert case.read_waiting(Tracer.LANGSMITH) == kept
    assert case.worker.waiting.count() == 2
    assert read_messages(caplog) == [
        "score export: 3 score(s) dropped, since 2 already wait to be sent"
    ]


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
        timings=WorkerTimings(window_seconds=10.0, drain_seconds=20.5),
    )
    case.put(tracer=Tracer.LANGFUSE, count=3)

    # Act
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.drain()

    # Assert: windows at 0, 10, 20 and 20.5 seconds, then nothing is left waiting
    assert case.clock.sleeps == [10.0, 10.0, 0.5]
    assert len(case.senders[Tracer.LANGFUSE].calls) == 4
    assert case.worker.waiting.count() == 0
    assert read_messages(caplog) == [
        "score export: 3 langfuse score(s) dropped: the exit drain ran out of time"
    ]
    assert case.senders[Tracer.LANGFUSE].closed


def test_the_drain_keeps_the_deadline_the_exit_hook_set() -> None:
    # Arrange
    case = build_worker_case(langfuse=[keep_waiting], timings=WorkerTimings(drain_seconds=30.0))
    case.put(tracer=Tracer.LANGFUSE)
    case.worker.drain_deadline = case.clock.now + 4.0

    # Act
    case.worker.drain()

    # Assert
    assert case.clock.sleeps == [4.0]
    assert case.worker.waiting.count() == 0


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


def test_the_drain_closes_every_sender_it_built_even_after_one_that_was_not() -> None:
    # Arrange
    case = build_worker_case()
    langfuse = case.senders[Tracer.LANGFUSE]
    case.worker.build_sender = lambda tracer: langfuse if tracer is Tracer.LANGFUSE else None
    case.put()
    case.put(tracer=Tracer.LANGFUSE)

    # Act
    case.worker.drain()

    # Assert
    assert langfuse.closed


def test_a_sender_that_fails_to_close_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    case = build_worker_case()
    case.senders[Tracer.LANGSMITH].fails_to_close = True
    case.put()
    case.put(tracer=Tracer.LANGFUSE)

    # Act
    with caplog.at_level(logging.DEBUG, logger=WORKER_LOGGER):
        case.worker.drain()

    # Assert
    record = read_record(caplog, "score export: closing the langsmith sender failed")
    exception = record.exc_info
    assert exception
    assert exception[0] is RuntimeError
    assert case.senders[Tracer.LANGFUSE].closed


def test_stop_drains_a_worker_whose_thread_never_started_until_its_deadline() -> None:
    # Arrange
    case = build_worker_case(langsmith=[keep_waiting], timings=WorkerTimings(drain_seconds=30.0))
    case.put()

    # Act
    case.worker.stop()

    # Assert
    assert case.clock.sleeps == [10.0, 10.0, 10.0]
    assert len(case.senders[Tracer.LANGSMITH].calls) == 4
    assert case.worker.waiting.count() == 0


def test_the_thread_sends_once_a_window_while_the_agent_runs() -> None:
    # Arrange: a real thread and clock, with a short window
    sender = ScriptedSender(answers=[write_all])
    worker = ScoreWorker(
        build_sender=lambda tracer: sender,
        timings=WorkerTimings(window_seconds=0.01, drain_seconds=5.0),
    )
    worker.start()

    # Act
    worker.put(build_score(queued_at=time.monotonic()))
    sent_before_stop = sender.sent.wait(timeout=5.0)
    worker.stop()

    # Assert
    assert sent_before_stop
    assert worker.thread is not None
    assert worker.thread.daemon
    assert worker.thread.name == "langchain-sync-monitors scores"
    assert not worker.thread.is_alive()
    assert sender.closed


def test_stop_waits_for_the_thread_to_drain_what_waits() -> None:
    # Arrange: a window far longer than the test, so only the drain sends
    sender = ScriptedSender(answers=[write_all])
    worker = ScoreWorker(
        build_sender=lambda tracer: sender,
        timings=WorkerTimings(window_seconds=60.0, drain_seconds=5.0),
    )
    worker.start()
    worker.put(build_score(queued_at=time.monotonic()))

    # Act
    worker.stop()

    # Assert
    assert worker.thread is not None
    assert not worker.thread.is_alive()
    assert len(sender.calls) == 1
    assert sender.closed


def test_stop_never_waits_past_the_drain_limit_for_a_stuck_thread(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: a sender that blocks far longer than the drain and its grace
    monkeypatch.setattr(score_worker, "JOIN_GRACE_SECONDS", 0.3)
    release = threading.Event()

    def block(scores: Sequence[PendingScore]) -> DeliveryReport:
        release.wait(timeout=10.0)
        return DeliveryReport(written=list(scores))

    sender = ScriptedSender(answers=[block])
    worker = ScoreWorker(
        build_sender=lambda tracer: sender,
        timings=WorkerTimings(window_seconds=60.0, drain_seconds=0.2),
    )
    worker.start()
    worker.put(build_score(queued_at=time.monotonic()))

    # Act
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        worker.stop()
    waited = time.monotonic() - started
    release.set()

    # Assert
    assert 0.45 <= waited < 3.0
    assert read_messages(caplog) == [
        "score export: the exit drain ran out of time; 1 score(s) are dropped"
    ]


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
    with caplog.at_level(logging.WARNING, logger=WORKER_LOGGER):
        case.worker.send_window()

    # Assert
    record = read_record(caplog, "score export: a send window failed")
    exception = record.exc_info
    assert exception
    assert exception[0] is RuntimeError
