"""The background worker that writes the monitors' suspicion scores, one per process.

A monitor puts each score on the worker's queue and carries on; putting never
blocks and never raises. The worker's thread wakes once a window, takes what
was queued, and hands each tool all of its waiting scores at once, so a
window costs a tool one lookup however many steps wait. A score the tool
cannot take yet, such as a step it has not ingested, waits for the next
window, and is dropped, with a warning, once it has waited
`give_up_seconds`. A tool that answers `429` gets no call until the pause
it asked for is over.

At exit, the worker sends what still waits, once every `drain_window_seconds`,
a shorter window that finds a step Langfuse has just ingested sooner, until
nothing waits or `drain_seconds` have passed, and logs what it drops. The
thread is a daemon, so a drain that overruns never keeps the process alive.
Failures, a sender's exceptions included, are logged and never reach a run.
The first time a tool's scores are dropped because their steps were never
found, the worker also says, once, what usually causes it.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.scores import DeliveryReport, PendingScore, ScoreSender, Tracer

logger = logging.getLogger(__name__)

WINDOW_SECONDS: Final = 10.0
"""How long the worker gathers scores before it sends them, and how often it asks again."""

GIVE_UP_SECONDS: Final = 300.0
"""How long a score may wait for its step to be ingested before it is dropped."""

DRAIN_SECONDS: Final = 30.0
"""How long the exit drain may keep sending."""

DRAIN_WINDOW_SECONDS: Final = 5.0
"""How often the exit drain sends, and so looks for the steps Langfuse has just ingested."""

MAX_WAITING_SCORES: Final = 10_000
"""The most scores that wait at once; more are dropped, with a warning."""

JOIN_GRACE_SECONDS: Final = 1.0
"""How long the exit hook waits for the thread past the drain's limit before it gives up."""

type SenderFactory = Callable[[Tracer], ScoreSender | None]
"""Builds the sender of one tool, or returns None when its credentials are missing."""

UNFOUND_STEP_HINTS: Final = {
    Tracer.LANGFUSE: (
        "score export: Langfuse scores go only on steps found in the project that "
        "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY reach. A Langfuse handler built with "
        "other keys, or another host, traces to a project whose steps are never scored"
    ),
}
"""What usually leaves a tool's steps unfound, said once per process when its scores are dropped."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerTimings:
    """How often the worker sends, how long a score may wait, and how long the exit drain lasts."""

    window_seconds: float = WINDOW_SECONDS
    give_up_seconds: float = GIVE_UP_SECONDS
    drain_seconds: float = DRAIN_SECONDS
    drain_window_seconds: float = DRAIN_WINDOW_SECONDS
    max_waiting: int = MAX_WAITING_SCORES


class WaitingScores:
    """The scores that wait to be written, by tool: queued by monitors, taken in by the worker.

    `put` may be called from any thread. Everything else runs on the worker's
    thread, or under its window lock.
    """

    def __init__(self, *, max_waiting: int) -> None:
        self.max_waiting = max_waiting
        self.incoming: queue.SimpleQueue[PendingScore] = queue.SimpleQueue()
        self.by_tracer: dict[Tracer, list[PendingScore]] = {}

    def put(self, score: PendingScore) -> None:
        """Queue a score, without blocking."""
        self.incoming.put(score)

    def take_incoming(self) -> None:
        """Move the queued scores to their tools' lists, dropping those past `max_waiting`."""
        waiting = sum(len(scores) for scores in self.by_tracer.values())
        dropped = 0
        while True:
            try:
                score = self.incoming.get_nowait()
            except queue.Empty:
                break
            if waiting >= self.max_waiting:
                dropped += 1
                continue
            self.by_tracer.setdefault(score.tracer, []).append(score)
            waiting += 1
        if dropped:
            logger.warning(
                "score export: %d score(s) dropped, since %d already wait to be sent",
                dropped,
                self.max_waiting,
            )

    def give_up_on_old(self, *, now: float, limit_seconds: float) -> list[Tracer]:
        """Drop, with a warning, each score that has waited `limit_seconds` or longer.

        Return the tools that lost scores.
        """
        gave_up: list[Tracer] = []
        for tracer, scores in self.by_tracer.items():
            kept = [score for score in scores if now - score.queued_at < limit_seconds]
            if len(kept) < len(scores):
                gave_up.append(tracer)
                logger.warning(
                    "score export: gave up on %d %s score(s) after %.0f seconds, since "
                    "their steps were not found or not accepted",
                    len(scores) - len(kept),
                    tracer,
                    limit_seconds,
                )
            self.by_tracer[tracer] = kept
        return gave_up

    def drop(self, tracer: Tracer, *, reason: str) -> int:
        """Drop the tool's waiting scores, with a warning that gives the reason; return how many."""
        scores = self.by_tracer.pop(tracer, [])
        if scores:
            logger.warning("score export: %d %s score(s) dropped: %s", len(scores), tracer, reason)
        return len(scores)

    def count(self) -> int:
        """Return how many scores wait, queued ones included."""
        return sum(len(scores) for scores in self.by_tracer.values()) + self.incoming.qsize()


class ScoreWorker:
    """Writes queued scores to their tools from one daemon thread, in windows.

    `build_sender` builds each tool's sender the first time a score for it
    is sent. `clock` and `sleep` are the monotonic clock and the pause the
    worker uses, so a test can drive windows without waiting. Everything the
    worker holds is process state: no monitor keeps any of it.
    """

    def __init__(
        self,
        *,
        build_sender: SenderFactory,
        timings: WorkerTimings | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.build_sender = build_sender
        self.timings = timings or WorkerTimings()
        self.clock = clock
        self.sleep = sleep
        self.waiting = WaitingScores(max_waiting=self.timings.max_waiting)
        self.senders: dict[Tracer, ScoreSender | None] = {}
        self.explained: set[Tracer] = set()
        self.paused_until: dict[Tracer, float] = {}
        self.window_lock = threading.Lock()
        self.stopping = threading.Event()
        self.drain_deadline: float | None = None
        self.thread: threading.Thread | None = None

    def put(self, score: PendingScore) -> None:
        """Queue a score for the next window, without blocking."""
        self.waiting.put(score)

    def start(self) -> None:
        """Start the worker's daemon thread."""
        self.thread = threading.Thread(
            target=self.run,
            name="langchain-sync-monitors scores",
            daemon=True,
        )
        self.thread.start()

    def run(self) -> None:
        """Send once a window until asked to stop, then drain."""
        while not self.stopping.wait(self.timings.window_seconds):
            self.send_window()
        self.drain()

    def stop(self) -> None:
        """Drain at exit, within `drain_seconds`, and log what could not be sent.

        The thread drains while this waits for it. A worker whose thread
        never started drains here.
        """
        self.drain_deadline = self.clock() + self.timings.drain_seconds
        self.stopping.set()
        if self.thread is None:
            self.drain()
            return
        self.thread.join(timeout=self.timings.drain_seconds + JOIN_GRACE_SECONDS)
        if self.thread.is_alive():
            logger.warning(
                "score export: the exit drain ran out of time; %d score(s) are dropped",
                self.waiting.count(),
            )

    def drain(self) -> None:
        """Send once a window until nothing waits or the deadline passes; drop what is left."""
        deadline = self.drain_deadline
        if deadline is None:
            deadline = self.clock() + self.timings.drain_seconds
        while True:
            self.send_window()
            if self.waiting.count() == 0:
                break
            remaining = deadline - self.clock()
            if remaining <= 0:
                self.drop_every_score(reason="the exit drain ran out of time")
                break
            self.sleep(min(self.timings.drain_window_seconds, remaining))
        self.close_senders()

    def send_window(self) -> None:
        """Take the queued scores, send each tool's waiting scores once, and give up on old ones."""
        with self.window_lock:
            try:
                self.waiting.take_incoming()
                for tracer in list(self.waiting.by_tracer):
                    self.send_waiting(tracer)
                gave_up = self.waiting.give_up_on_old(
                    now=self.clock(), limit_seconds=self.timings.give_up_seconds
                )
                self.explain_unfound_steps(gave_up)
            except Exception:
                logger.warning("score export: a send window failed", exc_info=True)

    def send_waiting(self, tracer: Tracer) -> None:
        """Hand the tool all its waiting scores at once, unless it asked for a pause."""
        scores = self.waiting.by_tracer.get(tracer, [])
        if not scores or self.clock() < self.paused_until.get(tracer, -math.inf):
            return
        sender = self.read_sender(tracer)
        if sender is None:
            self.waiting.drop(tracer, reason="its credentials are missing")
            return
        try:
            report = sender.send(scores)
        except Exception:
            logger.warning("score export: sending to %s failed", tracer, exc_info=True)
            return
        self.apply_report(tracer, report=report)

    def apply_report(self, tracer: Tracer, *, report: DeliveryReport) -> None:
        """Keep the scores that must wait, log the refused ones, and pause the tool if asked."""
        self.waiting.by_tracer[tracer] = report.waiting
        if report.refused:
            logger.warning(
                "score export: %s refused %d score(s): %s",
                tracer,
                len(report.refused),
                report.refusal,
            )
        if report.written:
            logger.debug("score export: %d score(s) written to %s", len(report.written), tracer)
        if report.pause_seconds is not None:
            self.paused_until[tracer] = self.clock() + report.pause_seconds

    def read_sender(self, tracer: Tracer) -> ScoreSender | None:
        """Return the tool's sender, built the first time; None when it cannot be built."""
        if tracer not in self.senders:
            try:
                self.senders[tracer] = self.build_sender(tracer)
            except ConfigurationError as error:
                logger.warning("score export: cannot write to %s: %s", tracer, error)
                self.senders[tracer] = None
        return self.senders[tracer]

    def drop_every_score(self, *, reason: str) -> None:
        """Drop every waiting score, queued ones included, with a warning per tool."""
        with self.window_lock:
            self.waiting.take_incoming()
            dropped = [
                tracer
                for tracer in list(self.waiting.by_tracer)
                if self.waiting.drop(tracer, reason=reason)
            ]
            self.explain_unfound_steps(dropped)

    def explain_unfound_steps(self, tracers: list[Tracer]) -> None:
        """Say, once per tool and process, what usually leaves its steps unfound."""
        for tracer in tracers:
            hint = UNFOUND_STEP_HINTS.get(tracer)
            if hint is not None and tracer not in self.explained:
                self.explained.add(tracer)
                logger.warning(hint)

    def close_senders(self) -> None:
        """Close every sender's HTTP client; a failure to close is logged."""
        for tracer, sender in self.senders.items():
            if sender is None:
                continue
            try:
                sender.close()
            except Exception:
                logger.debug("score export: closing the %s sender failed", tracer, exc_info=True)
