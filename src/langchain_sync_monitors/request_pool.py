"""A few daemon threads that run a request off the caller's thread, started up front.

The standard library's `concurrent.futures` does not fit an exit drain. Its
threads are not daemons, so a request still in flight holds the process at
exit, and its own exit hook runs before `atexit`, after which it refuses new
work, so the drain could not use it. The threads here are daemons, started
when the pool is built, and they keep serving during the drain. Python
3.12.0 to 3.12.2 refuse to start a thread once the interpreter shuts down,
`atexit` hooks included, which 3.12.3 allows again [@cpython2026]; a pool
that cannot start its threads runs each call in turn.

`PROCESS_POOL` is the pool of the process. It starts with the first call
submitted, and at exit it keeps running the calls submitted for up to
`EXIT_DRAIN_SECONDS`. A child forked from the process forgets the parent's
pool and lock, whose threads and holder did not come with it, and starts a
pool of its own.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SEND_THREADS = 4
"""How many requests the process pool runs at once."""

EXIT_DRAIN_SECONDS = 30.0
"""How long the process pool keeps running the calls submitted, once the process exits."""


def run_in_future[Result](future: Future[Result], *, call: Callable[[], Result]) -> None:
    """Run the call and hand its result, or the error it raised, to the future."""
    if not future.set_running_or_notify_cancel():
        return
    try:
        result = call()
    except Exception as error:
        future.set_exception(error)
    else:
        future.set_result(result)


def log_failed_request[Result](future: Future[Result]) -> None:
    """Log a request that raised, by the error's type alone."""
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.warning("score export: a request failed with %s", type(error).__name__)


class RequestPool:
    """Runs submitted calls on up to `size` daemon threads.

    A `size` of 1 starts no thread, and runs each call in turn, in the
    caller's thread.
    """

    def __init__(self, *, size: int) -> None:
        self.tasks: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self.threads: list[threading.Thread] = []
        for _ in range(size if size > 1 else 0):
            thread = threading.Thread(
                target=self.serve,
                name="langchain-sync-monitors requests",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError:
                # The interpreter is shutting down: run the calls in turn.
                break
            self.threads.append(thread)

    def serve(self) -> None:
        """Run tasks until the pool closes."""
        while (task := self.tasks.get()) is not None:
            task()

    def submit[Result](self, call: Callable[[], Result], /) -> Future[Result]:
        """Run the call on a thread of the pool, and return a future for its result.

        A call that raises is logged by the error's type alone, and the
        future holds the error.
        """
        future: Future[Result] = Future()
        future.add_done_callback(log_failed_request)
        if self.threads:
            self.tasks.put(lambda: run_in_future(future, call=call))
        else:
            run_in_future(future, call=call)
        return future

    def close(self) -> None:
        """Let every thread finish once the tasks before it are done; later calls run in turn."""
        for _ in self.threads:
            self.tasks.put(None)
        self.threads = []

    def drain(self, *, timeout: float) -> None:
        """Close the pool, and wait up to `timeout` seconds for the calls already submitted."""
        threads = self.threads
        self.close()
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(timeout=max(deadline - time.monotonic(), 0.0))


@dataclass(slots=True, kw_only=True)
class ProcessRequestPool:
    """The one request pool of this process, started with the first call submitted.

    The lock is taken only to start the pool, never to submit a call.
    """

    pool: RequestPool | None = None
    process_id: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def read_pool(self) -> RequestPool:
        """Return the process's pool, starting it and its exit drain the first time."""
        pool = self.pool
        if pool is not None and self.process_id == os.getpid():
            return pool
        with self.lock:
            # Another thread may have started the pool while this one waited for the lock.
            if self.pool is None or self.process_id != os.getpid():
                self.pool = self.start_pool()
            return self.pool

    def start_pool(self) -> RequestPool:
        """Start a pool for this process, with its exit drain in place of any older one's.

        The caller holds the lock.
        """
        if self.pool is not None:
            atexit.unregister(self.pool.drain)
        pool = RequestPool(size=SEND_THREADS)
        atexit.register(pool.drain, timeout=EXIT_DRAIN_SECONDS)
        self.process_id = os.getpid()
        return pool

    def forget_after_fork(self) -> None:
        """In a forked child, drop the parent's pool, its exit drain and its lock."""
        if self.pool is not None:
            atexit.unregister(self.pool.drain)
        self.lock = threading.Lock()
        self.pool = None
        self.process_id = None


PROCESS_POOL = ProcessRequestPool()
"""The request pool of this process."""
