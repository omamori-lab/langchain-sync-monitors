"""A few daemon threads that send a sender's requests side by side, started up front.

The standard library's `concurrent.futures` does not fit an exit drain. Its
threads are not daemons, so a request still in flight holds the process at
exit, and its own exit hook runs before `atexit`, after which it refuses new
work, so the drain could not use it. The threads here are daemons, started
when the pool is built, and they keep serving during the drain. Python
3.12.0 to 3.12.2 refuse to start a thread once the interpreter shuts down,
`atexit` hooks included, which 3.12.3 allows again [@cpython2026]; a pool
that cannot start its threads runs each call in turn.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


def run_safely[Item, Result](call: Callable[[Item], Result], *, item: Item) -> Result | None:
    """Return the call's result, or None, logged, when it raises."""
    try:
        return call(item)
    except Exception:
        logger.warning("score export: a request failed", exc_info=True)
        return None


class RequestPool:
    """Runs a call on several items at once, on up to `size` daemon threads.

    A `size` of 1 starts no thread, and runs the calls in turn.
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

    @property
    def width(self) -> int:
        """How many calls run at once: the number of threads, or one without any."""
        return max(len(self.threads), 1)

    def serve(self) -> None:
        """Run tasks until the pool closes."""
        while (task := self.tasks.get()) is not None:
            task()

    def run_all[Item, Result](
        self,
        call: Callable[[Item], Result],
        *,
        items: Sequence[Item],
    ) -> list[Result | None]:
        """Return the call's result for each item, in order; None where it raised."""
        if not self.threads:
            return [run_safely(call, item=item) for item in items]
        results: list[Result | None] = [None] * len(items)
        finished: queue.SimpleQueue[int] = queue.SimpleQueue()

        def build_task(position: int, *, item: Item) -> Callable[[], None]:
            def task() -> None:
                results[position] = run_safely(call, item=item)
                finished.put(position)

            return task

        for position, item in enumerate(items):
            self.tasks.put(build_task(position, item=item))
        for _ in items:
            finished.get()
        return results

    def close(self) -> None:
        """Let every thread finish once the tasks before it are done; later calls run in turn."""
        for _ in self.threads:
            self.tasks.put(None)
        self.threads = []
