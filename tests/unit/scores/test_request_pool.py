"""The request pool: the calls side by side on daemon threads, or in turn without any."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent import futures
from pathlib import Path

import pytest

from langchain_sync_monitors.request_pool import RequestPool

POOL_LOGGER = "langchain_sync_monitors.request_pool"

ERROR_TEXT = "text a request carried"
"""The text of the error a call raises, which the log must not carry."""

EXIT_SECONDS = 60.0
"""How long a process that leaves a pool open may take to exit; a hung one never does."""

LEAVES_A_POOL_OPEN = """
from langchain_sync_monitors.request_pool import RequestPool

# Not closed: the threads still wait for tasks when the interpreter exits.
pool = RequestPool(size=3)
print(len(pool.threads))
"""


SUBMITS_A_SLOW_CALL_AND_EXITS = """
import sys
import time
from pathlib import Path

from langchain_sync_monitors.request_pool import PROCESS_POOL


def write_after_a_second() -> None:
    time.sleep(1.0)
    Path(sys.argv[1]).write_text("written")


PROCESS_POOL.read_pool().submit(write_after_a_second)
"""


def test_a_pool_of_one_starts_no_thread_and_runs_the_call_in_turn() -> None:
    # Arrange
    pool = RequestPool(size=1)
    callers: list[threading.Thread] = []

    # Act
    future = pool.submit(lambda: callers.append(threading.current_thread()))

    # Assert: the call ran before submit returned, on the caller's thread
    assert pool.threads == []
    assert future.done()
    assert callers == [threading.current_thread()]


def test_every_thread_of_the_pool_is_a_daemon() -> None:
    # Arrange
    pool = RequestPool(size=4)

    # Act
    daemons = [thread.daemon for thread in pool.threads]
    pool.close()

    # Assert
    assert daemons == [True, True, True, True]


def test_a_process_exits_at_once_with_its_pool_left_open() -> None:
    # Arrange: a thread that is not a daemon would hold the interpreter at exit for good
    command = [sys.executable, "-c", LEAVES_A_POOL_OPEN]

    # Act
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=EXIT_SECONDS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the process did not exit within {EXIT_SECONDS:.0f} seconds")

    # Assert
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.split() == ["3"]


def test_closing_the_pool_ends_every_thread_and_later_calls_run_in_turn() -> None:
    # Arrange
    pool = RequestPool(size=3)
    threads = list(pool.threads)

    # Act
    pool.close()
    for thread in threads:
        thread.join(timeout=5.0)
    result = pool.submit(lambda: 30).result(timeout=5.0)

    # Assert
    assert len(threads) == 3
    assert not any(thread.is_alive() for thread in threads)
    assert pool.threads == []
    assert result == 30


@pytest.mark.parametrize("size", [1, 3])
def test_a_submitted_call_gives_its_result_through_its_future(size: int) -> None:
    # Arrange
    pool = RequestPool(size=size)

    # Act
    future = pool.submit(lambda: 20 + 1)
    result = future.result(timeout=5.0)
    pool.close()

    # Assert
    assert result == 21


def test_a_call_submitted_to_a_pool_of_threads_runs_on_one_of_them() -> None:
    # Arrange
    pool = RequestPool(size=3)
    threads = list(pool.threads)

    # Act
    thread = pool.submit(threading.current_thread).result(timeout=5.0)
    pool.close()

    # Assert
    assert thread in threads


@pytest.mark.parametrize("size", [1, 3])
def test_a_submitted_call_that_raises_is_logged_by_type_alone(
    size: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING, logger=POOL_LOGGER)
    pool = RequestPool(size=size)

    def refuse() -> int:
        raise ValueError(ERROR_TEXT)

    # Act
    future = pool.submit(refuse)
    error = future.exception(timeout=5.0)
    # The done callback that logs runs on the pool's thread, after the future is set.
    pool.drain(timeout=5.0)

    # Assert
    assert isinstance(error, ValueError)
    [record] = [record for record in caplog.records if record.name == POOL_LOGGER]
    assert record.getMessage() == "score export: a request failed with ValueError"
    assert record.exc_info is None
    assert ERROR_TEXT not in caplog.text


def test_draining_waits_for_the_calls_submitted_before_it() -> None:
    # Arrange
    pool = RequestPool(size=2)
    finished: list[str] = []

    def finish_after_a_moment() -> None:
        time.sleep(0.3)
        finished.append("done")

    pool.submit(finish_after_a_moment)

    # Act
    pool.drain(timeout=5.0)

    # Assert
    assert finished == ["done"]


def test_draining_gives_up_at_its_timeout_on_a_call_that_hangs() -> None:
    # Arrange
    pool = RequestPool(size=2)
    release = threading.Event()
    pool.submit(lambda: release.wait(timeout=10.0))

    # Act
    started = time.monotonic()
    pool.drain(timeout=0.2)
    waited = time.monotonic() - started
    release.set()

    # Assert
    assert 0.2 <= waited < 3.0


def test_a_process_finishes_the_calls_it_submitted_before_it_exits(tmp_path: Path) -> None:
    # Arrange
    written = tmp_path / "written.txt"
    command = [sys.executable, "-c", SUBMITS_A_SLOW_CALL_AND_EXITS, str(written)]

    # Act
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=EXIT_SECONDS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the process did not exit within {EXIT_SECONDS:.0f} seconds")

    # Assert
    assert finished.returncode == 0, finished.stderr
    assert written.read_text() == "written"


def test_a_call_submitted_while_the_pool_closes_still_completes() -> None:
    # Arrange: another thread closes the pool just as submit is about to queue the call. If the
    # close can finish before the call is queued, the call comes after the pool's stop signals.
    pool = RequestPool(size=2)
    real_tasks = pool.tasks
    interleaved: list[bool] = []

    class ClosingBeforeTheCallIsQueued:
        def put(self, task: Callable[[], None] | None) -> None:
            if task is not None and not interleaved:
                interleaved.append(True)
                closer = threading.Thread(target=pool.close)
                closer.start()
                closer.join(timeout=0.5)
            real_tasks.put(task)

        def get(self) -> Callable[[], None] | None:
            return real_tasks.get()

    pool.tasks = ClosingBeforeTheCallIsQueued()  # ty: ignore[invalid-assignment]

    # Act
    future = pool.submit(lambda: 21)
    _, not_done = futures.wait([future], timeout=5.0)
    pool.drain(timeout=5.0)

    # Assert
    assert not_done == set()
