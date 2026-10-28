"""The request pool: the calls side by side on daemon threads, or in turn without any."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from langchain_sync_monitors.request_pool import RequestPool

POOL_LOGGER = "langchain_sync_monitors.request_pool"

ERROR_TEXT = "text a request carried"
"""The text of the error a call raises, which the log must not carry."""

EXIT_SECONDS = 60.0
"""How long a process that leaves a pool open may take to exit; a hung one never does."""

LEAVES_POOLS_OPEN = """
from langchain_sync_monitors.langsmith_scores import LangSmithFeedbackSender
from langchain_sync_monitors.request_pool import RequestPool

# Neither is closed: the threads still wait for tasks when the interpreter exits.
sender = LangSmithFeedbackSender()
pool = RequestPool(size=3)
print(len(sender.pool.threads), len(pool.threads))
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


def multiply_or_refuse(item: int) -> int:
    """Return ten times the item, or raise for the item 2."""
    if item == 2:
        raise ValueError(ERROR_TEXT)
    return item * 10


def test_a_pool_of_one_starts_no_thread_and_runs_the_calls_in_turn() -> None:
    # Arrange
    pool = RequestPool(size=1)

    # Act
    results = pool.run_all(multiply_or_refuse, items=[1, 3])

    # Assert
    assert pool.threads == []
    assert pool.width == 1
    assert results == [10, 30]


@pytest.mark.parametrize("size", [1, 3])
def test_a_call_that_raises_gives_none_logged_and_the_others_their_results(
    size: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the calls run on a thread of their own, so that a hung pool fails the test
    caplog.set_level(logging.WARNING, logger=POOL_LOGGER)
    pool = RequestPool(size=size)
    results: list[list[int | None]] = []
    runner = threading.Thread(
        target=lambda: results.append(pool.run_all(multiply_or_refuse, items=[1, 2, 3])),
        daemon=True,
    )

    # Act
    runner.start()
    runner.join(timeout=5.0)
    pool.close()

    # Assert: the log names the error's type, never its text or its traceback
    assert results == [[10, None, 30]]
    [record] = [record for record in caplog.records if record.name == POOL_LOGGER]
    assert record.getMessage() == "score export: a request failed with ValueError"
    assert record.exc_info is None
    assert ERROR_TEXT not in caplog.text


def test_every_thread_of_the_pool_is_a_daemon() -> None:
    # Arrange
    pool = RequestPool(size=4)

    # Act
    daemons = [thread.daemon for thread in pool.threads]
    pool.close()

    # Assert
    assert daemons == [True, True, True, True]


def test_a_process_exits_at_once_with_its_pools_left_open() -> None:
    # Arrange: a thread that is not a daemon would hold the interpreter at exit for good
    command = [sys.executable, "-c", LEAVES_POOLS_OPEN]

    # Act
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=EXIT_SECONDS
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the process did not exit within {EXIT_SECONDS:.0f} seconds")

    # Assert
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.split() == ["6", "3"]


def test_closing_the_pool_ends_every_thread_and_later_calls_run_in_turn() -> None:
    # Arrange
    pool = RequestPool(size=3)
    threads = list(pool.threads)

    # Act
    pool.close()
    for thread in threads:
        thread.join(timeout=5.0)

    # Assert
    assert len(threads) == 3
    assert not any(thread.is_alive() for thread in threads)
    assert pool.threads == []
    assert pool.run_all(multiply_or_refuse, items=[1, 3]) == [10, 30]


@pytest.mark.xfail(strict=True, reason="PR 7: the pool submits calls")
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


@pytest.mark.xfail(strict=True, reason="PR 7: the pool submits calls")
def test_a_call_submitted_to_a_pool_of_threads_runs_on_one_of_them() -> None:
    # Arrange
    pool = RequestPool(size=3)
    threads = list(pool.threads)

    # Act
    thread = pool.submit(threading.current_thread).result(timeout=5.0)
    pool.close()

    # Assert
    assert thread in threads


@pytest.mark.xfail(strict=True, reason="PR 7: the pool submits calls")
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
    pool.close()

    # Assert
    assert isinstance(error, ValueError)
    [record] = [record for record in caplog.records if record.name == POOL_LOGGER]
    assert record.getMessage() == "score export: a request failed with ValueError"
    assert record.exc_info is None
    assert ERROR_TEXT not in caplog.text


@pytest.mark.xfail(strict=True, reason="PR 7: the pool drains")
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


@pytest.mark.xfail(strict=True, reason="PR 7: the pool drains")
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


@pytest.mark.xfail(strict=True, reason="PR 7: the process pool drains at exit")
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
