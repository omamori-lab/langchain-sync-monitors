"""The request pool: the calls side by side on daemon threads, or in turn without any."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading

import pytest

from langchain_sync_monitors.request_pool import RequestPool

POOL_LOGGER = "langchain_sync_monitors.request_pool"

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


def multiply_or_refuse(item: int) -> int:
    """Return ten times the item, or raise for the item 2."""
    if item == 2:
        message = "refused"
        raise ValueError(message)
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

    # Assert
    assert results == [[10, None, 30]]
    assert "score export: a request failed" in caplog.messages


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
