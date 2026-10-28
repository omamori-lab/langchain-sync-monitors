"""A real fork: the child forgets the locks its parent's threads held, and runs calls on a pool.

Each test forks the test process. The child runs each check on a thread of
its own, which must end within a second, and leaves through `os._exit` with a
code that names the first check that failed; an alarm ends a child that hangs
on its main thread. A child that a crash kills shows as a signal.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable, Sequence
from typing import NoReturn

import pytest

from langchain_sync_monitors.request_pool import PROCESS_POOL
from langchain_sync_monitors.score_export import PROCESS_NOTICES

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork"),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning"),
]

CHILD_ALARM_SECONDS = 10
"""How long a forked child may live before its alarm ends it."""

CHECK_SECONDS = 1.0
"""How long each check in the child may take on its thread."""


def is_check_passed_in_time(check: Callable[[], bool]) -> bool:
    """Run the check on a daemon thread; tell whether it returned True within a second."""
    results: list[bool] = []
    thread = threading.Thread(target=lambda: results.append(check()), daemon=True)
    thread.start()
    thread.join(timeout=CHECK_SECONDS)
    return results == [True]


def exit_forked_child(checks: Sequence[Callable[[], bool]]) -> NoReturn:
    """In the child, exit 0 when every check passes, 10 plus the first failure's index if not."""
    code = 1
    try:
        signal.alarm(CHILD_ALARM_SECONDS)
        outcomes = (is_check_passed_in_time(check) for check in checks)
        code = next((10 + index for index, passed in enumerate(outcomes) if not passed), 0)
    finally:
        os._exit(code)


def describe_exit(status: int) -> str:
    """Return how a child ended: its exit code, or the signal that killed it."""
    if os.WIFSIGNALED(status):
        return f"signal {os.WTERMSIG(status)}"
    return f"exit {os.WEXITSTATUS(status)}"


def fork_and_check(checks: Sequence[Callable[[], bool]]) -> str:
    """Fork, run the checks in the child, and return how the child ended."""
    process_id = os.fork()
    if process_id == 0:
        exit_forked_child(checks)
    _, status = os.waitpid(process_id, 0)
    return describe_exit(status)


def is_call_run_on_the_pool() -> bool:
    """Submit a call to the process's pool, which must run it on a thread of the child's own."""
    return PROCESS_POOL.read_pool().submit(lambda: 7).result(timeout=CHECK_SECONDS / 2) == 7


def is_notice_said() -> bool:
    """Say a notice once, which takes the notices' lock."""
    PROCESS_NOTICES.warn_once("score export: a notice said in a forked child")
    return True


def test_a_forked_child_runs_calls_on_a_pool_of_its_own_though_the_parents_locks_were_held() -> (
    None
):
    # Arrange: the parent has a pool whose threads the child does not inherit, and another
    # thread holds the process pool's and the notices' locks at the fork
    PROCESS_POOL.read_pool()
    held, release = threading.Event(), threading.Event()

    def hold_both_locks() -> None:
        with PROCESS_POOL.lock, PROCESS_NOTICES.lock:
            held.set()
            release.wait(timeout=CHILD_ALARM_SECONDS)

    holder = threading.Thread(target=hold_both_locks, daemon=True)
    holder.start()
    assert held.wait(timeout=5.0)

    # Act
    try:
        ending = fork_and_check([is_call_run_on_the_pool, is_notice_said])
    finally:
        release.set()
        holder.join(timeout=5.0)

    # Assert: exit 10 means the child's pool did not run the call in time, 11 the notices' lock
    assert ending == "exit 0"
