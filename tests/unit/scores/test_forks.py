"""A real fork: the child forgets the locks its parent's threads held, and builds clients safely.

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

import httpx
import pytest

from langchain_sync_monitors.score_export import PROCESS_NOTICES, PROCESS_SCORE_WORKER
from langchain_sync_monitors.score_requests import PROCESS_ORIGIN, build_http_client
from langchain_sync_monitors.score_worker import ScoreWorker

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork"),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning"),
]

CHILD_ALARM_SECONDS = 10
"""How long a forked child may live before its alarm ends it."""

CHECK_SECONDS = 1.0
"""How long each check in the child may take on its thread."""

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


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


def is_worker_read() -> bool:
    """Read the process's score worker, which takes its lock when none runs yet."""
    return PROCESS_SCORE_WORKER.read_worker() is not None


def is_notice_said() -> bool:
    """Say a notice once, which takes the notices' lock."""
    PROCESS_NOTICES.warn_once("score export: a notice said in a forked child")
    return True


def is_marked_forked() -> bool:
    """Tell whether the at-fork hook marked the process as forked."""
    return PROCESS_ORIGIN.forked


def is_client_built() -> bool:
    """Build and close a client as the score senders build theirs."""
    build_http_client(base_url="https://service.test").close()
    return True


def test_a_forked_child_forgets_the_locks_its_parents_threads_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: no worker runs yet, and another thread holds the worker's and the notices' locks
    monkeypatch.setattr(
        PROCESS_SCORE_WORKER, "build_worker", lambda: ScoreWorker(build_sender=lambda tracer: None)
    )
    monkeypatch.setattr(PROCESS_SCORE_WORKER, "worker", None)
    held, release = threading.Event(), threading.Event()

    def hold_both_locks() -> None:
        with PROCESS_SCORE_WORKER.lock, PROCESS_NOTICES.lock:
            held.set()
            release.wait(timeout=CHILD_ALARM_SECONDS)

    holder = threading.Thread(target=hold_both_locks, daemon=True)
    holder.start()
    assert held.wait(timeout=5.0)

    # Act
    try:
        ending = fork_and_check([is_worker_read, is_notice_said, is_marked_forked])
    finally:
        release.set()
        holder.join(timeout=5.0)

    # Assert: exit 10 means the worker's lock hung, 11 the notices' lock, 12 no fork mark
    assert ending == "exit 0"


def test_a_forked_child_builds_its_clients_after_its_parent_read_the_system_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: no proxy variable; the parent has a thread and has read the system's proxies
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    httpx.Client().close()
    stop = threading.Event()
    idle = threading.Thread(target=stop.wait, daemon=True)
    idle.start()

    # Act
    try:
        ending = fork_and_check([is_client_built, is_marked_forked])
    finally:
        stop.set()
        idle.join(timeout=5.0)

    # Assert: on macOS, the system proxy lookup killed such a child with SIGSEGV or SIGABRT
    assert ending == "exit 0"
