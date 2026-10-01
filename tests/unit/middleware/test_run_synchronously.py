"""Driving a protocol coroutine to completion without an event loop."""

from __future__ import annotations

import asyncio
import gc
import inspect
import threading
from collections.abc import Callable, Coroutine

import pytest

from langchain_sync_monitors.errors import SynchronousRunError
from langchain_sync_monitors.pending_steps import run_synchronously


async def return_after_awaiting_plain_coroutines() -> str:
    async def inner() -> str:
        return "decided"

    return await inner()


async def sleep_then_return(*, delay: float, cleanups: list[str]) -> str:
    try:
        await asyncio.sleep(delay)
    finally:
        cleanups.append("closed")
    return "unreachable"


async def raise_runtime_error() -> str:
    message = "a protocol bug"
    raise RuntimeError(message)


def test_a_coroutine_that_never_suspends_returns_its_result() -> None:
    # Act
    result = run_synchronously(return_after_awaiting_plain_coroutines())

    # Assert
    assert result == "decided"


def test_a_coroutine_that_yields_to_the_loop_is_closed_and_raises() -> None:
    # Arrange
    cleanups: list[str] = []
    coroutine = sleep_then_return(delay=0, cleanups=cleanups)

    # Act
    with pytest.raises(SynchronousRunError, match="ainvoke"):
        run_synchronously(coroutine)

    # Assert
    assert cleanups == ["closed"]
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_real_async_work_without_a_running_loop_raises_the_library_error() -> None:
    # Arrange
    cleanups: list[str] = []

    # Act
    with pytest.raises(SynchronousRunError) as raised:
        run_synchronously(sleep_then_return(delay=0.01, cleanups=cleanups))

    # Assert
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "no running event loop" in str(raised.value.__cause__)
    assert cleanups == ["closed"]


async def test_real_async_work_inside_a_running_loop_raises_instead_of_hanging() -> None:
    # Arrange
    cleanups: list[str] = []

    # Act
    with pytest.raises(SynchronousRunError):
        run_synchronously(sleep_then_return(delay=0.01, cleanups=cleanups))

    # Assert
    assert cleanups == ["closed"]


def test_other_runtime_errors_propagate_unchanged() -> None:
    # Act
    with pytest.raises(RuntimeError, match="a protocol bug") as raised:
        run_synchronously(raise_runtime_error())

    # Assert
    assert not isinstance(raised.value, SynchronousRunError)


async def record_call(calls: list[str]) -> str:
    calls.append("called")
    return "called"


async def gather_two_calls(calls: list[str]) -> str:
    await asyncio.gather(record_call(calls), record_call(calls))
    return "unreachable"


async def create_a_task(calls: list[str]) -> str:
    await asyncio.create_task(record_call(calls))
    return "unreachable"


async def ensure_a_future(calls: list[str]) -> str:
    await asyncio.ensure_future(record_call(calls))
    return "unreachable"


SCHEDULING_PROTOCOLS = [gather_two_calls, create_a_task, ensure_a_future]


def run_in_fresh_thread(work: Callable[[], object]) -> type[BaseException] | None:
    """Run `work` in a new thread, which has no event loop, and return the type it raised.

    Only the type leaves the thread, so the unawaited coroutines the error's
    traceback holds can be collected, and warned about, inside the test.
    """
    raised: list[type[BaseException]] = []

    def target() -> None:
        try:
            work()
        except BaseException as error:
            raised.append(type(error))

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return raised[0] if raised else None


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
@pytest.mark.parametrize("schedule", SCHEDULING_PROTOCOLS)
def test_scheduling_work_where_no_loop_runs_raises_the_library_error(
    schedule: Callable[[list[str]], Coroutine[object, object, str]],
) -> None:
    # Arrange
    calls: list[str] = []

    # Act
    raised = run_in_fresh_thread(lambda: run_synchronously(schedule(calls)))
    gc.collect()

    # Assert
    assert raised is SynchronousRunError
    assert calls == []


@pytest.mark.parametrize("schedule", SCHEDULING_PROTOCOLS)
async def test_work_scheduled_inside_a_running_loop_is_cancelled_before_it_runs(
    schedule: Callable[[list[str]], Coroutine[object, object, str]],
) -> None:
    # Arrange
    calls: list[str] = []

    # Act
    with pytest.raises(SynchronousRunError):
        run_synchronously(schedule(calls))
    await asyncio.sleep(0.05)

    # Assert
    assert calls == []


async def schedule_then_fail(scheduled: list[asyncio.Future[str]], *, error: Exception) -> str:
    scheduled.append(asyncio.ensure_future(record_call([])))
    raise error


STEP_ERROR = "The monitor started asynchronous work in evaluate_sync."
FAILURES_AFTER_SCHEDULING: dict[str, tuple[Callable[[], Exception], str]] = {
    "the-pending-step-s-own-error": (lambda: SynchronousRunError(STEP_ERROR), STEP_ERROR),
    "asyncio-s-missing-loop-error": (
        lambda: RuntimeError("no running event loop"),
        "A control protocol awaited real asynchronous work during a synchronous invoke(). "
        "Under invoke() a protocol may await only the pending step's own methods; "
        "run the agent with ainvoke() to use anything else.",
    ),
}
"""How a protocol can fail after it scheduled work, and what the error that ends the call says."""


@pytest.mark.parametrize(
    ("build_error", "expected_message"),
    FAILURES_AFTER_SCHEDULING.values(),
    ids=FAILURES_AFTER_SCHEDULING.keys(),
)
async def test_work_scheduled_before_a_synchronous_run_error_is_cancelled_before_it_runs(
    build_error: Callable[[], Exception],
    expected_message: str,
) -> None:
    # Arrange
    scheduled: list[asyncio.Future[str]] = []

    # Act
    with pytest.raises(SynchronousRunError) as raised:
        run_synchronously(schedule_then_fail(scheduled, error=build_error()))
    await asyncio.sleep(0.05)

    # Assert
    assert str(raised.value) == expected_message
    [task] = scheduled
    assert task.cancelled()
