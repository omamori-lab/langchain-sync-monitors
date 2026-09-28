"""Driving a protocol coroutine to completion without an event loop."""

from __future__ import annotations

import asyncio
import inspect

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
