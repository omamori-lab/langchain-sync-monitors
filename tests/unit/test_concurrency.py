"""The library's concurrent calls raise the first failure and log any other by its type alone."""

from __future__ import annotations

import asyncio
import logging

import pytest

from langchain_sync_monitors.concurrency import run_concurrently
from tests.unit.monitors.doubles import PLANTED_SECRET

LOGGER = "langchain_sync_monitors.concurrency"


async def fail(text: str) -> str:
    """Raise at once, so every call fails before the task group cancels the others."""
    raise RuntimeError(text)


async def return_text(text: str) -> str:
    return text


def test_a_second_failure_is_logged_by_its_type_and_never_by_its_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the second error quotes the request, as a provider's error can
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    calls = [fail("first call failed"), fail(f"second call sent {PLANTED_SECRET}")]

    # Act
    with pytest.raises(RuntimeError, match="first call failed"):
        asyncio.run(run_concurrently(calls))

    # Assert
    assert [record.getMessage() for record in caplog.records] == [
        "A concurrent call also failed, with RuntimeError.",
    ]
    assert PLANTED_SECRET not in caplog.text


def test_calls_that_all_succeed_return_in_order_and_log_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    # Act
    results = asyncio.run(run_concurrently([return_text("one"), return_text("two")]))

    # Assert
    assert results == ["one", "two"]
    assert caplog.records == []
