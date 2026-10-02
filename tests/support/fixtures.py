"""Fixtures shared by several test directories, re-exported by their conftest files."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import stamina
from stamina.instrumentation import LoggingOnRetryHook, RetryDetails

from tests.support.agents import RunMode


@pytest.fixture(params=["invoke", "ainvoke"])
def run_mode(request: pytest.FixtureRequest) -> RunMode:
    """Run each agent test once under `invoke()` and once under `ainvoke()`."""
    return cast("RunMode", request.param)


@pytest.fixture
def retry_details() -> Iterator[list[RetryDetails]]:
    """Collect what every stamina retry hook is handed, beside stamina's own logging hook."""
    collected: list[RetryDetails] = []

    def collect(details: RetryDetails) -> None:
        collected.append(details)

    stamina.instrumentation.set_on_retry_hooks([LoggingOnRetryHook, collect])
    yield collected
    stamina.instrumentation.set_on_retry_hooks(None)
