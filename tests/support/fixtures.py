"""Fixtures shared by several test directories, re-exported by their conftest files."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import cast

import pytest
import stamina
from stamina.instrumentation import LoggingOnRetryHook, RetryDetails

from tests.support.agents import RunMode
from tests.support.log_records import RecordCollector


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


@pytest.fixture
def every_log_record(caplog: pytest.LogCaptureFixture) -> Iterator[list[logging.LogRecord]]:
    """Collect every record at every level, from every logger, propagating or not.

    Every logger that exists, httpx's and stamina's included, is set to DEBUG,
    and the collector sits on the root logger and on each logger that does not
    propagate. A logger made later inherits the root's DEBUG level and
    propagates. caplog restores the levels afterwards.
    """
    collector = RecordCollector()
    caplog.set_level(logging.DEBUG)
    loggers = [
        logger
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        caplog.set_level(logging.DEBUG, logger=logger.name)
    holders = [logging.root, *(logger for logger in loggers if not logger.propagate)]
    for holder in holders:
        holder.addHandler(collector)
    yield collector.records
    for holder in holders:
        holder.removeHandler(collector)
