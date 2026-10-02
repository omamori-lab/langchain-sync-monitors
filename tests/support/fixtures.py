"""Fixtures shared by several test directories, re-exported by their conftest files."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
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


@dataclass
class RetryHookCalls:
    """What each stamina retry hook was handed, and what the frames of its error held then."""

    details: list[RetryDetails] = field(default_factory=list)
    frame_locals: list[dict[str, str]] = field(default_factory=list)


def read_frame_locals(error: BaseException) -> dict[str, str]:
    """Return the repr of every local of every frame in the error's traceback.

    Each is named `function.local`. A hook, or an error reporter that records
    each frame's locals by repr, reads these from the error it is handed.
    """
    frame_locals: dict[str, str] = {}
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        for name, value in frame.f_locals.items():
            frame_locals[f"{frame.f_code.co_name}.{name}"] = repr(value)
        traceback = traceback.tb_next
    return frame_locals


@pytest.fixture
def retry_hook_calls() -> Iterator[RetryHookCalls]:
    """Collect each stamina retry hook call, beside stamina's own logging hook.

    The frames' locals are read while the hook runs, as a hook of the user's
    would read them.
    """
    calls = RetryHookCalls()

    def collect(details: RetryDetails) -> None:
        calls.details.append(details)
        calls.frame_locals.append(read_frame_locals(details.caused_by))

    stamina.instrumentation.set_on_retry_hooks([LoggingOnRetryHook, collect])
    yield calls
    stamina.instrumentation.set_on_retry_hooks(None)


@pytest.fixture
def retry_details(retry_hook_calls: RetryHookCalls) -> list[RetryDetails]:
    """Collect what every stamina retry hook is handed, beside stamina's own logging hook."""
    return retry_hook_calls.details


@pytest.fixture
def retry_frame_locals(retry_hook_calls: RetryHookCalls) -> list[dict[str, str]]:
    """Collect, for each retry, the repr of each local of the frames in the error a hook gets."""
    return retry_hook_calls.frame_locals


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
