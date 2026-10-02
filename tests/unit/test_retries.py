"""The retries every retry site shares: a stand-in for the error, then the error itself."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import httpx
import pytest
import stamina
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors.retries import (
    build_stand_in,
    call_with_retries,
    call_with_retries_sync,
    read_http_status,
)
from tests.support.fixtures import retry_details as retry_details
from tests.support.fixtures import retry_frame_locals as retry_frame_locals
from tests.support.fixtures import retry_hook_calls as retry_hook_calls
from tests.support.log_records import find_frame_leaks
from tests.unit.monitors.doubles import PLANTED_SECRET

type CallPath = Literal["async", "sync"]


class StatusError(Exception):
    """An error in the shape provider SDKs give one: the HTTP status as `status_code`."""

    def __init__(self, status_code: object) -> None:
        super().__init__(f"Error code: {status_code} - {{'echo': '{PLANTED_SECRET}'}}")
        self.status_code = status_code


def is_rate_limit(error: Exception) -> bool:
    """Retry only a `StatusError` with HTTP 429."""
    return isinstance(error, StatusError) and error.status_code == 429


class ScriptedBlock:
    """Raises the scripted errors in turn, then returns `"done"`, and counts its runs."""

    def __init__(self, *, errors: list[Exception]) -> None:
        self.errors = errors
        self.runs = 0

    def run(self) -> str:
        self.runs += 1
        if self.runs <= len(self.errors):
            raise self.errors[self.runs - 1]
        return "done"

    async def run_async(self) -> str:
        return self.run()


async def call_on_path(block: ScriptedBlock, *, call_path: CallPath) -> str:
    """Run the block with three attempts, through the async or the sync runner."""
    if call_path == "async":
        return await call_with_retries(block.run_async, is_retried=is_rate_limit, attempts=3)
    return call_with_retries_sync(block.run, is_retried=is_rate_limit, attempts=3)


@pytest.fixture(autouse=True)
def three_attempts_without_waiting() -> Iterator[None]:
    with stamina.set_testing(True, attempts=3):
        yield


@pytest.fixture(params=["async", "sync"])
def call_path(request: pytest.FixtureRequest) -> CallPath:
    path: CallPath = request.param
    return path


async def test_a_retried_error_is_handed_to_the_hooks_as_its_stand_in(
    call_path: CallPath,
    retry_details: list[RetryDetails],
) -> None:
    # Arrange
    block = ScriptedBlock(errors=[StatusError(429)])

    # Act
    result = await call_on_path(block, call_path=call_path)

    # Assert
    assert result == "done"
    assert block.runs == 2
    (details,) = retry_details
    assert repr(details.caused_by) == "RetriedCallError(error_type='StatusError', http_status=429)"
    assert PLANTED_SECRET not in str(details.caused_by)


async def test_no_frame_in_the_stand_ins_traceback_shows_a_retried_error(
    call_path: CallPath,
    retry_details: list[RetryDetails],
    retry_frame_locals: list[dict[str, str]],
) -> None:
    # Arrange: two retried errors, whose reprs quote the secret, then a reply
    block = ScriptedBlock(errors=[StatusError(429), StatusError(429)])

    # Act
    result = await call_on_path(block, call_path=call_path)

    # Assert: each hook read the library's frames, where the errors are kept, by number alone
    assert result == "done"
    assert len(retry_details) == len(retry_frame_locals) == 2
    suffix = "" if call_path == "async" else "_sync"
    holders = [f"run_attempt{suffix}.failures", f"call_with_retries{suffix}.failures"]
    kept = [[frame_locals[name] for name in holders] for frame_locals in retry_frame_locals]
    assert kept == [["KeptFailures(count=1)"] * 2, ["KeptFailures(count=2)"] * 2]

    # Assert: and no local in those frames quotes the secret
    assert find_frame_leaks(retry_frame_locals, secrets=[PLANTED_SECRET]) == []


async def test_an_error_not_retried_after_a_retried_one_is_raised_as_it_is(
    call_path: CallPath,
    retry_details: list[RetryDetails],
) -> None:
    # Arrange: a server error on the second run, which the site does not retry
    server_error = StatusError(500)
    block = ScriptedBlock(errors=[StatusError(429), server_error])

    # Act
    with pytest.raises(StatusError) as raised:
        await call_on_path(block, call_path=call_path)

    # Assert: the error itself, with nothing chained to it, and no third run
    assert raised.value is server_error
    assert (raised.value.__cause__, raised.value.__context__) == (None, None)
    assert block.runs == 2
    assert len(retry_details) == 1


async def test_a_retried_error_on_every_attempt_raises_the_last_one(
    call_path: CallPath,
    retry_details: list[RetryDetails],
) -> None:
    # Arrange: a fourth run would succeed
    errors: list[Exception] = [StatusError(429) for _ in range(3)]
    block = ScriptedBlock(errors=errors)

    # Act
    with pytest.raises(StatusError) as raised:
        await call_on_path(block, call_path=call_path)

    # Assert
    assert raised.value is errors[-1]
    assert (raised.value.__cause__, raised.value.__context__) == (None, None)
    assert block.runs == 3
    assert len(retry_details) == 2


def build_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://service.test/v1")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"status {status_code}", request=request, response=response)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(build_status_error(503), 503, id="httpx-status-error"),
        pytest.param(StatusError(429), 429, id="status-code-attribute"),
        pytest.param(StatusError("429"), None, id="status-code-as-text"),
        pytest.param(StatusError(True), None, id="status-code-as-bool"),
        pytest.param(httpx.ConnectError("connection reset"), None, id="no-status"),
    ],
)
def test_the_stand_in_names_an_integer_status_and_the_type_alone(
    error: Exception,
    expected: int | None,
) -> None:
    # Act
    stand_in = build_stand_in(error)

    # Assert
    assert read_http_status(error) == stand_in.http_status == expected
    assert stand_in.error_type == type(error).__name__
    assert PLANTED_SECRET not in f"{stand_in}{stand_in!r}"
