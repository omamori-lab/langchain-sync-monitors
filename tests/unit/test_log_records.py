"""The log-record check the privacy tests rely on reports live HTTP objects and quoted secrets.

An httpx error's text quotes no header, but the error holds the live request,
so a record that carries one is reported whatever its text says.
"""

from __future__ import annotations

import logging
from types import TracebackType

import httpx
import pytest

from tests.support.log_records import find_logged_leaks

PLANTED_KEY = "planted-log-key-5e1a"

REQUEST = httpx.Request("GET", "https://service.test/items", headers={"x-api-key": PLANTED_KEY})

type ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]


def build_record(
    *,
    args: tuple[object, ...] = (),
    exc_info: ExceptionInfo | None = None,
    extras: dict[str, object] | None = None,
) -> logging.LogRecord:
    """Return a record as `logger.info` makes it, with the given arguments, error and extras."""
    record = logging.LogRecord(
        "score",
        logging.INFO,
        __file__,
        1,
        "score export failed" + " with %s" * len(args),
        args,
        exc_info,
    )
    record.__dict__.update(extras or {})
    return record


def build_status_error() -> httpx.HTTPStatusError:
    """Return the error a `503` answer to the keyed request raises."""
    return httpx.HTTPStatusError(
        "Server error", request=REQUEST, response=httpx.Response(503, request=REQUEST)
    )


def raise_status_error() -> ExceptionInfo:
    """Raise and catch a status error, and return the exception info a handler would hold."""
    try:
        httpx.Response(503, request=REQUEST).raise_for_status()
    except httpx.HTTPStatusError as error:
        return (type(error), error, error.__traceback__)
    raise AssertionError("a 503 answer raises")


@pytest.mark.parametrize(
    "error",
    [build_status_error(), httpx.ConnectError("Connection refused", request=REQUEST)],
    ids=["status", "transport"],
)
def test_a_record_whose_args_hold_an_httpx_error_is_reported(error: httpx.HTTPError) -> None:
    # Arrange: the error's text quotes neither the key nor a header
    record = build_record(args=("POST", error))

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert
    assert PLANTED_KEY not in f"{error} {error!r}"
    assert leaks == [f"score: a live {type(error).__name__}"]


def test_a_record_whose_exc_info_holds_a_status_error_is_reported() -> None:
    # Arrange
    record = build_record(exc_info=raise_status_error())

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert
    assert leaks == ["score: a live HTTPStatusError"]


async def test_a_record_whose_extras_hold_an_async_client_is_reported() -> None:
    # Arrange
    async with httpx.AsyncClient(headers={"x-api-key": PLANTED_KEY}) as client:
        record = build_record(extras={"clients": {"langfuse": client}})

        # Act
        leaks = find_logged_leaks([record], secrets=[])

    # Assert
    assert leaks == ["score: a live AsyncClient"]


def test_a_record_naming_only_the_path_and_the_error_type_is_clean() -> None:
    # Arrange: what `send_request` logs, the method, path and error's account
    record = build_record(
        args=("POST", REQUEST.url.path, "HTTPStatusError (HTTP 503)"),
        extras={"stamina.caused_by": repr(build_status_error()), "stamina.args": ()},
    )

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY, REQUEST.headers["x-api-key"]])

    # Assert
    assert leaks == []


def test_values_that_hold_themselves_are_walked_once() -> None:
    # Arrange: a list and a mapping that each hold themselves, and a secret beside them
    looped: list[object] = [f"key={PLANTED_KEY}"]
    looped.append(looped)
    mapping: dict[str, object] = {"looped": looped}
    mapping["self"] = mapping
    record = build_record(extras={"cycle": mapping})

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert
    assert leaks == ["score: a str quoting a secret"]
