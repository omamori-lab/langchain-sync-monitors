"""The log-record check the privacy tests rely on reports live HTTP objects and quoted secrets.

An httpx error's text quotes no header, but the error holds the live request,
and an auth object's text quotes no password, but it holds the credential, so
a record that carries either is reported whatever its text says. So is a
record whose error was raised from, or while handling, an httpx error.
"""

from __future__ import annotations

import logging
from types import TracebackType

import httpx
import pytest

from langchain_sync_monitors.score_requests import build_http_client
from tests.support.log_records import find_logged_leaks

PLANTED_KEY = "planted-log-key-5e1a"

PLANTED_PASSWORD = "planted-log-password-9b4d"

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


def raise_while_handling(handled: Exception, *, hide_cause: bool) -> ExceptionInfo:
    """Raise a plain error while handling another, and return the exception info.

    Raised `from` the handled error, the plain error holds it as its cause and
    its context; raised `from None`, as its context alone, which a traceback
    then leaves out.
    """
    try:
        try:
            raise handled
        except Exception as error:
            if hide_cause:
                raise RuntimeError("score export failed") from None
            raise RuntimeError("score export failed") from error
    except RuntimeError as wrapper:
        return (type(wrapper), wrapper, wrapper.__traceback__)
    raise AssertionError("the handler raises")


def read_score_client_auth() -> httpx.Auth:
    """Return the auth the score client keeps from a key pair, as the Langfuse client is built."""
    with build_http_client(
        base_url="https://service.test", auth=("pk-planted", PLANTED_PASSWORD)
    ) as client:
        auth = client.auth
    if auth is None:
        raise AssertionError("a client given a key pair keeps an auth")
    return auth


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


@pytest.mark.parametrize(
    "auth",
    [httpx.BasicAuth("pk-planted", PLANTED_PASSWORD), read_score_client_auth()],
    ids=["basic", "score-client"],
)
def test_a_record_whose_extras_hold_an_auth_object_is_reported(auth: httpx.Auth) -> None:
    # Arrange: the auth's text quotes no password, though it holds the header built from one
    record = build_record(extras={"auth": auth})

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_PASSWORD])

    # Assert
    assert PLANTED_PASSWORD not in f"{auth} {auth!r}"
    assert leaks == [f"score: a live {type(auth).__name__}"]


@pytest.mark.parametrize("hide_cause", [False, True], ids=["cause", "context"])
def test_a_record_whose_error_was_raised_while_handling_an_httpx_error_is_reported(
    *, hide_cause: bool
) -> None:
    # Arrange: the plain error's text quotes neither the key nor the transport error
    transport_error = httpx.ConnectError("Connection refused", request=REQUEST)
    exc_info = raise_while_handling(transport_error, hide_cause=hide_cause)
    record = build_record(exc_info=exc_info)

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert: the transport error is reported once, though `from` makes it cause and context
    _, error, _ = exc_info
    assert PLANTED_KEY not in f"{error} {error!r}"
    assert leaks == ["score: a live ConnectError"]


@pytest.mark.parametrize("hide_cause", [False, True], ids=["cause", "context"])
def test_a_record_whose_error_was_raised_while_handling_a_plain_error_is_clean(
    *, hide_cause: bool
) -> None:
    # Arrange: neither error quotes the key
    exc_info = raise_while_handling(ValueError("score out of range"), hide_cause=hide_cause)
    record = build_record(exc_info=exc_info)

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert
    assert leaks == []


def test_a_record_naming_only_the_path_and_the_error_type_is_clean() -> None:
    # Arrange: what `send_request` logs, the method, path and error's account
    record = build_record(
        args=("POST", REQUEST.url.path, "HTTPStatusError (HTTP 503)"),
        extras={"stamina.caused_by": repr(build_status_error()), "stamina.args": ()},
    )

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

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


def test_errors_whose_chain_loops_are_walked_once() -> None:
    # Arrange: each error is the other's cause or context, and one quotes the key
    wrapper = RuntimeError("score export failed")
    handled = ValueError(f"key={PLANTED_KEY}")
    wrapper.__cause__ = handled
    handled.__context__ = wrapper
    record = build_record(exc_info=(RuntimeError, wrapper, None))

    # Act
    leaks = find_logged_leaks([record], secrets=[PLANTED_KEY])

    # Assert
    assert leaks == ["score: a ValueError quoting a secret"]
