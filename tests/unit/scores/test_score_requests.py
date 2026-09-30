"""The senders' requests: which failures are retried, and how long a `429` pauses a tool.

`Retry-After` may be a number of seconds or an HTTP date [@langfuse2026apilimits];
anything unreadable pauses for the default, and a date already past for nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.score_requests import (
    DEFAULT_PAUSE_SECONDS,
    is_transient_failure,
    read_environment_value,
    read_pause_seconds,
    send_request,
)

REQUEST = httpx.Request("GET", "https://service.test/items")


def build_rate_limited(retry_after: str | None) -> httpx.Response:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(429, headers=headers, request=REQUEST)


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("7", 7.0), ("1.5", 1.5), ("0", 0.0), ("-3", 0.0)],
)
def test_a_retry_after_in_seconds_is_the_pause(retry_after: str, expected: float) -> None:
    # Act
    pause = read_pause_seconds(build_rate_limited(retry_after))

    # Assert
    assert pause == expected


def test_a_retry_after_date_pauses_until_that_date() -> None:
    # Arrange
    moment = datetime.now(UTC) + timedelta(seconds=90)

    # Act
    pause = read_pause_seconds(build_rate_limited(format_datetime(moment, usegmt=True)))

    # Assert
    assert 85.0 <= pause <= 90.0


def test_a_retry_after_date_in_the_past_asks_for_no_pause() -> None:
    # Arrange
    moment = datetime.now(UTC) - timedelta(minutes=5)

    # Act
    pause = read_pause_seconds(build_rate_limited(format_datetime(moment, usegmt=True)))

    # Assert
    assert pause == 0.0


@pytest.mark.parametrize("retry_after", [None, "", "soon", "inf", "nan"])
def test_a_missing_or_unreadable_retry_after_gives_the_default_pause(
    retry_after: str | None,
) -> None:
    # Act
    pause = read_pause_seconds(build_rate_limited(retry_after))

    # Assert
    assert pause == DEFAULT_PAUSE_SECONDS


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
        (
            httpx.HTTPStatusError(
                "server", request=REQUEST, response=httpx.Response(503, request=REQUEST)
            ),
            True,
        ),
        (
            httpx.HTTPStatusError(
                "server", request=REQUEST, response=httpx.Response(500, request=REQUEST)
            ),
            True,
        ),
        (
            httpx.HTTPStatusError(
                "limited", request=REQUEST, response=httpx.Response(429, request=REQUEST)
            ),
            False,
        ),
        (
            httpx.HTTPStatusError(
                "client", request=REQUEST, response=httpx.Response(499, request=REQUEST)
            ),
            False,
        ),
        (httpx.UnsupportedProtocol("ftp"), False),
        (httpx.LocalProtocolError("bad header"), False),
        (ValueError("not HTTP"), False),
    ],
)
def test_only_transport_failures_and_server_errors_are_retried(
    error: Exception,
    expected: bool,
) -> None:
    # Act
    retried = is_transient_failure(error)

    # Assert
    assert retried is expected


def test_a_server_error_is_retried_and_then_leaves_no_response() -> None:
    # Arrange
    answers = [httpx.Response(502), httpx.Response(503)]
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(answer))

    # Act
    response = send_request(client, request=client.build_request("GET", "https://service.test/"))

    # Assert
    assert response is None
    assert len(seen) == 2


def test_a_server_error_then_a_success_returns_the_success() -> None:
    # Arrange
    answers = [httpx.Response(500), httpx.Response(200, json={"ok": True})]
    client = httpx.Client(transport=httpx.MockTransport(lambda request: answers.pop(0)))

    # Act
    response = send_request(client, request=client.build_request("GET", "https://service.test/"))

    # Assert
    assert response is not None
    assert response.status_code == 200


def test_a_client_error_is_returned_without_a_retry() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(answer))

    # Act
    response = send_request(client, request=client.build_request("GET", "https://service.test/"))

    # Assert
    assert response is not None
    assert response.status_code == 404
    assert len(seen) == 1


def test_a_variable_is_trimmed_as_the_sdks_trim_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("FIRST_NAME_TEST", "   ")
    monkeypatch.setenv("SECOND_NAME_TEST", ' "value" ')

    # Act
    value = read_environment_value("FIRST_NAME_TEST", "SECOND_NAME_TEST")

    # Assert
    assert value == "value"


def test_a_variable_no_header_may_carry_is_refused_without_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("KEY_NAME_TEST", "secret\x07value")

    # Act
    with pytest.raises(ConfigurationError) as raised:
        read_environment_value("KEY_NAME_TEST")

    # Assert
    assert "KEY_NAME_TEST" in str(raised.value)
    assert "secret" not in str(raised.value)
