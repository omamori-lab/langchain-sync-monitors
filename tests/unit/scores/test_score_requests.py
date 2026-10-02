"""The senders' requests: which failures are retried, and how long a `429` pauses a tool.

`Retry-After` may be a number of seconds or an HTTP date [@langfuse2026apilimits];
anything unreadable pauses for the default, and a date already past for nothing.
"""

from __future__ import annotations

import functools
import logging
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import MagicMock

import certifi
import httpx
import pytest
from pydantic import SecretStr
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors import score_requests
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.langfuse_scores import LangfuseScoreSender
from langchain_sync_monitors.langsmith_scores import LangSmithFeedbackSender
from langchain_sync_monitors.score_requests import (
    DEFAULT_PAUSE_SECONDS,
    MAX_PAUSE_SECONDS,
    PROCESS_ORIGIN,
    build_http_client,
    is_system_proxy_lookup_safe,
    is_transient_failure,
    read_environment_certificates,
    read_environment_value,
    read_pause_seconds,
    send_request,
)
from langchain_sync_monitors.scores import LangSmithCredentials, PendingScore, ScoreSender, Tracer
from tests.support.log_records import find_frame_leaks, find_leaks, find_logged_leaks
from tests.support.malformed_replies import PLANTED_REPLY_HEADER, build_malformed_reply_error
from tests.support.score_services import (
    LANGSMITH_ENDPOINT,
    PROJECT_NAME,
    FakeLangfuse,
    FakeLangSmith,
    build_step_id,
)

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")

REQUEST = httpx.Request("GET", "https://service.test/items")


def build_rate_limited(retry_after: str | None) -> httpx.Response:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(429, headers=headers, request=REQUEST)


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("7", 7.0), ("1.5", 1.5), ("0", 0.0), ("-3", 0.0), ("86400", 3600.0)],
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


def test_a_retry_after_date_without_a_zone_is_read_as_utc() -> None:
    # Arrange: RFC 2822 writes an unknown zone as -0000, which Python reads as a naive time
    moment = (datetime.now(UTC) + timedelta(seconds=60)).replace(tzinfo=None)

    # Act
    pause = read_pause_seconds(build_rate_limited(format_datetime(moment)))

    # Assert
    assert 55.0 <= pause <= 60.0


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


def test_a_server_error_is_retried_and_then_leaves_no_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    answers = [httpx.Response(502), httpx.Response(503)]
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(answer))

    # Act
    with caplog.at_level(logging.INFO, logger="langchain_sync_monitors.score_requests"):
        response = send_request(
            client, request=client.build_request("GET", "https://service.test/items")
        )

    # Assert: the log names the path, the error's type and the last status, never a header
    assert response is None
    assert len(seen) == 2
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "langchain_sync_monitors.score_requests"
    ] == ["score export: GET /items failed with HTTPStatusError (HTTP 503)"]
    assert all(record.exc_info is None for record in caplog.records)


PLANTED_KEY = "planted-score-key-7e1b"


@dataclass(frozen=True, slots=True, kw_only=True)
class RetriedSend:
    """A sender whose first request meets a server error, its score, and what its fake received.

    `header` names the request header that carries the planted key.
    """

    sender: ScoreSender
    score: PendingScore
    requests: list[httpx.Request]
    header: str


def build_langsmith_send() -> RetriedSend:
    """Return a LangSmith sender whose project lookup fails once, with the key in `x-api-key`."""
    service = FakeLangSmith(queued_answers=[httpx.Response(503)])
    connection = LangSmithCredentials(
        api_key=SecretStr(PLANTED_KEY), endpoint=LANGSMITH_ENDPOINT, workspace_id=None
    )
    score = PendingScore(
        step_id=build_step_id(),
        name="monitor_suspicion",
        value=0.9,
        tracer=Tracer.LANGSMITH,
        project=PROJECT_NAME,
        queued_at=0.0,
        connection=connection,
    )
    sender = LangSmithFeedbackSender(build_client=service.build_client, posts_in_flight=1)
    return RetriedSend(sender=sender, score=score, requests=service.requests, header="x-api-key")


def build_langfuse_send() -> RetriedSend:
    """Return a Langfuse sender whose step lookup fails once, with the key in Basic auth."""
    service = FakeLangfuse(queued_answers=[httpx.Response(503)])
    step_id = build_step_id()
    service.add_step(str(step_id))
    score = PendingScore(
        step_id=step_id,
        name="monitor_suspicion",
        value=0.9,
        tracer=Tracer.LANGFUSE,
        project=None,
        queued_at=0.0,
    )
    sender = LangfuseScoreSender(
        http_client=service.build_client(auth=("pk-planted", PLANTED_KEY)), clock=lambda: 0.0
    )
    return RetriedSend(
        sender=sender, score=score, requests=service.requests, header="authorization"
    )


@pytest.mark.parametrize(
    "build_send", [build_langsmith_send, build_langfuse_send], ids=["langsmith", "langfuse"]
)
def test_a_retried_request_hands_no_retry_hook_the_request_and_logs_no_key(
    build_send: Callable[[], RetriedSend],
    retry_details: list[RetryDetails],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the sender's first request, a lookup, meets a server error, so it is sent again
    caplog.set_level(logging.DEBUG)
    case = build_send()

    # Act
    report = case.sender.send([case.score])
    case.sender.close()

    # Assert: the lookup was sent again and the score written; no record, on any logger,
    # holds the key, the header that carried it or a live request, and no hook was handed one
    first, retried, _ = case.requests
    assert retried.url == first.url
    assert report.written == [case.score]
    secrets = [PLANTED_KEY, first.headers[case.header]]
    assert "stamina.retry_scheduled" in [record.getMessage() for record in caplog.records]
    assert find_logged_leaks(caplog.records, secrets=secrets) == []
    (details,) = retry_details
    assert (details.name, details.args, details.kwargs) == ("<context block>", (), {})
    assert find_leaks(details.caused_by, secrets=secrets) == []


PLANTED_PASSWORD = "planted-url-password-31c9"


@pytest.mark.parametrize(
    "base_url",
    [
        f"https://monitor:{PLANTED_PASSWORD}@service.test/api",
        "https://monitor@service.test/api",
        f"https://:{PLANTED_PASSWORD}@service.test/api",
        "https://:@service.test/api",
    ],
    ids=["user-and-password", "user-only", "password-only", "both-empty"],
)
@pytest.mark.parametrize("auth", [None, ("pk-given", "sk-given")], ids=["url", "given"])
def test_credentials_in_the_base_url_authenticate_as_httpx_would_and_stay_out_of_every_url(
    monkeypatch: pytest.MonkeyPatch,
    retry_details: list[RetryDetails],
    caplog: pytest.LogCaptureFixture,
    *,
    base_url: str,
    auth: tuple[str, str] | None,
) -> None:
    # Arrange: every request meets a server error, whose text quotes the request's URL
    caplog.set_level(logging.DEBUG)
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    plain_client = httpx.Client
    with plain_client(base_url=base_url, auth=auth, transport=httpx.MockTransport(answer)) as own:
        own.get("/sessions")
    caplog.clear()  # httpx's own client logs its URL, password included
    monkeypatch.setattr(
        score_requests.httpx,
        "Client",
        functools.partial(plain_client, transport=httpx.MockTransport(answer)),
    )

    # Act
    with build_http_client(base_url=base_url, auth=auth) as client:
        response = send_request(
            client, request=client.build_request("GET", "/sessions", params={"limit": 1})
        )

    # Assert: the request authenticates as httpx's own client from that URL would, with no
    # header when the URL's user name and password are both empty, and neither its URL nor
    # any record holds the password
    httpx_request, first, retried = seen
    assert response is None
    assert first.headers.get("authorization") == httpx_request.headers.get("authorization")
    assert str(first.url) == str(retried.url) == "https://service.test/api/sessions?limit=1"
    assert len(retry_details) == 1
    assert find_logged_leaks(caplog.records, secrets=[PLANTED_PASSWORD]) == []


def test_a_malformed_reply_is_retried_and_hands_a_retry_hook_no_part_of_it(
    retry_details: list[RetryDetails],
    retry_frame_locals: list[dict[str, str]],
    every_log_record: list[logging.LogRecord],
) -> None:
    # Arrange: the first reply cannot be parsed, and httpx's error quotes its header line
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise build_malformed_reply_error()
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(answer), headers={"x-api-key": PLANTED_KEY})

    # Act
    response = send_request(client, request=client.build_request("GET", "https://service.test/"))

    # Assert: the request was sent again, and stamina logged the error's type alone
    assert response is not None
    assert response.status_code == 200
    assert len(seen) == 2
    (details,) = retry_details
    expected = "RetriedCallError(error_type='RemoteProtocolError', http_status=None)"
    assert repr(details.caused_by) == expected

    # Assert: no record on any logger, and nothing a retry hook is handed, holds the key, the
    # reply's header or a live httpx object
    secrets = [PLANTED_KEY, PLANTED_REPLY_HEADER]
    assert find_logged_leaks(every_log_record, secrets=secrets) == []
    assert find_leaks(details.caused_by, secrets=secrets) == []

    # Assert: nor does any local of the frames in the stand-in's traceback, read by repr
    (frame_locals,) = retry_frame_locals
    assert "run_attempt_sync.failures" in frame_locals
    assert find_frame_leaks(retry_frame_locals, secrets=secrets) == []


def test_a_malformed_reply_on_every_attempt_leaves_no_response_and_logs_its_type(
    retry_details: list[RetryDetails],
    every_log_record: list[logging.LogRecord],
) -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise build_malformed_reply_error()

    client = httpx.Client(transport=httpx.MockTransport(answer))

    # Act
    response = send_request(
        client, request=client.build_request("GET", "https://service.test/items")
    )

    # Assert: the sender gives up as before, naming the path and the error's type
    assert response is None
    assert len(seen) == 2
    assert len(retry_details) == 1
    assert [
        record.getMessage()
        for record in every_log_record
        if record.name == "langchain_sync_monitors.score_requests"
    ] == ["score export: GET /items failed with RemoteProtocolError"]
    assert find_logged_leaks(every_log_record, secrets=[PLANTED_REPLY_HEADER]) == []


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
    monkeypatch.setenv("SECOND_NAME_TEST", ' "X-value-X" ')

    # Act
    value = read_environment_value("FIRST_NAME_TEST", "SECOND_NAME_TEST")

    # Assert
    assert value == "X-value-X"


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


def test_a_retry_after_date_more_than_an_hour_ahead_pauses_for_an_hour() -> None:
    # Arrange
    tomorrow = format_datetime(datetime.now(UTC) + timedelta(days=1), usegmt=True)

    # Act
    pause = read_pause_seconds(build_rate_limited(tomorrow))

    # Assert
    assert pause == MAX_PAUSE_SECONDS == 3600.0


def clear_proxy_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every proxy variable, in both cases, as on a machine with none."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def place_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str,
    forked: bool,
) -> None:
    """Make the process look as if it ran on `platform`, forked or not."""
    monkeypatch.setattr(score_requests.sys, "platform", platform)
    monkeypatch.setattr(PROCESS_ORIGIN, "forked", forked)


@pytest.mark.parametrize(
    ("platform", "forked", "proxy", "expected"),
    [
        ("darwin", True, None, False),
        ("darwin", True, "http://proxy.test:3128", True),
        ("darwin", False, None, True),
        ("linux", True, None, True),
    ],
)
def test_only_a_forked_macos_child_without_a_proxy_variable_skips_the_system_lookup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str,
    forked: bool,
    proxy: str | None,
    expected: bool,
) -> None:
    # Arrange
    clear_proxy_variables(monkeypatch)
    if proxy is not None:
        monkeypatch.setenv("HTTPS_PROXY", proxy)
    place_process(monkeypatch, platform=platform, forked=forked)

    # Act
    safe = is_system_proxy_lookup_safe()

    # Assert
    assert safe is expected


def test_a_client_reads_the_environment_as_httpx_does_where_the_lookup_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    place_process(monkeypatch, platform="linux", forked=True)

    # Act
    with build_http_client(base_url="https://service.test", headers={"x-key": "k"}) as client:
        # Assert
        assert client.trust_env is True
        assert client.headers["x-key"] == "k"
        assert client.timeout.read == score_requests.REQUEST_TIMEOUT_SECONDS


def test_a_forked_macos_child_client_skips_the_system_proxies_and_keeps_the_certificates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    clear_proxy_variables(monkeypatch)
    place_process(monkeypatch, platform="darwin", forked=True)
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    recorder = MagicMock(wraps=httpx.Client)
    monkeypatch.setattr(score_requests.httpx, "Client", recorder)

    # Act
    with build_http_client(base_url="https://service.test", auth=("public", "secret")) as client:
        # Assert
        assert client.trust_env is False
        assert client.timeout.read == score_requests.REQUEST_TIMEOUT_SECONDS
    assert isinstance(recorder.call_args.kwargs["verify"], ssl.SSLContext)


@pytest.mark.parametrize(
    ("variable", "expected_type"),
    [("SSL_CERT_FILE", ssl.SSLContext), ("SSL_CERT_DIR", ssl.SSLContext), (None, bool)],
)
def test_the_certificates_come_from_the_variables_httpx_reads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    variable: str | None,
    expected_type: type,
) -> None:
    # Arrange
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    if variable == "SSL_CERT_FILE":
        monkeypatch.setenv(variable, certifi.where())
    elif variable == "SSL_CERT_DIR":
        monkeypatch.setenv(variable, str(Path(certifi.where()).parent))

    # Act
    certificates = read_environment_certificates()

    # Assert
    assert isinstance(certificates, expected_type)
    assert certificates is not False
