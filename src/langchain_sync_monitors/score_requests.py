"""How the score senders reach LangSmith and Langfuse: credentials, retries and rate limits.

Both senders send with httpx [@httpx2024] and retry a transport failure or a
server error with stamina [@schlawack2026stamina], once, within a short time
budget, so that a window never outlasts the exit drain by much. A `429`
answer is not retried here: the sender reports the pause its `Retry-After`
header asks for, and the worker holds every call to that tool until then,
since both tools count requests per organisation or per key, not per call
[@langfuse2026apilimits; @langsmith2026retention].

The credentials are the variables the tools' own SDKs read, so a process
that traces to a tool can write its scores with no set-up of its own.
Neither sender imports the tool's SDK. Each client takes its proxies and
certificates from the environment, as httpx does by default, except for the
macOS System Settings proxies in a forked child, which
`is_system_proxy_lookup_safe` explains.
"""

from __future__ import annotations

import email.utils
import logging
import math
import os
import ssl
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, TypeGuard

import httpx

from langchain_sync_monitors.monitors.openrouter_decisions import check_key_characters
from langchain_sync_monitors.retries import call_with_retries_sync

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS: Final = 5.0
"""How long one request may take, connecting and reading included."""

RETRY_ATTEMPTS: Final = 2
"""How many times a request is sent, so one retry, before a failure leaves its score waiting."""

RETRY_BUDGET_SECONDS: Final = 10.0
"""How long stamina keeps retrying one request."""

DEFAULT_PAUSE_SECONDS: Final = 30.0
"""The pause after a `429` whose `Retry-After` is missing or unreadable."""

MAX_PAUSE_SECONDS: Final = 3600.0
"""The longest pause a `Retry-After` may ask for: an hour, as LangSmith's hourly limit lasts."""


@dataclass(slots=True)
class ProcessOrigin:
    """Whether this process is a child forked from another, which the at-fork hook records."""

    forked: bool = False


PROCESS_ORIGIN = ProcessOrigin()
"""This process's origin; `score_export`'s at-fork hook marks a forked child."""


def is_system_proxy_lookup_safe() -> bool:
    """Tell whether a client may look up the system's proxies, as httpx does by default.

    On macOS, when no proxy variable is set, `urllib.request.getproxies`
    reads the System Settings proxies through the System Configuration
    framework [@cpython2026], which httpx calls for every client it builds
    [@httpx2024]. A child forked from a process that has threads, or that
    has read them already, is killed by that call, with SIGSEGV or SIGABRT.
    So a forked child on macOS reads them only from the environment.
    """
    if sys.platform != "darwin" or not PROCESS_ORIGIN.forked:
        return True
    return bool(urllib.request.getproxies_environment())


def read_environment_certificates() -> ssl.SSLContext | bool:
    """Return the certificates httpx trusts from the environment, as its `trust_env` reads them.

    `SSL_CERT_FILE`, else `SSL_CERT_DIR`, else True, httpx's own default
    [@httpx2024].
    """
    certificate_file = os.environ.get("SSL_CERT_FILE")
    if certificate_file:
        return ssl.create_default_context(cafile=certificate_file)
    certificate_directory = os.environ.get("SSL_CERT_DIR")
    if certificate_directory:
        return ssl.create_default_context(capath=certificate_directory)
    return True


def build_http_client(
    *,
    base_url: str,
    headers: Mapping[str, str] | None = None,
    auth: tuple[str, str] | None = None,
) -> httpx.Client:
    """Return a client for one service, set up from the environment as httpx sets one up.

    Where the system's proxies cannot be looked up safely, the client skips
    them and still trusts the environment's certificates.

    A user name or password in the base URL becomes the client's Basic
    authentication, as httpx would apply it, unless `auth` is given, which
    httpx prefers; a URL whose user name and password are both empty, such
    as `https://:@host`, authenticates with nothing (method
    `BaseClient._build_request_auth`, module `httpx/_client.py`) [@httpx2024].
    The URL keeps neither, so no request's URL, and no error that quotes
    one, holds the password.
    """
    url = httpx.URL(base_url)
    if url.userinfo:
        if auth is None and (url.username or url.password):
            auth = (url.username, url.password)
        base_url = str(url.copy_with(userinfo=b""))
    if is_system_proxy_lookup_safe():
        return httpx.Client(
            base_url=base_url, headers=headers, auth=auth, timeout=REQUEST_TIMEOUT_SECONDS
        )
    return httpx.Client(
        base_url=base_url,
        headers=headers,
        auth=auth,
        timeout=REQUEST_TIMEOUT_SECONDS,
        trust_env=False,
        verify=read_environment_certificates(),
    )


def read_environment_value(*names: str) -> str | None:
    """Return the first variable of these that is set and not blank, trimmed as the SDKs trim it.

    A value that no HTTP header may carry raises `ConfigurationError`, which
    names the variable and no part of the value.
    """
    for name in names:
        value = os.environ.get(name, "").strip().strip("\"'")
        if value:
            check_key_characters(value, source=name)
            return value
    return None


def is_transient_failure(error: Exception) -> bool:
    """Retry a transport failure or a server error; never a client error, a `429` included.

    A request the client itself got wrong, such as an unsupported URL
    scheme, fails the same way every time, though httpx counts it among its
    transport errors [@httpx2024].
    """
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR
    if isinstance(error, httpx.LocalProtocolError | httpx.UnsupportedProtocol):
        return False
    return isinstance(error, httpx.TransportError)


def send_with_retries(http_client: httpx.Client, *, request: httpx.Request) -> httpx.Response:
    """Send the request, again after a transport failure or a server error.

    `call_with_retries_sync` retries it [@schlawack2026stamina], so no retry
    hook is handed the client, the request or httpx's error, whose request
    holds the key in its headers: stamina's retry log holds the wait and a
    `RetriedCallError`'s repr, which names httpx's error type and the status
    alone. The retried block is a nested function, not a `functools.partial`,
    so its repr names no request. After the last attempt, httpx's error is
    raised.
    """

    def send_once() -> httpx.Response:
        return send_raising_on_server_error(http_client, request=request)

    return call_with_retries_sync(
        send_once,
        is_retried=is_transient_failure,
        attempts=RETRY_ATTEMPTS,
        timeout=RETRY_BUDGET_SECONDS,
    )


def send_raising_on_server_error(
    http_client: httpx.Client, *, request: httpx.Request
) -> httpx.Response:
    """Send the request once, raising `httpx.HTTPStatusError` on a server error alone."""
    response = http_client.send(request)
    if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
        response.raise_for_status()
    return response


def send_request(http_client: httpx.Client, *, request: httpx.Request) -> httpx.Response | None:
    """Send the request with retries, or return None, logged, when it still failed.

    The log names the request's path and `describe_failure`'s account of
    the error, never a header. httpx's own request log quotes the URL, and
    the URL holds no key and no text of a run: the tool's endpoint, from
    which `build_http_client` takes any user name and password; the path;
    and the query of a lookup. That query names the LangSmith project, or
    asks Langfuse for a page of the `monitor step` observations that started
    in a time window, by the span's name, the window's bounds, the page size
    and the cursor Langfuse returned. The score itself goes in the body.
    """
    try:
        return send_with_retries(http_client, request=request)
    except httpx.HTTPError as error:
        logger.info(
            "score export: %s %s failed with %s",
            request.method,
            request.url.path,
            describe_failure(error),
        )
        return None


def describe_failure(error: httpx.HTTPError) -> str:
    """Return the error's type, with the status of a server's error answer, but never its text.

    An httpx error's text can quote the request's URL, so it is left out.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return f"{type(error).__name__} (HTTP {error.response.status_code})"
    return type(error).__name__


def is_rate_limited(response: httpx.Response | None) -> TypeGuard[httpx.Response]:
    """Tell whether the service answered `429`, asking for a pause; None is no answer at all."""
    return response is not None and response.status_code == httpx.codes.TOO_MANY_REQUESTS


def read_pause_seconds(response: httpx.Response) -> float:
    """Return the pause a `429` answer asks for: its `Retry-After`, in seconds or as a date.

    A missing, unreadable or infinite value gives `DEFAULT_PAUSE_SECONDS`,
    a date in the past no pause, and anything longer than an hour an hour
    [@langsmith2026retention].
    """
    header = response.headers.get("retry-after", "").strip()
    try:
        seconds = float(header)
    except ValueError:
        seconds = read_seconds_until(header)
    if not math.isfinite(seconds):
        return DEFAULT_PAUSE_SECONDS
    return min(max(seconds, 0.0), MAX_PAUSE_SECONDS)


def read_seconds_until(http_date: str) -> float:
    """Return the seconds from now until an HTTP date, or the default pause for anything else."""
    try:
        moment = email.utils.parsedate_to_datetime(http_date)
    except (TypeError, ValueError):
        return DEFAULT_PAUSE_SECONDS
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - datetime.now(UTC)).total_seconds()
