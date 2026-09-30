"""How the score senders reach LangSmith and Langfuse: credentials, retries and rate limits.

Both senders send with httpx [@httpx2024] and retry a transport failure or a
server error with stamina [@schlawack2026stamina], a few times within a short
time budget, so that a window never outlasts the exit drain by much. A `429`
answer is not retried here: the sender reports the pause its `Retry-After`
header asks for, and the worker holds every call to that tool until then,
since both tools count requests per organisation or per key, not per call
[@langfuse2026apilimits; @langsmith2026retention].

The credentials are the variables the tools' own SDKs read, so a process
that traces to a tool can write its scores with no set-up of its own.
Neither sender imports the tool's SDK.
"""

from __future__ import annotations

import email.utils
import logging
import math
import os
from datetime import UTC, datetime
from typing import Final

import httpx
import stamina

from langchain_sync_monitors.monitors.openrouter_decisions import check_key_characters

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS: Final = 5.0
"""How long one request may take, connecting and reading included."""

RETRY_ATTEMPTS: Final = 2
"""How many times a request is sent before a transport failure or server error leaves it waiting."""

RETRY_BUDGET_SECONDS: Final = 10.0
"""How long stamina keeps retrying one request."""

DEFAULT_PAUSE_SECONDS: Final = 30.0
"""The pause after a `429` whose `Retry-After` is missing or unreadable."""


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


@stamina.retry(on=is_transient_failure, attempts=RETRY_ATTEMPTS, timeout=RETRY_BUDGET_SECONDS)
def send_with_retries(http_client: httpx.Client, *, request: httpx.Request) -> httpx.Response:
    """Send the request, raising on a server error so that stamina retries it."""
    response = http_client.send(request)
    if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
        response.raise_for_status()
    return response


def send_request(http_client: httpx.Client, *, request: httpx.Request) -> httpx.Response | None:
    """Send the request with retries, or return None, logged, when it still failed.

    The log names the request's path and the error's type, never a header.
    """
    try:
        return send_with_retries(http_client, request=request)
    except httpx.HTTPError as error:
        logger.info(
            "score export: %s %s failed with %s",
            request.method,
            request.url.path,
            type(error).__name__,
        )
        return None


def is_rate_limited(response: httpx.Response | None) -> bool:
    """Tell whether the service answered `429`, asking for a pause."""
    return response is not None and response.status_code == httpx.codes.TOO_MANY_REQUESTS


def read_pause_seconds(response: httpx.Response) -> float:
    """Return the pause a `429` answer asks for: its `Retry-After`, in seconds or as a date.

    A missing, unreadable or infinite value gives `DEFAULT_PAUSE_SECONDS`,
    and a date in the past no pause.
    """
    header = response.headers.get("retry-after", "").strip()
    try:
        seconds = float(header)
    except ValueError:
        seconds = read_seconds_until(header)
    if not math.isfinite(seconds):
        return DEFAULT_PAUSE_SECONDS
    return max(seconds, 0.0)


def read_seconds_until(http_date: str) -> float:
    """Return the seconds from now until an HTTP date, or the default pause for anything else."""
    try:
        moment = email.utils.parsedate_to_datetime(http_date)
    except (TypeError, ValueError):
        return DEFAULT_PAUSE_SECONDS
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - datetime.now(UTC)).total_seconds()
