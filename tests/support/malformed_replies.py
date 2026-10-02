"""A reply httpx cannot parse, whose error quotes the bytes it received, for the retry tests."""

from __future__ import annotations

from collections.abc import Callable

import httpx

PLANTED_REPLY_HEADER = "planted-reply-header-77d0e2"
"""A value in a header line of a reply httpx cannot parse."""


def build_malformed_reply_error() -> httpx.RemoteProtocolError:
    """Build the error httpx raises for a reply whose header line has no colon.

    The message is h11's, which quotes the line; httpx raised it, word for
    word, for such a reply from a socket server on localhost.
    """
    line = f"x-request-id {PLANTED_REPLY_HEADER}"
    return httpx.RemoteProtocolError(f"illegal header line: bytearray(b'{line}')")


def raise_on_send(error: Exception) -> Callable[[httpx.Request], httpx.Response]:
    """Build a `MockTransport` handler that raises `error` for any request."""

    def fail(_request: httpx.Request) -> httpx.Response:
        raise error

    return fail


def fail_with_a_malformed_reply(_request: httpx.Request) -> httpx.Response:
    """Fail, for any request, as httpx fails on a reply whose header line has no colon."""
    raise build_malformed_reply_error()
