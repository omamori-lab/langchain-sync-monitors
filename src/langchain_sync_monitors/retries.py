"""Retries whose hooks see only a stand-in for the error, never the error itself.

stamina hands the error it retries on to every retry hook, and its logging
hook logs the error's repr [@schlawack2026stamina]. An error's repr can hold
what a call sent or received. OpenRouter's `TooManyRequestsResponseError` is a
dataclass whose repr holds the whole reply: its body, the account's `user_id`
included, and its headers [@openrouterpythonsdk2026]. httpx's
`RemoteProtocolError` quotes the bytes of a malformed reply it could not
parse, such as a header line [@httpx2024; @smith2025h11]. And a hook that
reads more than the repr finds an httpx error's request, which holds the key
in its headers and the transcript in its body.

So each retry site retries its block through `call_with_retries`: a failure
the site retries is kept out of the hooks' reach, and a `RetriedCallError` is
raised in its place, which names the error's type and HTTP status alone.
After the last attempt the kept error is raised, so the caller sees the error
it would see without the stand-in. Every other error is raised at once, as it
is.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import TypedDict, Unpack

import httpx
import stamina

from langchain_sync_monitors.errors import RetriedCallError

type RetriedErrorPredicate = Callable[[Exception], bool]
"""Tells whether a retry site retries an error, as stamina's `on` predicate would."""


class RetryOptions(TypedDict, total=False):
    """The stamina settings a retry site sets; any it leaves out keeps stamina's default."""

    attempts: int
    timeout: float
    wait_initial: float


def read_http_status(error: Exception) -> int | None:
    """Return the error's HTTP status, or None when it carries none as an integer.

    httpx puts the status on the error's response [@httpx2024]; provider SDKs
    put it on the error as `status_code` [@openrouterpythonsdk2026].
    """
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def build_stand_in(error: Exception) -> RetriedCallError:
    """Return the stand-in for an error a site retries: its type's name and its HTTP status."""
    return RetriedCallError(error_type=type(error).__name__, http_status=read_http_status(error))


async def run_attempt[Result](
    block: Callable[[], Awaitable[Result]],
    *,
    is_retried: RetriedErrorPredicate,
    failures: list[Exception],
) -> Result:
    """Await the block once; on an error the site retries, keep it and raise its stand-in.

    Any other error is raised as it is. The stand-in is raised after the
    handler that caught the error has ended, so the error is not its
    `__context__`: `from None` only hides a chained error from a traceback,
    and a hook could still read it.
    """
    try:
        return await block()
    except Exception as error:
        if not is_retried(error):
            raise
        failures.append(error)
    raise build_stand_in(failures[-1]) from None


def run_attempt_sync[Result](
    block: Callable[[], Result],
    *,
    is_retried: RetriedErrorPredicate,
    failures: list[Exception],
) -> Result:
    """Run the block once without an event loop, as `run_attempt` awaits it."""
    try:
        return block()
    except Exception as error:
        if not is_retried(error):
            raise
        failures.append(error)
    raise build_stand_in(failures[-1]) from None


async def call_with_retries[Result](
    block: Callable[[], Awaitable[Result]],
    *,
    is_retried: RetriedErrorPredicate,
    **options: Unpack[RetryOptions],
) -> Result:
    """Await the block, again after each error `is_retried` accepts, as stamina's options say.

    The retries wrap a block, not a function, so no hook is handed the
    block's arguments, and they retry on the stand-in, so no hook is handed
    the error either. After the last attempt, the error from it is raised,
    outside the handler, so nothing is chained to it.
    """
    failures: list[Exception] = []
    with contextlib.suppress(RetriedCallError):
        async for attempt in stamina.retry_context(on=RetriedCallError, **options):
            with attempt:
                result = await run_attempt(block, is_retried=is_retried, failures=failures)
        return result
    raise failures[-1]


def call_with_retries_sync[Result](
    block: Callable[[], Result],
    *,
    is_retried: RetriedErrorPredicate,
    **options: Unpack[RetryOptions],
) -> Result:
    """Run the block without an event loop, as `call_with_retries` awaits it."""
    failures: list[Exception] = []
    with contextlib.suppress(RetriedCallError):
        for attempt in stamina.retry_context(on=RetriedCallError, **options):
            with attempt:
                result = run_attempt_sync(block, is_retried=is_retried, failures=failures)
        return result
    raise failures[-1]
