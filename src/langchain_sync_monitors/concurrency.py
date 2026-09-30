"""Run the library's own concurrent calls so that one failure stops the rest.

Parallel resampling, `RepeatedMonitor` and a chat monitor that draws several
replies, such as a guard model that samples its label, make several model calls
at once. `asyncio.gather` lets the other calls run on, and spend tokens, after
one of them has failed; an `asyncio.TaskGroup` cancels them instead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Iterable

logger = logging.getLogger(__name__)


async def run_concurrently[ResultT](
    coroutines: Iterable[Coroutine[object, object, ResultT]],
) -> list[ResultT]:
    """Run the coroutines at once and return their results in order.

    When one raises, the task group cancels the others, and the first failure
    is raised as it is, not wrapped in an `ExceptionGroup`, so a caller sees
    the same error as from a single call. Any further failures are logged.
    """
    first_failure: BaseException | None = None
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coroutine) for coroutine in coroutines]
    except BaseExceptionGroup as failures:
        first_failure, *other_failures = failures.exceptions
        for other_failure in other_failures:
            logger.warning("A concurrent call also failed: %r", other_failure)
    # Raised outside the `except`, so the error is not chained onto the exception group.
    if first_failure is not None:
        raise first_failure
    return [task.result() for task in tasks]
