"""Collect log records, and find what one holds beyond its text: live HTTP objects and secrets.

`RecordCollector` keeps every record a handler is handed. A formatter prints
a record's message, but a handler can read every attribute, the extras a
library adds included, and a live request in them still holds its headers,
which its repr leaves out. So each value is walked, through mappings,
sequences and the errors an error was raised from or while handling, rather
than read as the repr of the record.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping

import httpx

HTTP_OBJECT_TYPES = (
    httpx.Request,
    httpx.Response,
    httpx.Headers,
    httpx.Client,
    httpx.AsyncClient,
    httpx.HTTPError,
    httpx.Auth,
)
"""The httpx objects a record must never hold, since each holds a credential or a request's headers.

An httpx error counts: its `request`, and a status error's `response`, are the
live request and answer, though its text quotes neither's headers. An auth
object counts: it holds the credential that becomes a request's
`Authorization` header, though its text quotes no password.
"""

CONTAINER_TYPES = (Mapping, list, tuple, set, frozenset)
"""The values whose parts are searched in place of their text."""

WALKED_ONCE_TYPES = (*CONTAINER_TYPES, BaseException)
"""The values that hold others, which the walk meets once each, an httpx error included."""


class RecordCollector(logging.Handler):
    """Keep every record it is handed, whatever its level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def find_logged_leaks(
    records: Iterable[logging.LogRecord],
    *,
    secrets: Collection[str],
) -> list[str]:
    """Return, for each record, every HTTP object in it and every value that holds a secret."""
    leaks: list[str] = []
    for record in records:
        values = [record.getMessage(), *vars(record).values()]
        leaks.extend(f"{record.name}: {leak}" for leak in find_leaks(values, secrets=secrets))
    return leaks


def find_frame_leaks(
    frame_locals: Iterable[Mapping[str, str]],
    *,
    secrets: Collection[str],
) -> list[str]:
    """Return the name of every frame local whose repr holds a secret."""
    return [
        name
        for locals_of_one_retry in frame_locals
        for name, text in locals_of_one_retry.items()
        if any(secret in text for secret in secrets)
    ]


def find_leaks(value: object, *, secrets: Collection[str]) -> list[str]:
    """Return the HTTP objects in a value and the parts of it whose text holds a secret."""
    return search_value(value, secrets=secrets, visited={})


def search_value(
    value: object, *, secrets: Collection[str], visited: dict[int, object]
) -> list[str]:
    """Return the leaks in a value, meeting each container and each error once.

    So a cycle ends, and an error raised `from` another, which holds it as both
    its cause and its context, reports it once. `visited` maps each container's
    and error's id to the object itself, which keeps it alive, so no object
    made during the walk can take a walked object's id.

    The walk does not reach an error's `args`, an exception group's members or
    a traceback's frames.
    """
    if isinstance(value, WALKED_ONCE_TYPES):
        if id(value) in visited:
            return []
        visited[id(value)] = value
    if isinstance(value, HTTP_OBJECT_TYPES):
        return [f"a live {type(value).__name__}"]
    leaks = [] if isinstance(value, CONTAINER_TYPES) else find_quoted_secret(value, secrets=secrets)
    for part in read_parts(value):
        leaks.extend(search_value(part, secrets=secrets, visited=visited))
    return leaks


def read_parts(value: object) -> list[object]:
    """Return the values a value holds that a handler can reach.

    These are a mapping's keys and values, a collection's items, and the
    errors an error was raised from and while handling, its `__cause__` and
    `__context__`. A record that holds an error holds both, even where a
    traceback hides the context, as `raise ... from None` does.
    """
    if isinstance(value, Mapping):
        return [part for pair in value.items() for part in pair]
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    if isinstance(value, BaseException):
        return [error for error in (value.__cause__, value.__context__) if error is not None]
    return []


def find_quoted_secret(value: object, *, secrets: Collection[str]) -> list[str]:
    """Return a leak for a value whose text quotes a secret, and none otherwise."""
    if any(secret in text for text in (str(value), repr(value)) for secret in secrets):
        return [f"a {type(value).__name__} quoting a secret"]
    return []
