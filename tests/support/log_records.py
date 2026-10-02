"""Find what a log record holds beyond its text: live HTTP objects, and secrets in any value.

A formatter prints a record's message, but a handler can read every attribute,
the extras a library adds included, and a live request in them still holds its
headers, which its repr leaves out. So each value is walked, through mappings
and sequences, rather than read as the repr of the record.
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
)
"""The httpx objects a record must never hold, since each can reach a request's headers.

An httpx error counts: its `request`, and a status error's `response`, are the
live request and answer, though its text quotes neither's headers.
"""


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


def find_leaks(value: object, *, secrets: Collection[str]) -> list[str]:
    """Return the HTTP objects in a value and the parts of it whose text holds a secret."""
    return search_value(value, secrets=secrets, visited={})


def search_value(
    value: object, *, secrets: Collection[str], visited: dict[int, object]
) -> list[str]:
    """Return the leaks in a value, walking each container once, so that a cycle ends.

    `visited` maps each container's id to the container itself, which keeps it
    alive, so no object made during the walk can take a walked container's id.
    """
    if isinstance(value, HTTP_OBJECT_TYPES):
        return [f"a live {type(value).__name__}"]
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        if id(value) in visited:
            return []
        visited[id(value)] = value
        parts = (
            [part for pair in value.items() for part in pair]
            if isinstance(value, Mapping)
            else value
        )
        return [
            leak for part in parts for leak in search_value(part, secrets=secrets, visited=visited)
        ]
    if any(secret in text for text in (str(value), repr(value)) for secret in secrets):
        return [f"a {type(value).__name__} quoting a secret"]
    return []
