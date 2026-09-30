"""Fixtures for the score export tests."""

from collections.abc import Iterator

import pytest
import stamina

from langchain_sync_monitors.score_requests import RETRY_ATTEMPTS
from tests.support.score_services import score_services as score_services


@pytest.fixture(autouse=True)
def stamina_without_waiting() -> Iterator[None]:
    """Make stamina retry as often as the senders ask, without backoff."""
    with stamina.set_testing(True, attempts=RETRY_ATTEMPTS):
        yield
