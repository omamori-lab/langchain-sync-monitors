"""Fixtures shared by the agent-level test directories, re-exported by their conftest files."""

from __future__ import annotations

from typing import cast

import pytest

from tests.support.agents import RunMode


@pytest.fixture(params=["invoke", "ainvoke"])
def run_mode(request: pytest.FixtureRequest) -> RunMode:
    """Run each agent test once under `invoke()` and once under `ainvoke()`."""
    return cast("RunMode", request.param)
