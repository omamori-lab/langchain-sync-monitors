"""Fixtures of the live harness's tests. The Deep Agents test is not collected without the extra."""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

import httpx
import pytest

import tests.live.harness as harness_module
from tests.live.costs import build_async_decisions_client, build_decisions_client
from tests.live.fakes import FakeModelFactory, build_decisions_answer

collect_ignore = [] if importlib.util.find_spec("deepagents") else ["test_deep_agents_subagents.py"]


@pytest.fixture(scope="session")
def report_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return where the live runs write their JSON reports: `LIVE_REPORT_DIR`, else a temp dir."""
    configured = os.environ.get("LIVE_REPORT_DIR")
    directory = Path(configured) if configured else tmp_path_factory.mktemp("live-reports")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> FakeModelFactory:
    """Swap every model of the harness for a fake, and the Decisions API for a mock transport."""
    factory = FakeModelFactory()
    transport = httpx.MockTransport(build_decisions_answer)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(harness_module, "build_chat_model", factory.build_chat_model)
    monkeypatch.setattr(
        harness_module,
        "build_decisions_client",
        functools.partial(build_decisions_client, transport=transport),
    )
    monkeypatch.setattr(
        harness_module,
        "build_async_decisions_client",
        functools.partial(build_async_decisions_client, transport=transport),
    )
    return factory
