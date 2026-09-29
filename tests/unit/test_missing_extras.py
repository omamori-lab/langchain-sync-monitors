"""Features behind an optional extra fail with one error type that names the extra."""

from __future__ import annotations

import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors import (
    ConfigurationError,
    LLMMonitor,
    MissingExtraError,
    MonitorMiddleware,
    TrustedMonitoring,
    TypeSafeDecisionModel,
    monitor_subagents,
)


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=LLMMonitor(model=FakeListChatModel(responses=["unused"])),
        protocol=TrustedMonitoring(flag_threshold=0.6),
    )


def test_monitor_subagents_without_deep_agents_names_the_extra(
    middleware: MonitorMiddleware,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    for module_name in ("deepagents", "deepagents.middleware", "deepagents.middleware.subagents"):
        monkeypatch.setitem(sys.modules, module_name, None)

    # Act / Assert
    with pytest.raises(MissingExtraError, match=r"langchain-sync-monitors\[deepagents\]"):
        monitor_subagents(middleware=middleware)


def test_typesafe_adapter_without_its_extra_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setitem(sys.modules, "langchain_typesafe", None)

    # Act / Assert
    with pytest.raises(MissingExtraError, match=r"langchain-sync-monitors\[typesafe\]"):
        TypeSafeDecisionModel(classifier=object())  # ty: ignore[invalid-argument-type]


def test_a_missing_extra_is_both_a_configuration_and_an_import_error() -> None:
    # Act
    error = MissingExtraError("pip install 'langchain-sync-monitors[deepagents]'")

    # Assert
    assert isinstance(error, ConfigurationError)
    assert isinstance(error, ImportError)
