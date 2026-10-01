"""Features behind an optional extra fail with one error type that names the extra."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors import (
    ConfigurationError,
    DeferToTrustedModel,
    LLMMonitor,
    MissingExtraError,
    MonitorMiddleware,
    TrustedMonitoring,
    TypeSafeDecisionModel,
    monitor_subagents,
)
from langchain_sync_monitors.model_calls import resolve_chat_model

OPENROUTER_MODEL = "openrouter:vendor/model"


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

    # Act
    with pytest.raises(MissingExtraError) as refusal:
        TypeSafeDecisionModel(classifier=object())  # ty: ignore[invalid-argument-type]

    # Assert
    assert str(refusal.value) == (
        "TypeSafeDecisionModel needs the typesafe extra: "
        "pip install 'langchain-sync-monitors[typesafe]'"
    )


@pytest.mark.parametrize(
    "build_with_model",
    [
        lambda model: LLMMonitor(model=model),
        lambda model: DeferToTrustedModel(trusted_model=model),
    ],
    ids=["monitor", "trusted-model-fallback"],
)
def test_an_openrouter_model_string_without_its_extra_names_the_extra(
    build_with_model: Callable[[str], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setitem(sys.modules, "langchain_openrouter", None)

    # Act / Assert
    with pytest.raises(MissingExtraError, match=r"langchain-sync-monitors\[openrouter\]"):
        build_with_model(OPENROUTER_MODEL)


def test_another_provider_without_its_package_keeps_the_langchain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setitem(sys.modules, "langchain_mistralai", None)

    # Act
    with pytest.raises(ImportError, match="langchain-mistralai") as caught:
        resolve_chat_model("mistralai:mistral-large-latest")

    # Assert
    assert not isinstance(caught.value, MissingExtraError)


def test_an_openrouter_model_string_with_its_extra_builds_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    langchain_openrouter = pytest.importorskip("langchain_openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")

    # Act
    model = resolve_chat_model(OPENROUTER_MODEL)

    # Assert
    assert isinstance(model, langchain_openrouter.ChatOpenRouter)


def test_a_missing_extra_is_both_a_configuration_and_an_import_error() -> None:
    # Act
    error = MissingExtraError("pip install 'langchain-sync-monitors[deepagents]'")

    # Assert
    assert isinstance(error, ConfigurationError)
    assert isinstance(error, ImportError)
