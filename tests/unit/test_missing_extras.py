"""Features behind an optional extra fail with one error type that names the extra."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

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
from langchain_sync_monitors.deepagents import INSTALL_HINT
from langchain_sync_monitors.model_calls import resolve_chat_model

OPENROUTER_MODEL = "openrouter:vendor/model"
DEEP_AGENTS_MODULES = ("deepagents", "deepagents.middleware", "deepagents.middleware.subagents")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def middleware() -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=LLMMonitor(model=FakeListChatModel(responses=["unused"])),
        protocol=TrustedMonitoring(audit_threshold=0.6),
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


@pytest.mark.parametrize(
    ("missing_modules", "use_feature", "expected_message"),
    [
        (
            DEEP_AGENTS_MODULES,
            lambda middleware: monitor_subagents(middleware=middleware),
            "monitor_subagents needs the deepagents extra. Install it with: "
            'uv add "langchain-sync-monitors[deepagents]" '
            '(or pip install "langchain-sync-monitors[deepagents]")',
        ),
        (
            ("langchain_openrouter",),
            lambda _middleware: LLMMonitor(model=OPENROUTER_MODEL),
            "An 'openrouter:' model string needs the openrouter extra. Install it with: "
            'uv add "langchain-sync-monitors[openrouter]" '
            '(or pip install "langchain-sync-monitors[openrouter]")',
        ),
        (
            ("langchain_typesafe",),
            lambda _middleware: TypeSafeDecisionModel(classifier=object()),  # ty: ignore[invalid-argument-type]
            "TypeSafeDecisionModel needs the typesafe extra. Install it with: "
            'uv add "langchain-sync-monitors[typesafe]" '
            '(or pip install "langchain-sync-monitors[typesafe]")',
        ),
    ],
    ids=["deepagents", "openrouter", "typesafe"],
)
def test_every_missing_extra_names_the_uv_and_the_pip_command(
    missing_modules: tuple[str, ...],
    use_feature: Callable[[MonitorMiddleware], object],
    expected_message: str,
    middleware: MonitorMiddleware,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    for module_name in missing_modules:
        monkeypatch.setitem(sys.modules, module_name, None)

    # Act
    with pytest.raises(MissingExtraError) as refusal:
        use_feature(middleware)

    # Assert
    assert str(refusal.value) == expected_message


def test_the_subagents_guide_quotes_the_deep_agents_message_word_for_word() -> None:
    # Arrange
    guide = REPOSITORY_ROOT / "docs" / "how-to" / "monitor-deep-agents-subagents.md"

    # Act
    # A quote wrapped across lines renders with a space at each line break.
    text = " ".join(guide.read_text(encoding="utf-8").split())

    # Assert
    assert f"`{INSTALL_HINT}`" in text


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
    error = MissingExtraError(INSTALL_HINT)

    # Assert
    assert isinstance(error, ConfigurationError)
    assert isinstance(error, ImportError)
