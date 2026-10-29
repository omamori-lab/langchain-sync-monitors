"""The library raises hyoshigi's `ConfigurationError`, which is not a `MonitorError`."""

from __future__ import annotations

import sys
from collections.abc import Callable

import hyoshigi
import pytest

import langchain_sync_monitors
from langchain_sync_monitors import (
    DecisionMonitor,
    DeferToResample,
    FeedbackTemplate,
    LLMMonitor,
    MissingExtraError,
    MonitorError,
    MonitorMiddleware,
)
from tests.support.chat_models import ScriptedChatModel

OPENROUTER_MODEL = "openrouter:vendor/model"


def test_the_exported_configuration_error_is_hyoshigis() -> None:
    # Act
    exported = langchain_sync_monitors.ConfigurationError

    # Assert
    assert exported is hyoshigi.ConfigurationError


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: FeedbackTemplate(text=42),  # ty: ignore[invalid-argument-type]
            id="an-option-of-the-wrong-type",
        ),
        pytest.param(
            lambda: DecisionMonitor(model="anthropic:claude-haiku-5-5"),
            id="a-decision-model-of-an-unknown-provider",
        ),
        pytest.param(
            lambda: LLMMonitor(
                model=ScriptedChatModel(responses=[]),
                lowest_score=5,
                highest_score=1,
            ),
            id="a-monitor-option-hyoshigi-refuses",
        ),
        pytest.param(
            lambda: MonitorMiddleware(
                monitor=LLMMonitor(model=ScriptedChatModel(responses=[])),
                protocol=DeferToResample(
                    defer_threshold=0.9,
                    audit_threshold=0.5,
                    max_resamples=0,
                ),
            ),
            id="a-protocol-setting-hyoshigi-refuses",
        ),
    ],
)
def test_an_error_raised_when_building_is_hyoshigis_configuration_error(
    build: Callable[[], object],
) -> None:
    # Act
    with pytest.raises(hyoshigi.ConfigurationError) as raised:
        build()

    # Assert
    assert (type(raised.value), raised.value.__cause__) == (hyoshigi.ConfigurationError, None)


def test_a_missing_extra_is_hyoshigis_configuration_error_but_not_a_monitor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setitem(
        sys.modules,
        "langchain_openrouter",
        None,
    )

    # Act
    with pytest.raises(hyoshigi.ConfigurationError) as raised:
        LLMMonitor(model=OPENROUTER_MODEL)

    # Assert
    assert isinstance(
        raised.value,
        MissingExtraError,
    )
    assert not isinstance(
        raised.value,
        MonitorError,
    )
