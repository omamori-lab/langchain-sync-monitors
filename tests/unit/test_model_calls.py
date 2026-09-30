"""Resolving chat models and tagging the library's own model calls."""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model
from langchain_sync_monitors.monitors import GuardModelMonitor, LLMMonitor
from langchain_sync_monitors.protocols import DeferToTrustedModel

GUARD_POLICY = ChatPromptTemplate.from_messages([("human", "{transcript}\n{proposed_step}")])


def test_a_chat_model_instance_is_used_as_given() -> None:
    # Arrange
    model = FakeListChatModel(responses=["ok"])

    # Act
    resolved = resolve_chat_model(model)

    # Assert
    assert resolved is model


def build_llm_monitor(model: BaseChatModel) -> object:
    """Build a chat judge over `model`."""
    return LLMMonitor(model=model)


def build_guard_monitor(model: BaseChatModel) -> object:
    """Build a guard monitor over `model`."""
    return GuardModelMonitor(
        model=model,
        policy_prompt=GUARD_POLICY,
        suspicious_labels={"violation"},
        safe_labels={"no_violation"},
    )


def build_trusted_fallback(model: BaseChatModel) -> object:
    """Build the fallback that hands the step to `model`."""
    return DeferToTrustedModel(trusted_model=model)


@pytest.mark.parametrize(
    "build",
    [resolve_chat_model, build_llm_monitor, build_guard_monitor, build_trusted_fallback],
    ids=["resolve_chat_model", "LLMMonitor", "GuardModelMonitor", "DeferToTrustedModel"],
)
@pytest.mark.parametrize(
    "wrap",
    [lambda model: model.with_retry(), lambda model: model.bind(temperature=0.5)],
    ids=["with_retry", "bind"],
)
def test_a_chat_model_wrapped_in_a_runnable_is_a_configuration_error(
    build: Callable[[BaseChatModel], object],
    wrap: Callable[[BaseChatModel], object],
) -> None:
    # Arrange: the parameter is typed str | BaseChatModel, and a Runnable is neither.
    wrapped = wrap(FakeListChatModel(responses=["ok"]))
    expected = f"got {type(wrapped).__name__}. Pass the chat model itself"

    # Act and Assert
    with pytest.raises(ConfigurationError, match=re.escape(expected)):
        build(wrapped)  # ty: ignore[invalid-argument-type]


def test_internal_call_config_names_its_source() -> None:
    # Act
    config = build_internal_call_config(source="monitor")

    # Assert
    metadata = config.get("metadata", {})
    assert metadata["lc_source"] == "monitor"
    assert len(metadata) > 1


def test_internal_call_config_keeps_the_call_out_of_langsmith_message_view() -> None:
    # Act
    config = build_internal_call_config(source="monitor")

    # Assert
    assert config.get("metadata", {})["ls_message_view_exclude"] is True
    assert "tags" not in config
