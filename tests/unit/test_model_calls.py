"""Resolving chat models and tagging the library's own model calls."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model


def test_a_chat_model_instance_is_used_as_given() -> None:
    # Arrange
    model = FakeListChatModel(responses=["ok"])

    # Act
    resolved = resolve_chat_model(model)

    # Assert
    assert resolved is model


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
