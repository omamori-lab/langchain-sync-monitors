"""Resolving chat models and tagging the library's own model calls."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_sync_monitors import model_calls, spans
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model
from langchain_sync_monitors.monitors import GuardModelMonitor, LLMMonitor
from langchain_sync_monitors.protocols import DeferToTrustedModel

GUARD_POLICY = ChatPromptTemplate.from_messages([("human", "{transcript}\n{proposed_step}")])
MONITOR_RUN_NAMES = frozenset(
    {
        "monitor step",
        "monitor judgement",
        "monitor classifier",
        "monitor decision",
        "monitor call",
    },
)
"""The names the tracing guide lists for leaving the monitor's own runs out of a trace."""
TRACING_GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "how-to"
    / "see-decisions-in-langsmith-and-langfuse.md"
)
SPANS_FILTER_ROW = (
    "Everything but the monitor's spans",
    '`and(neq(name, "monitor step"), neq(name, "monitor judgement"), '
    'neq(name, "monitor classifier"), neq(name, "monitor decision"))`',
    "Name none of `monitor step`, `monitor judgement`, `monitor classifier` and `monitor decision`",
)
"""The tracing guide's filter-table row, cell by cell, that leaves out the monitor's spans."""
SPANS_AND_CALLS_FILTER_ROW = (
    "Everything but the monitor's spans and model calls",
    '`and(neq(name, "monitor step"), neq(name, "monitor judgement"), '
    'neq(name, "monitor classifier"), neq(name, "monitor decision"), '
    'neq(name, "monitor call"), neq(metadata_key, "ls_message_view_exclude"))`',
    "Name none of `monitor step`, `monitor judgement`, `monitor classifier`, "
    "`monitor decision` and `monitor call`, which misses the attempts inside a classifier "
    "wrapped in `with_retry()`",
)
"""The row that also leaves out the monitor's calls.

The five names miss the attempts inside a classifier wrapped in `with_retry()`,
which keep the classifier's own name, so the LangSmith filter also drops the
calls' metadata key, and the Langfuse filter, which cannot, says what it misses.
"""
EXCLUDED_NAME_PATTERN = re.compile(r'neq\(name, "(?P<name>[^"]+)"\)')


def test_a_chat_model_instance_is_used_as_given() -> None:
    # Arrange
    model = FakeListChatModel(responses=["ok"])

    # Act
    resolved = resolve_chat_model(model)

    # Assert
    assert resolved is model


def build_llm_monitor(model: BaseChatModel) -> object:
    """Build an LLM monitor over `model`."""
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


def test_internal_call_config_names_the_run_monitor_call() -> None:
    # Act
    config = build_internal_call_config(source="monitor")

    # Assert
    assert config.get("run_name") == "monitor call"


def read_table_rows(page: str) -> list[tuple[str, ...]]:
    """Return every row of every Markdown table on the page, as its stripped cells."""
    return [
        tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        for line in page.splitlines()
        if line.startswith("|")
    ]


def test_the_monitor_s_runs_have_exactly_five_fixed_names() -> None:
    # Act: every fixed run name the library defines, span or model call.
    names = {
        value
        for module in (spans, model_calls)
        for key, value in vars(module).items()
        if key.endswith("_NAME") and isinstance(value, str)
    }

    # Assert
    assert names == MONITOR_RUN_NAMES


def test_the_tracing_guide_s_filter_rows_are_the_ones_pinned_here() -> None:
    # Arrange
    guide = TRACING_GUIDE.read_text(encoding="utf-8")

    # Act
    rows = read_table_rows(guide)

    # Assert
    assert SPANS_FILTER_ROW in rows
    assert SPANS_AND_CALLS_FILTER_ROW in rows


@pytest.mark.parametrize(
    ("row", "expected_names"),
    [
        (SPANS_FILTER_ROW, MONITOR_RUN_NAMES - {model_calls.MONITOR_CALL_NAME}),
        (SPANS_AND_CALLS_FILTER_ROW, MONITOR_RUN_NAMES),
    ],
    ids=["spans", "spans_and_calls"],
)
def test_each_filter_row_names_the_same_runs_in_both_tools(
    row: tuple[str, str, str],
    expected_names: frozenset[str],
) -> None:
    # Arrange
    _, langsmith_filter, langfuse_filter = row

    # Act
    langsmith_names = [match["name"] for match in EXCLUDED_NAME_PATTERN.finditer(langsmith_filter)]
    langfuse_names = re.findall(r"`(monitor [a-z]+)`", langfuse_filter)

    # Assert: each name once, in both tools.
    assert sorted(langsmith_names) == sorted(expected_names)
    assert sorted(langfuse_names) == sorted(expected_names)
