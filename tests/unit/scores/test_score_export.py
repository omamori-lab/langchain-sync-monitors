"""The `export_scores` option: checked when the monitor is built, and needing no environment key.

The option is off by default and takes a set of `Tracer` members. Langfuse is
refused with `MissingExtraError` when its package is missing; the tracers
hold their own credentials, so none is read from the environment. What is
sent for a run is tested through the agent, in `tests/integration/test_step_scores.py`.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest

from langchain_sync_monitors import AuditResponse, score_export
from langchain_sync_monitors.contracts import TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError, MissingExtraError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample
from langchain_sync_monitors.score_export import Tracer, is_langfuse_handler
from tests.support.agents import build_keyword_monitor
from tests.support.score_backends import FakeLangfuseHandler
from tests.support.tracing import RecordingTracer

CREDENTIAL_VARIABLES = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


def build_monitor(**options: Any) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=DeferToResample(
            defer_threshold=0.8,
            audit_threshold=0.8,
            max_resamples=0,
            on_audit=AuditResponse.FLAG,
        ),
        **options,
    )


@pytest.fixture(autouse=True)
def without_credentials_and_with_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave no tool's key in the environment, and count Langfuse's package as installed."""
    for name in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(score_export, "is_package_installed", lambda name: True)


def test_export_scores_is_off_by_default() -> None:
    # Act
    monitor = build_monitor()

    # Assert
    assert monitor.export_scores == frozenset()


def test_the_tools_asked_for_are_kept_as_a_frozenset() -> None:
    # Act
    monitor = build_monitor(export_scores={Tracer.LANGSMITH, Tracer.LANGFUSE})

    # Assert
    assert monitor.export_scores == frozenset({Tracer.LANGSMITH, Tracer.LANGFUSE})
    assert isinstance(monitor.export_scores, frozenset)


@pytest.mark.parametrize(
    "value",
    [Tracer.LANGSMITH, "langsmith", [Tracer.LANGSMITH], {"langsmith"}, None],
    ids=["a-member", "a-string", "a-list", "a-set-of-strings", "none"],
)
def test_anything_but_a_set_of_tracers_is_refused(value: object) -> None:
    # Act, Assert
    with pytest.raises(ConfigurationError, match="export_scores"):
        build_monitor(export_scores=value)


def test_a_monitor_exports_to_langsmith_with_no_key_in_the_environment() -> None:
    # Act
    monitor = build_monitor(export_scores={Tracer.LANGSMITH})

    # Assert
    assert monitor.export_scores == frozenset({Tracer.LANGSMITH})


def test_a_monitor_exports_to_langfuse_with_no_key_in_the_environment() -> None:
    # Act
    monitor = build_monitor(export_scores={Tracer.LANGFUSE})

    # Assert
    assert monitor.export_scores == frozenset({Tracer.LANGFUSE})


def test_langfuse_without_its_package_raises_the_missing_extra_error_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(score_export, "is_package_installed", lambda name: False)

    # Act, Assert
    with pytest.raises(MissingExtraError, match=r'needs the langfuse extra.*\[langfuse\]"'):
        build_monitor(export_scores={Tracer.LANGFUSE})


def test_langsmith_alone_needs_no_langfuse_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(score_export, "is_package_installed", lambda name: False)

    # Act
    monitor = build_monitor(export_scores={Tracer.LANGSMITH})

    # Assert
    assert monitor.export_scores == frozenset({Tracer.LANGSMITH})


def test_a_subagent_monitor_and_a_copied_one_keep_the_option_and_it_pickles() -> None:
    # Arrange
    monitor = build_monitor(export_scores={Tracer.LANGFUSE})

    # Act
    subagent_monitor = monitor.copy_for_subagent(subagent_name="researcher")
    # A deep copy rebuilds the monitor through the same reduce protocol pickle uses.
    copied = copy.deepcopy(monitor)
    pickled = pickle.dumps(monitor)

    # Assert
    assert subagent_monitor.export_scores == {Tracer.LANGFUSE}
    assert subagent_monitor.task_author is TaskAuthor.PARENT_AGENT
    assert copied is not monitor
    assert copied.export_scores == {Tracer.LANGFUSE}
    assert pickled.startswith(pickle.PROTO)


def test_langfuse_handler_is_recognised_by_its_package_and_so_is_a_subclass() -> None:
    # Arrange
    class OwnLangfuseHandler(FakeLangfuseHandler):
        """A user's handler built on Langfuse's."""

    # Act
    recognised = [
        is_langfuse_handler(FakeLangfuseHandler()),
        is_langfuse_handler(OwnLangfuseHandler()),
        is_langfuse_handler(RecordingTracer()),
    ]

    # Assert
    assert recognised == [True, True, False]
