"""Default thresholds warn at the user's constructor call, and the docs state them in each
protocol's guide and on the pages that sum them up; chosen thresholds stay quiet."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    TrustedMonitoring,
)
from langchain_sync_monitors.thresholds import (
    DEFAULT_AUDIT_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_DEFER_THRESHOLD,
    DEFAULT_FLAG_THRESHOLD,
    LIBRARY_DIRECTORY,
    DefaultThreshold,
    UncalibratedThresholdWarning,
    resolve_threshold,
)

DOCS = Path(__file__).resolve().parents[3] / "docs"
HOW_TO_GUIDES = DOCS / "how-to"


@pytest.mark.parametrize("value", [0.0, 0.42, 1.0])
def test_chosen_threshold_is_returned_without_a_warning(value: float) -> None:
    # Arrange
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Act
        resolved = resolve_threshold(parameter_name="defer_threshold", threshold=value)

    # Assert
    assert resolved == value


def test_default_threshold_is_returned_with_a_calibration_hint() -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        resolved = resolve_threshold(
            parameter_name="block_threshold",
            threshold=DefaultThreshold(0.6),
        )

    # Assert
    assert resolved == 0.6
    message = str(record[0].message)
    assert "block_threshold" in message
    assert "honest" in message
    assert "percentile" in message
    assert "CalibratedMonitor" in message


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan])
def test_chosen_threshold_outside_the_unit_interval_is_rejected(value: float) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="audit_threshold must be between 0 and 1"):
        resolve_threshold(parameter_name="audit_threshold", threshold=value)


def test_default_threshold_outside_the_unit_interval_is_rejected() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        resolve_threshold(parameter_name="flag_threshold", threshold=DefaultThreshold(2.0))


@pytest.mark.parametrize(
    ("page", "parameter_name", "default"),
    [
        ("use-trusted-monitoring.md", "flag_threshold", DEFAULT_FLAG_THRESHOLD),
        ("use-auto-mode.md", "block_threshold", DEFAULT_BLOCK_THRESHOLD),
        ("use-defer-to-resample.md", "defer_threshold", DEFAULT_DEFER_THRESHOLD),
        ("use-defer-to-resample.md", "audit_threshold", DEFAULT_AUDIT_THRESHOLD),
        ("use-defer-to-trusted.md", "defer_threshold", DEFAULT_DEFER_THRESHOLD),
        ("use-defer-to-trusted.md", "audit_threshold", DEFAULT_AUDIT_THRESHOLD),
    ],
    ids=[
        "trusted-monitoring-flag",
        "auto-mode-block",
        "defer-to-resample-defer",
        "defer-to-resample-audit",
        "defer-to-trusted-defer",
        "defer-to-trusted-audit",
    ],
)
def test_each_protocol_guide_documents_the_default_threshold(
    page: str,
    parameter_name: str,
    default: DefaultThreshold,
) -> None:
    # Act
    guide = (HOW_TO_GUIDES / page).read_text(encoding="utf-8")

    # Assert
    assert f"| `{parameter_name}` | {default.value}, with a warning |" in guide


@pytest.mark.parametrize(
    ("page", "actions"),
    [
        ("explanation/design.md", "defer, block or flag"),
        ("how-to/choose-a-protocol.md", "flag, defer or block"),
        ("how-to/combine-and-calibrate-monitors.md", "defer, block or flag"),
    ],
    ids=["design", "choose-a-protocol", "combine-and-calibrate-monitors"],
)
def test_each_page_that_sums_up_the_defaults_states_every_default_threshold(
    page: str,
    actions: str,
) -> None:
    # Arrange: each page gives defer, block and flag one shared value.
    shared_default = DEFAULT_DEFER_THRESHOLD.value
    summary = f"{shared_default} to {actions} and {DEFAULT_AUDIT_THRESHOLD.value} to audit"

    # Act
    prose = " ".join((DOCS / page).read_text(encoding="utf-8").split())

    # Assert
    assert DEFAULT_BLOCK_THRESHOLD.value == DEFAULT_FLAG_THRESHOLD.value == shared_default
    assert summary in prose


def test_trusted_monitoring_warns_once_at_the_constructor_call() -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        TrustedMonitoring()

    # Assert
    assert len(record) == 1
    assert record[0].filename == __file__
    assert "flag_threshold" in str(record[0].message)


def test_the_uncalibrated_warning_names_the_default_and_says_how_to_calibrate() -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        AutoMode()

    # Assert
    assert [str(warning.message) for warning in record] == [
        "block_threshold uses the uncalibrated default 0.6. Suspicion scores differ between "
        "monitors and agents, so a fixed default may block honest work or miss attacks. Run "
        "the agent on honest tasks, collect the suspicion scores in monitor_log, and set the "
        "threshold at a high percentile of them (for example the 98th), or wrap the monitor in "
        "CalibratedMonitor so that its scores are honest-run percentiles. Pass block_threshold "
        "explicitly to silence this warning."
    ]


def test_the_warning_points_at_a_caller_whose_path_only_begins_like_the_library_s() -> None:
    # Arrange: a sibling package such as langchain_sync_monitors_extras is not this library
    sibling_file = os.path.join(LIBRARY_DIRECTORY.rstrip(os.sep) + "_extras", "protocols.py")
    build_protocol = compile("AutoMode()", sibling_file, "exec")

    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        exec(build_protocol, {"AutoMode": AutoMode})

    # Assert
    assert [warning.filename for warning in record] == [sibling_file]


def test_auto_mode_warns_once_at_the_constructor_call() -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        AutoMode()

    # Assert
    assert len(record) == 1
    assert record[0].filename == __file__
    assert "block_threshold" in str(record[0].message)


def test_defer_to_resample_warns_for_each_default_threshold(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=defer_to_trusted_model)

    # Assert
    messages = [str(warning.message) for warning in record]
    assert len(messages) == 2
    assert messages[0].startswith("defer_threshold")
    assert messages[1].startswith("audit_threshold")
    assert {warning.filename for warning in record} == {__file__}


def test_turning_auditing_off_leaves_one_warning(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=defer_to_trusted_model, audit_threshold=None)

    # Assert
    assert [str(warning.message).split()[0] for warning in record] == ["defer_threshold"]


def test_defer_to_trusted_warning_skips_the_parent_constructor(
    trusted_model: FakeListChatModel,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToTrusted(trusted_model=trusted_model)

    # Assert
    assert len(record) == 2
    assert {warning.filename for warning in record} == {__file__}


def test_protocols_with_chosen_thresholds_do_not_warn(
    defer_to_trusted_model: DeferToTrustedModel,
    trusted_model: FakeListChatModel,
) -> None:
    # Arrange
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Act
        protocols = [
            TrustedMonitoring(flag_threshold=0.7),
            AutoMode(block_threshold=0.7),
            DeferToResample(
                fallback=defer_to_trusted_model,
                defer_threshold=0.7,
                audit_threshold=0.95,
            ),
            DeferToTrusted(trusted_model=trusted_model, defer_threshold=0.7, audit_threshold=None),
        ]

    # Assert
    assert len(protocols) == 4
