"""Default thresholds warn at the user's constructor call, and the docs state them in each
protocol's guide and on the pages that sum them up; chosen thresholds stay quiet."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from types import FunctionType

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.contracts import Resampling
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    TrustedMonitoring,
)
from langchain_sync_monitors.thresholds import (
    AUDIT_THRESHOLD_SECTION_TITLE,
    DEFAULT_AUDIT_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_DEFER_THRESHOLD,
    DEFAULT_FLAG_THRESHOLD,
    DEFER_TO_RESAMPLE_GUIDE_TITLE,
    LIBRARY_DIRECTORY,
    DefaultThreshold,
    ThresholdStatistic,
    UncalibratedThresholdWarning,
    build_uncalibrated_message,
    resolve_threshold,
)

DOCS = Path(__file__).resolve().parents[3] / "docs"
HOW_TO_GUIDES = DOCS / "how-to"

SINGLE_SAMPLE_AUDIT_WARNING = (
    "audit_threshold uses the uncalibrated default 0.9. Suspicion scores differ between "
    "monitors and agents, so a fixed default may block honest work or miss attacks. Run "
    "the agent on honest tasks, collect the suspicion scores in monitor_log, and set the "
    "threshold at a high percentile of them (for example the 98th), or wrap the monitor in "
    "CalibratedMonitor so that its scores are honest-run percentiles. Pass audit_threshold "
    "explicitly to silence this warning."
)
LOWEST_SAMPLE_SCORE_AUDIT_WARNING = (
    "audit_threshold uses the uncalibrated default 0.9. Suspicion scores differ between "
    "monitors and agents, so a fixed default may block honest work or miss attacks. "
    "audit_threshold is compared with the lowest suspicion score across a step's samples, "
    "which is never above the first sample's score, so a percentile of single-sample scores "
    "sets it too high. Run the agent on honest tasks under the protocol and settings you "
    "will deploy, take each step's lowest score from monitor_log, and set the threshold at "
    'a high percentile of them (for example the 99th), as the section "Calibrate the audit '
    'threshold" of the guide "Use Defer to Resample" shows. Pass audit_threshold explicitly '
    "to silence this warning."
)


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


def build_default_auto_mode() -> AutoMode:
    return AutoMode()


def test_the_warning_points_at_a_caller_whose_path_only_begins_like_the_library_s() -> None:
    # Arrange: a sibling package such as langchain_sync_monitors_extras is not this library
    sibling_file = os.path.join(LIBRARY_DIRECTORY.rstrip(os.sep) + "_extras", "protocols.py")
    code_in_sibling = build_default_auto_mode.__code__.replace(co_filename=sibling_file)
    build_in_sibling = FunctionType(code_in_sibling, build_default_auto_mode.__globals__)

    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        build_in_sibling()

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


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        (ThresholdStatistic.SINGLE_SAMPLE_SCORE, SINGLE_SAMPLE_AUDIT_WARNING),
        (ThresholdStatistic.LOWEST_SAMPLE_SCORE, LOWEST_SAMPLE_SCORE_AUDIT_WARNING),
    ],
    ids=["single-sample-score", "lowest-sample-score"],
)
def test_the_uncalibrated_warning_advises_calibrating_on_the_statistic_given(
    statistic: ThresholdStatistic,
    expected: str,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        resolved = resolve_threshold(
            parameter_name="audit_threshold",
            threshold=DefaultThreshold(0.9),
            statistic=statistic,
        )

    # Assert
    assert resolved == 0.9
    assert [str(warning.message) for warning in record] == [expected]


def test_resolve_threshold_defaults_to_the_single_sample_advice() -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        resolve_threshold(parameter_name="audit_threshold", threshold=DefaultThreshold(0.9))

    # Assert
    assert [str(warning.message) for warning in record] == [SINGLE_SAMPLE_AUDIT_WARNING]


def test_defer_to_resample_audit_warning_advises_the_lowest_score_across_samples(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Act: the repro of issue 108, with the default two resamples
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=defer_to_trusted_model, defer_threshold=0.5)

    # Assert
    assert [str(warning.message) for warning in record] == [LOWEST_SAMPLE_SCORE_AUDIT_WARNING]
    assert [warning.filename for warning in record] == [__file__]
    assert "CalibratedMonitor" not in str(record[0].message)


def read_lowest_score_audit_warning(fallback: DeferToTrustedModel) -> str:
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=fallback, defer_threshold=0.5)
    return str(record[0].message)


def read_defer_to_resample_guide_lines() -> list[str]:
    return (HOW_TO_GUIDES / "use-defer-to-resample.md").read_text(encoding="utf-8").splitlines()


def test_the_lowest_score_advice_names_the_defer_to_resample_guide_by_its_title(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Arrange
    warning = read_lowest_score_audit_warning(defer_to_trusted_model)

    # Act
    guide_lines = read_defer_to_resample_guide_lines()

    # Assert
    assert f'the guide "{DEFER_TO_RESAMPLE_GUIDE_TITLE}"' in warning
    assert guide_lines[0] == f"# {DEFER_TO_RESAMPLE_GUIDE_TITLE}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="pull request 104 adds this section to the guide; remove this mark once it is on main",
)
def test_the_lowest_score_advice_names_a_section_of_the_defer_to_resample_guide(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Arrange
    warning = read_lowest_score_audit_warning(defer_to_trusted_model)

    # Act
    guide_lines = read_defer_to_resample_guide_lines()

    # Assert
    assert f'the section "{AUDIT_THRESHOLD_SECTION_TITLE}"' in warning
    assert f"### {AUDIT_THRESHOLD_SECTION_TITLE}" in guide_lines


@pytest.mark.parametrize("resampling", list(Resampling), ids=[mode.value for mode in Resampling])
def test_one_resample_is_enough_for_the_lowest_score_advice(
    defer_to_trusted_model: DeferToTrustedModel,
    resampling: Resampling,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(
            fallback=defer_to_trusted_model,
            defer_threshold=0.5,
            max_resamples=1,
            resampling=resampling,
        )

    # Assert
    assert [str(warning.message) for warning in record] == [LOWEST_SAMPLE_SCORE_AUDIT_WARNING]


def test_without_resamples_the_audit_warning_keeps_the_single_sample_advice(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=defer_to_trusted_model, defer_threshold=0.5, max_resamples=0)

    # Assert
    assert [str(warning.message) for warning in record] == [SINGLE_SAMPLE_AUDIT_WARNING]
    assert "lowest" not in str(record[0].message)


def test_defer_to_trusted_audit_warning_keeps_the_single_sample_advice(
    trusted_model: FakeListChatModel,
) -> None:
    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToTrusted(trusted_model=trusted_model, defer_threshold=0.5)

    # Assert
    assert [str(warning.message) for warning in record] == [SINGLE_SAMPLE_AUDIT_WARNING]


def test_resampling_leaves_the_defer_threshold_warning_on_single_samples(
    defer_to_trusted_model: DeferToTrustedModel,
) -> None:
    # Arrange: each sample is compared with defer_threshold on its own
    expected = build_uncalibrated_message(
        parameter_name="defer_threshold",
        value=0.6,
        statistic=ThresholdStatistic.SINGLE_SAMPLE_SCORE,
    )

    # Act
    with pytest.warns(UncalibratedThresholdWarning) as record:
        DeferToResample(fallback=defer_to_trusted_model, audit_threshold=0.9, max_resamples=2)

    # Assert
    assert [str(warning.message) for warning in record] == [expected]
    assert "CalibratedMonitor" in expected


@pytest.mark.parametrize(
    ("max_resamples", "statistic"),
    [
        (0, ThresholdStatistic.SINGLE_SAMPLE_SCORE),
        (1, ThresholdStatistic.LOWEST_SAMPLE_SCORE),
        (2, ThresholdStatistic.LOWEST_SAMPLE_SCORE),
    ],
)
def test_the_audit_statistic_follows_the_resample_count(
    defer_to_trusted_model: DeferToTrustedModel,
    max_resamples: int,
    statistic: ThresholdStatistic,
) -> None:
    # Arrange
    protocol = DeferToResample(
        fallback=defer_to_trusted_model,
        defer_threshold=0.5,
        audit_threshold=0.9,
        max_resamples=max_resamples,
    )

    # Act
    audit_statistic = protocol.audit_statistic

    # Assert
    assert audit_statistic is statistic


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
