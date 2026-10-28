"""The placement check names a monitor whose place inside another monitor corrupts the records.

`test_stacked_monitors` runs each of these stacks in an agent and shows which
ones lose or pile up records; this file checks that the warning names exactly
those.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from hyoshigi import FailureResponse
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_sync_monitors import AuditResponse, BlockAndFeedBack, DeferToResample
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
from langchain_sync_monitors.protocols import ProtocolConfiguration
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.unit.middleware.test_placement import (
    CommandingMiddleware,
    UnsupportedContentMiddleware,
)

THRESHOLD = 0.5
DEFER_BELOW_AUDIT = 0.9
"""An audit threshold above `THRESHOLD`, so a step between the two asks for a trusted step."""

type MonitorBuilder = Callable[[str], MonitorMiddleware]


def build_monitor(
    protocol: ProtocolConfiguration,
    *,
    label: str,
    trusted_model: BaseChatModel | None = None,
) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=protocol,
        label=label,
        trusted_model=trusted_model,
    )


def build_trusted_model() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[AIMessage("safe")])


def build_trusted_monitoring() -> DeferToResample:
    return DeferToResample(
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
        max_resamples=0,
        on_audit=AuditResponse.FLAG,
    )


def build_resampling(*, max_resamples: int) -> DeferToResample:
    return DeferToResample(
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
        max_resamples=max_resamples,
    )


def build_deferral() -> DeferToResample:
    return DeferToResample(
        defer_threshold=THRESHOLD,
        audit_threshold=DEFER_BELOW_AUDIT,
        max_resamples=0,
    )


SINGLE_CALL_MONITORS: dict[str, MonitorBuilder] = {
    "trusted-monitoring": lambda label: build_monitor(build_trusted_monitoring(), label=label),
    "audit-at-the-defer-threshold": lambda label: build_monitor(
        build_resampling(max_resamples=0),
        label=label,
    ),
    "defer-without-a-trusted-model": lambda label: build_monitor(build_deferral(), label=label),
}
"""The settings that call the rest of the stack at most once per step."""

REPEATING_MONITORS: dict[str, MonitorBuilder] = {
    "defer-to-trusted": lambda label: build_monitor(
        build_deferral(),
        label=label,
        trusted_model=build_trusted_model(),
    ),
    "resample-once": lambda label: build_monitor(build_resampling(max_resamples=1), label=label),
    "block-when-no-answer": lambda label: build_monitor(
        DeferToResample(
            defer_threshold=THRESHOLD,
            audit_threshold=THRESHOLD,
            max_resamples=0,
            on_unavailable=FailureResponse.BLOCK,
        ),
        label=label,
    ),
    "block-and-feed-back": lambda label: build_monitor(
        BlockAndFeedBack(block_threshold=THRESHOLD),
        label=label,
    ),
    "block-and-feed-back-one-in-a-row": lambda label: build_monitor(
        BlockAndFeedBack(block_threshold=THRESHOLD, blocks_in_a_row=1),
        label=label,
    ),
    "block-and-feed-back-one-in-total": lambda label: build_monitor(
        BlockAndFeedBack(block_threshold=THRESHOLD, blocks_in_total=1),
        label=label,
    ),
}
"""The settings that may call the rest of the stack again, each next to a sound one."""


def check_without_warnings(stack: Sequence[AgentMiddleware[Any, Any, Any]]) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return check_monitor_placement(middleware=stack)


@pytest.mark.parametrize("build_outer", SINGLE_CALL_MONITORS.values(), ids=SINGLE_CALL_MONITORS)
def test_a_monitor_inside_one_that_calls_the_model_once_is_not_named(
    build_outer: MonitorBuilder,
) -> None:
    # Arrange
    inner_protocol = DeferToResample(
        defer_threshold=THRESHOLD,
        audit_threshold=DEFER_BELOW_AUDIT,
        max_resamples=2,
    )
    stack = [
        build_outer("outer"),
        build_monitor(inner_protocol, label="inner", trusted_model=build_trusted_model()),
    ]

    # Act
    named = check_without_warnings(stack)

    # Assert
    assert named == []


@pytest.mark.parametrize("build_outer", REPEATING_MONITORS.values(), ids=REPEATING_MONITORS)
def test_a_monitor_inside_one_that_can_call_the_model_again_is_named(
    build_outer: MonitorBuilder,
) -> None:
    # Arrange
    outer = build_outer("outer")
    stack = [outer, build_monitor(build_trusted_monitoring(), label="inner")]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="more than once in a step") as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]"]
    message = str(caught[0].message)
    assert message.startswith("inner[main] sits inside outer[main]")
    assert f"({type(outer.protocol).__name__})" in message
    assert "never reach monitor_log" in message


def test_the_warning_for_a_monitor_inside_one_that_can_call_again_reads_in_full() -> None:
    # Arrange
    outer = build_monitor(build_resampling(max_resamples=2), label="outer")
    stack = [outer, build_monitor(build_trusted_monitoring(), label="inner")]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        check_monitor_placement(middleware=stack)

    # Assert: it recommends the single-call settings and a combined monitor, never a cascade
    [message] = [str(warning.message) for warning in caught]
    assert message == (
        "inner[main] sits inside outer[main] (DeferToResample), which can call the model "
        "more than once in a step. LangChain keeps the commands of the last call only, so "
        "the samples inner[main] judged in earlier calls never reach monitor_log, and calls "
        "drawn at once under ainvoke() leave one record each under the same step number, so "
        "later step numbers are skipped. As the outer monitor, use one that calls the model "
        "at most once per step: DeferToResample with max_resamples=0 and no trusted_model. "
        "Or judge with both monitors in one MonitorMiddleware, through MonitorTwins of your "
        "own whose twins run both and keep the higher suspicion."
    )


@pytest.mark.parametrize("build_alone", REPEATING_MONITORS.values(), ids=REPEATING_MONITORS)
def test_a_lone_monitor_that_can_call_the_model_again_is_not_named(
    build_alone: MonitorBuilder,
) -> None:
    # Act
    named = check_without_warnings([build_alone("alone")])

    # Assert
    assert named == []


def test_every_monitor_inside_one_that_can_call_again_is_named() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(max_resamples=2), label="outer"),
        build_monitor(build_trusted_monitoring(), label="middle"),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="more than once"):
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["middle[main]", "inner[main]"]


def test_only_the_monitors_inside_the_one_that_can_call_again_are_named() -> None:
    # Arrange
    stack = [
        build_monitor(build_trusted_monitoring(), label="outer"),
        build_monitor(build_resampling(max_resamples=2), label="middle"),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="more than once"):
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]"]


def test_a_monitor_inside_two_that_can_call_again_is_named_once_with_both() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(max_resamples=2), label="outer"),
        build_monitor(build_resampling(max_resamples=1), label="middle"),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="more than once") as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["middle[main]", "inner[main]"]
    [inner_warning] = [
        str(warning.message) for warning in caught if str(warning.message).startswith("inner")
    ]
    assert "outer[main] (DeferToResample), middle[main] (DeferToResample)" in inner_warning


@pytest.mark.parametrize(
    ("build_outer", "expected"),
    [
        (lambda: build_resampling(max_resamples=2), ["CommandingMiddleware"]),
        (build_trusted_monitoring, []),
    ],
    ids=["can-call-again", "calls-once"],
)
def test_a_middleware_between_two_monitors_is_named_inside_one_that_can_call_again(
    build_outer: Callable[[], DeferToResample],
    expected: list[str],
) -> None:
    # Arrange
    stack = [
        build_monitor(build_outer(), label="outer"),
        UnsupportedContentMiddleware(),
        CommandingMiddleware(),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert [name for name in named if name != "inner[main]"] == expected
    assert all(issubclass(warning.category, MonitorPlacementWarning) for warning in caught)
    assert len(caught) == len(named)


def test_the_warning_for_a_middleware_between_monitors_reads_in_full() -> None:
    # Arrange: the middleware sits inside two monitors that can each call the model again
    stack = [
        build_monitor(build_resampling(max_resamples=2), label="outer"),
        build_monitor(BlockAndFeedBack(block_threshold=THRESHOLD), label="middle"),
        CommandingMiddleware(),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert: the middle monitor, which may block inside the outer one, is named for that too
    assert named == ["middle[main]", "CommandingMiddleware", "inner[main]", "middle[main]"]
    [message] = [
        str(warning.message)
        for warning in caught
        if str(warning.message).startswith("CommandingMiddleware")
    ]
    assert message == (
        "CommandingMiddleware wraps model calls inside outer[main] (DeferToResample), "
        "middle[main] (BlockAndFeedBack), which can call the model more than once in a step, "
        "so a state update it returns may come from a sample the protocol does not commit. "
        "List it before outer[main]."
    )


def test_the_repeated_call_warnings_come_before_the_blocked_reply_warnings() -> None:
    # Arrange: the middle monitor may block inside the outer one, and redraws the inner one
    stack = [
        build_monitor(build_trusted_monitoring(), label="outer"),
        build_monitor(BlockAndFeedBack(block_threshold=THRESHOLD), label="middle"),
        build_monitor(build_trusted_monitoring(), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]", "middle[main]"]
    messages = [str(warning.message) for warning in caught]
    assert messages[0].startswith("inner[main] sits inside middle[main] (BlockAndFeedBack)")
    assert messages[1].startswith("middle[main] may commit blocked replies")


@pytest.mark.parametrize(
    "build_inner",
    [
        lambda label: build_monitor(build_trusted_monitoring(), label=label),
        lambda label: build_monitor(
            DeferToResample(
                defer_threshold=THRESHOLD,
                audit_threshold=DEFER_BELOW_AUDIT,
                max_resamples=2,
            ),
            label=label,
            trusted_model=build_trusted_model(),
        ),
        lambda label: build_monitor(build_resampling(max_resamples=0), label=label),
        lambda label: build_monitor(
            build_deferral(),
            label=label,
            trusted_model=build_trusted_model(),
        ),
    ],
    ids=["trusted-monitoring", "resample-then-defer", "audit-at-the-defer-threshold", "defer"],
)
def test_an_inner_monitor_that_never_blocks_is_not_named(build_inner: MonitorBuilder) -> None:
    # Arrange
    stack = [build_monitor(build_trusted_monitoring(), label="outer"), build_inner("inner")]

    # Act
    named = check_without_warnings(stack)

    # Assert
    assert named == []


@pytest.mark.parametrize(
    "inner_protocol",
    [
        BlockAndFeedBack(block_threshold=THRESHOLD),
        BlockAndFeedBack(block_threshold=THRESHOLD, blocks_in_a_row=1),
        DeferToResample(
            defer_threshold=THRESHOLD,
            audit_threshold=THRESHOLD,
            max_resamples=0,
            on_unavailable=FailureResponse.BLOCK,
        ),
    ],
    ids=["block-and-feed-back", "block-and-feed-back-one-in-a-row", "block-when-no-answer"],
)
def test_an_inner_monitor_that_may_block_is_named_for_its_blocked_replies(
    inner_protocol: ProtocolConfiguration,
) -> None:
    # Arrange
    inner = build_monitor(inner_protocol, label="inner")
    stack = [build_monitor(build_trusted_monitoring(), label="outer"), inner]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="blocked replies") as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]"]
    assert "Put the monitor that blocks outermost" in str(caught[0].message)


def test_an_inner_monitor_with_both_faults_is_named_for_each() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(max_resamples=2), label="outer"),
        build_monitor(BlockAndFeedBack(block_threshold=THRESHOLD), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]", "inner[main]"]
    messages = [str(warning.message) for warning in caught]
    assert "more than once in a step" in messages[0]
    assert "blocked replies" in messages[1]


def test_a_stacked_monitor_warning_points_at_the_code_that_checks_the_list() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(max_resamples=2), label="outer"),
        build_monitor(BlockAndFeedBack(block_threshold=THRESHOLD), label="inner"),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        check_monitor_placement(middleware=stack)

    # Assert
    assert [warning.filename for warning in caught] == [__file__, __file__]
