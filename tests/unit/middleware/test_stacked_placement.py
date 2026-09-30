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
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import ControlProtocol, FeedbackVisibility
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    HaltRun,
    TrustedMonitoring,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst
from tests.unit.middleware.test_placement import (
    CommandingMiddleware,
    UnsupportedContentMiddleware,
)

THRESHOLD = 0.5

type ProtocolBuilder = Callable[[], ControlProtocol]


class LoudHaltRun(HaltRun):
    """A project's own halt, which could call the model before it halts."""


class TeamTrustedMonitoring(TrustedMonitoring):
    """A project's own Trusted Monitoring, which could draw more than once or block."""


class TeamDeferToResample(DeferToResample):
    """A project's own Defer to Resample, which could draw again or block."""


class TeamAutoMode(AutoMode):
    """A project's own Auto Mode, which could draw again after the first block."""


def build_trusted_fallback() -> DeferToTrustedModel:
    return DeferToTrustedModel(trusted_model=ScriptedChatModel(responses=[AIMessage("safe")]))


def build_trusted_monitoring() -> ControlProtocol:
    return TrustedMonitoring(flag_threshold=THRESHOLD)


def build_resampling(
    *, fallback: HaltRun | DeferToTrustedModel, max_resamples: int
) -> ControlProtocol:
    return DeferToResample(
        fallback=fallback,
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
        max_resamples=max_resamples,
    )


def build_auto_mode(
    *, max_consecutive_blocks: int, when_limit_reached: HaltRun | DeferToTrustedModel
) -> ControlProtocol:
    return AutoMode(
        block_threshold=THRESHOLD,
        max_consecutive_blocks=max_consecutive_blocks,
        when_limit_reached=when_limit_reached,
    )


SINGLE_CALL_PROTOCOLS: dict[str, ProtocolBuilder] = {
    "trusted-monitoring": build_trusted_monitoring,
    "resample-none-then-halt": lambda: build_resampling(fallback=HaltRun(), max_resamples=0),
    "auto-mode-one-block-then-halt": lambda: build_auto_mode(
        max_consecutive_blocks=1, when_limit_reached=HaltRun()
    ),
}
"""The settings that call the rest of the stack at most once per step."""

REPEATING_PROTOCOLS: dict[str, ProtocolBuilder] = {
    "defer-to-trusted": lambda: DeferToTrusted(
        trusted_model=ScriptedChatModel(responses=[]),
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
    ),
    "resample-once-then-halt": lambda: build_resampling(fallback=HaltRun(), max_resamples=1),
    "resample-none-then-defer": lambda: build_resampling(
        fallback=build_trusted_fallback(), max_resamples=0
    ),
    "resample-none-then-halt-subclass": lambda: build_resampling(
        fallback=LoudHaltRun(), max_resamples=0
    ),
    "auto-mode-two-blocks-then-halt": lambda: build_auto_mode(
        max_consecutive_blocks=2, when_limit_reached=HaltRun()
    ),
    "auto-mode-one-block-then-defer": lambda: build_auto_mode(
        max_consecutive_blocks=1, when_limit_reached=build_trusted_fallback()
    ),
    "trusted-monitoring-subclass": lambda: TeamTrustedMonitoring(flag_threshold=THRESHOLD),
    "resample-subclass-none-then-halt": lambda: TeamDeferToResample(
        fallback=HaltRun(),
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
        max_resamples=0,
    ),
    "auto-mode-subclass-one-block-then-halt": lambda: TeamAutoMode(
        block_threshold=THRESHOLD, max_consecutive_blocks=1, when_limit_reached=HaltRun()
    ),
    "own-protocol": AcceptFirst,
}
"""The settings that may call the rest of the stack again, each next to a sound one."""


def build_monitor(
    protocol: ControlProtocol,
    *,
    label: str,
    feedback_visibility: FeedbackVisibility = FeedbackVisibility.HIDDEN,
) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=protocol,
        label=label,
        feedback_visibility=feedback_visibility,
    )


def check_without_warnings(stack: Sequence[AgentMiddleware[Any, Any, Any]]) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return check_monitor_placement(middleware=stack)


@pytest.mark.parametrize("build_outer", SINGLE_CALL_PROTOCOLS.values(), ids=SINGLE_CALL_PROTOCOLS)
def test_a_monitor_inside_one_that_calls_the_model_once_is_not_named(
    build_outer: ProtocolBuilder,
) -> None:
    # Arrange
    inner_protocol = build_resampling(fallback=build_trusted_fallback(), max_resamples=2)
    stack = [
        build_monitor(build_outer(), label="outer"),
        build_monitor(inner_protocol, label="inner"),
    ]

    # Act
    named = check_without_warnings(stack)

    # Assert
    assert named == []


@pytest.mark.parametrize("build_outer", REPEATING_PROTOCOLS.values(), ids=REPEATING_PROTOCOLS)
def test_a_monitor_inside_one_that_can_call_the_model_again_is_named(
    build_outer: ProtocolBuilder,
) -> None:
    # Arrange
    outer = build_monitor(build_outer(), label="outer")
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


@pytest.mark.parametrize("build_protocol", REPEATING_PROTOCOLS.values(), ids=REPEATING_PROTOCOLS)
def test_a_lone_monitor_that_can_call_the_model_again_is_not_named(
    build_protocol: ProtocolBuilder,
) -> None:
    # Act
    named = check_without_warnings([build_monitor(build_protocol(), label="alone")])

    # Assert
    assert named == []


def test_every_monitor_inside_one_that_can_call_again_is_named() -> None:
    # Arrange
    resampling = build_resampling(fallback=HaltRun(), max_resamples=2)
    stack = [
        build_monitor(resampling, label="outer"),
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
    resampling = build_resampling(fallback=HaltRun(), max_resamples=2)
    stack = [
        build_monitor(build_trusted_monitoring(), label="outer"),
        build_monitor(resampling, label="middle"),
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
        build_monitor(build_resampling(fallback=HaltRun(), max_resamples=2), label="outer"),
        build_monitor(build_resampling(fallback=HaltRun(), max_resamples=1), label="middle"),
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
        (lambda: build_resampling(fallback=HaltRun(), max_resamples=2), ["CommandingMiddleware"]),
        (build_trusted_monitoring, []),
    ],
    ids=["can-call-again", "calls-once"],
)
def test_a_middleware_between_two_monitors_is_named_inside_one_that_can_call_again(
    build_outer: ProtocolBuilder,
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


@pytest.mark.parametrize(
    "build_inner",
    [
        build_trusted_monitoring,
        lambda: build_resampling(fallback=build_trusted_fallback(), max_resamples=2),
        lambda: build_resampling(fallback=HaltRun(), max_resamples=0),
        lambda: DeferToTrusted(
            trusted_model=ScriptedChatModel(responses=[]),
            defer_threshold=THRESHOLD,
            audit_threshold=THRESHOLD,
        ),
    ],
    ids=["trusted-monitoring", "resample-then-defer", "resample-none-then-halt", "defer"],
)
def test_an_inner_monitor_that_never_blocks_may_keep_feedback_in_the_transcript(
    build_inner: ProtocolBuilder,
) -> None:
    # Arrange
    inner = build_monitor(
        build_inner(), label="inner", feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT
    )
    stack = [build_monitor(build_trusted_monitoring(), label="outer"), inner]

    # Act
    named = check_without_warnings(stack)

    # Assert
    assert named == []


@pytest.mark.parametrize(
    "build_inner",
    [
        lambda: build_auto_mode(max_consecutive_blocks=3, when_limit_reached=HaltRun()),
        lambda: build_auto_mode(max_consecutive_blocks=1, when_limit_reached=HaltRun()),
        lambda: build_resampling(fallback=LoudHaltRun(), max_resamples=2),
        lambda: TeamTrustedMonitoring(flag_threshold=THRESHOLD),
        lambda: TeamDeferToResample(
            fallback=HaltRun(), defer_threshold=THRESHOLD, audit_threshold=THRESHOLD
        ),
        AcceptFirst,
    ],
    ids=[
        "auto-mode",
        "auto-mode-one-block",
        "resample-then-own-fallback",
        "trusted-monitoring-subclass",
        "resample-subclass",
        "own-protocol",
    ],
)
def test_an_inner_monitor_that_may_block_is_named_for_feedback_in_the_transcript(
    build_inner: ProtocolBuilder,
) -> None:
    # Arrange
    inner = build_monitor(
        build_inner(), label="inner", feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT
    )
    stack = [build_monitor(build_trusted_monitoring(), label="outer"), inner]

    # Act
    with pytest.warns(MonitorPlacementWarning, match="first blocked proposal") as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]"]
    assert "FeedbackVisibility.HIDDEN" in str(caught[0].message)


@pytest.mark.parametrize(
    "outer_visibility",
    [FeedbackVisibility.HIDDEN, FeedbackVisibility.IN_TRANSCRIPT],
    ids=["outer-hidden", "outer-in-transcript"],
)
def test_blocked_attempts_are_safe_when_hidden_inside_or_shown_by_the_outermost_monitor(
    outer_visibility: FeedbackVisibility,
) -> None:
    # Arrange: only a monitor inside another hands its blocked attempts to a judge
    auto_mode = build_auto_mode(max_consecutive_blocks=1, when_limit_reached=HaltRun())
    stack = [
        build_monitor(auto_mode, label="outer", feedback_visibility=outer_visibility),
        build_monitor(
            build_auto_mode(max_consecutive_blocks=3, when_limit_reached=HaltRun()),
            label="inner",
        ),
    ]

    # Act
    named = check_without_warnings(stack)

    # Assert
    assert named == []


def test_an_inner_monitor_with_both_faults_is_named_for_each() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(fallback=HaltRun(), max_resamples=2), label="outer"),
        build_monitor(
            build_auto_mode(max_consecutive_blocks=3, when_limit_reached=HaltRun()),
            label="inner",
            feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT,
        ),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        named = check_monitor_placement(middleware=stack)

    # Assert
    assert named == ["inner[main]", "inner[main]"]
    messages = [str(warning.message) for warning in caught]
    assert any("more than once in a step" in message for message in messages)
    assert any("first blocked proposal" in message for message in messages)


def test_a_stacked_monitor_warning_points_at_the_code_that_checks_the_list() -> None:
    # Arrange
    stack = [
        build_monitor(build_resampling(fallback=HaltRun(), max_resamples=2), label="outer"),
        build_monitor(
            build_auto_mode(max_consecutive_blocks=3, when_limit_reached=HaltRun()),
            label="inner",
            feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT,
        ),
    ]

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        check_monitor_placement(middleware=stack)

    # Assert
    assert [warning.filename for warning in caught] == [__file__, __file__]
