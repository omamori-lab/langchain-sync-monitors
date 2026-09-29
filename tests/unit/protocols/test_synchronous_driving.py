"""Every protocol finishes with one `send(None)`, the way the middleware runs it under `invoke()`.

Check S2 (issue #4) found that a protocol written once as a coroutine works
under `invoke()` only if it awaits nothing but the pending step. These tests
drive each protocol down its longest path without an event loop.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    Outcome,
    PendingStep,
    Resampling,
    StepDecision,
)
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    TrustedMonitoring,
)

from .scripted_step import ScriptedPendingStep, run_without_event_loop


class EventLoopProtocol(ControlProtocol):
    """A protocol that wrongly awaits real asynchronous work before deciding."""

    def __init__(self, *, inner: ControlProtocol) -> None:
        """Wrap a protocol whose decision this one returns after sleeping."""
        self.inner = inner

    async def decide(self, step: PendingStep) -> StepDecision:
        """Yield to the event loop, then let the inner protocol decide."""
        await asyncio.sleep(0)
        return await self.inner.decide(step)


@pytest.fixture
def protocols_on_their_longest_paths(
    defer_to_trusted_model: DeferToTrustedModel,
    trusted_model: FakeListChatModel,
) -> dict[str, tuple[ControlProtocol, list[float], Outcome]]:
    return {
        "trusted monitoring": (TrustedMonitoring(flag_threshold=0.6), [0.9], Outcome.ALLOWED),
        "sequential resampling": (
            DeferToResample(
                fallback=defer_to_trusted_model, defer_threshold=0.6, audit_threshold=0.9
            ),
            [0.9, 0.8, 0.7],
            Outcome.DEFERRED_TO_TRUSTED,
        ),
        "parallel resampling": (
            DeferToResample(
                fallback=defer_to_trusted_model,
                defer_threshold=0.6,
                audit_threshold=0.9,
                resampling=Resampling.PARALLEL,
            ),
            [0.9, 0.8, 0.7],
            Outcome.DEFERRED_TO_TRUSTED,
        ),
        "defer to trusted": (
            DeferToTrusted(trusted_model=trusted_model, defer_threshold=0.6, audit_threshold=0.9),
            [0.9],
            Outcome.DEFERRED_TO_TRUSTED,
        ),
        "auto mode steered": (AutoMode(block_threshold=0.6), [0.9, 0.8, 0.1], Outcome.STEERED),
        "auto mode halted": (AutoMode(block_threshold=0.6), [0.9, 0.8, 0.7], Outcome.HALTED),
    }


@pytest.mark.parametrize(
    "path_name",
    [
        "trusted monitoring",
        "sequential resampling",
        "parallel resampling",
        "defer to trusted",
        "auto mode steered",
        "auto mode halted",
    ],
)
def test_protocol_finishes_with_one_send(
    protocols_on_their_longest_paths: dict[str, tuple[ControlProtocol, list[float], Outcome]],
    path_name: str,
) -> None:
    # Arrange
    protocol, suspicions, expected_outcome = protocols_on_their_longest_paths[path_name]
    step = ScriptedPendingStep(suspicions=suspicions)

    # Act
    decision = run_without_event_loop(protocol.decide(step))

    # Assert
    assert decision.outcome is expected_outcome
    assert len(step.drawn_samples) == len(suspicions)


def test_protocol_that_awaits_the_event_loop_is_caught() -> None:
    # Arrange
    protocol = EventLoopProtocol(inner=TrustedMonitoring(flag_threshold=0.6))
    step = ScriptedPendingStep(suspicions=[0.1])

    # Act / Assert
    with pytest.raises(AssertionError, match="suspended"):
        run_without_event_loop(protocol.decide(step))
