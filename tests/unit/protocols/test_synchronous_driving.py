"""The protocol tests fail a protocol that awaits anything but the pending step.

Check S2 (issue #4) found that a protocol written once as a coroutine works
under `invoke()` only if it awaits nothing but the pending step. Every protocol
test runs through the `drive` fixture, once without an event loop, as the
middleware runs it under `invoke()`. This test checks that this driver fails a
protocol that suspends.
"""

from __future__ import annotations

import asyncio

import pytest

from langchain_sync_monitors.contracts import ControlProtocol, PendingStep, StepDecision
from langchain_sync_monitors.protocols import TrustedMonitoring

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


def test_protocol_that_awaits_the_event_loop_is_caught() -> None:
    # Arrange
    protocol = EventLoopProtocol(inner=TrustedMonitoring(audit_threshold=0.6))
    step = ScriptedPendingStep(suspicions=[0.1])

    # Act / Assert
    with pytest.raises(AssertionError, match="suspended"):
        run_without_event_loop(protocol.decide(step))
