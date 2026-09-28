"""Fixtures shared by the protocol tests."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors.protocols.fallbacks import DeferToTrustedModel

from .scripted_step import StepDriver, run_on_event_loop, run_without_event_loop


@pytest.fixture(params=[run_on_event_loop, run_without_event_loop], ids=["ainvoke", "invoke"])
def drive(request: pytest.FixtureRequest) -> StepDriver:
    """Run each test on an event loop and again without one, as the middleware does."""
    driver: StepDriver = request.param
    return driver


@pytest.fixture
def trusted_model() -> FakeListChatModel:
    """A stand-in for the trusted model; the scripted step never calls it."""
    return FakeListChatModel(responses=["unused"])


@pytest.fixture
def defer_to_trusted_model(trusted_model: FakeListChatModel) -> DeferToTrustedModel:
    """The fallback that hands the step to the trusted model."""
    return DeferToTrustedModel(trusted_model=trusted_model)
