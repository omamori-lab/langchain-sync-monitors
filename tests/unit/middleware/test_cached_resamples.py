"""A LangChain response cache turns resamples into copies, so resampling under one warns.

LangChain answers an identical request from its cache, and a resample repeats
the first sample's request exactly, so every resample is the first sample
again. Auto Mode's retries carry the feedback, so their requests differ and
do not warn.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors.contracts import ControlProtocol
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.model_calls import CachedResampleWarning, is_response_cache_active
from langchain_sync_monitors.pending_steps import warn_about_cached_resamples
from langchain_sync_monitors.protocols import AutoMode, DeferToResample, HaltRun
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel


@pytest.fixture(autouse=True)
def fresh_warning() -> Iterator[None]:
    """Let each test see the warning, which the library shows once per process."""
    warn_about_cached_resamples.cache_clear()
    yield
    warn_about_cached_resamples.cache_clear()


@pytest.fixture
def global_cache() -> Iterator[InMemoryCache]:
    earlier_cache = get_llm_cache()
    cache = InMemoryCache()
    set_llm_cache(cache)
    yield cache
    set_llm_cache(earlier_cache)


def build_defer_to_resample() -> DeferToResample:
    return DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=0.9)


def build_agent(model: ScriptedChatModel, *, protocol: ControlProtocol) -> Runnable[Any, Any]:
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    return create_agent(model=model, tools=Workspace().build_tools(), middleware=[middleware])


def build_model(**settings: Any) -> ScriptedChatModel:
    replies = [build_exfiltration_step(), build_read_step(), build_read_step(), AIMessage("Done.")]
    return ScriptedChatModel(responses=replies, **settings)


def run_and_collect_warnings(agent: Runnable[Any, Any], *, mode: RunMode) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=mode)
    return [str(warning.message) for warning in caught if warning.category is CachedResampleWarning]


@pytest.mark.usefixtures("global_cache")
def test_resampling_under_the_global_cache_warns(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(build_model(), protocol=build_defer_to_resample())

    # Act
    caught = run_and_collect_warnings(agent, mode=run_mode)

    # Assert
    assert caught
    assert "cache=False" in caught[0]


def test_resampling_under_a_model_cache_warns(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(build_model(cache=InMemoryCache()), protocol=build_defer_to_resample())

    # Act
    caught = run_and_collect_warnings(agent, mode=run_mode)

    # Assert
    assert caught


@pytest.mark.usefixtures("global_cache")
def test_a_model_that_turns_the_cache_off_resamples_without_a_warning(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(build_model(cache=False), protocol=build_defer_to_resample())

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error", CachedResampleWarning)
        result = run_agent(agent, mode=run_mode)

    # Assert
    assert result["monitor_log"][0]["outcome"] == "resampled"


@pytest.mark.usefixtures("global_cache")
def test_auto_mode_retries_differ_so_they_do_not_warn(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(build_model(), protocol=AutoMode(block_threshold=0.6))

    # Act
    caught = run_and_collect_warnings(agent, mode=run_mode)

    # Assert
    assert caught == []


@pytest.mark.usefixtures("global_cache")
def test_a_step_that_is_never_resampled_does_not_warn(run_mode: RunMode) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[build_read_step(), AIMessage("Done.")])
    agent = build_agent(model, protocol=build_defer_to_resample())

    # Act
    caught = run_and_collect_warnings(agent, mode=run_mode)

    # Assert
    assert caught == []


@pytest.mark.usefixtures("global_cache")
def test_the_warning_shows_once_even_when_every_warning_is_shown(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(build_model(), protocol=build_defer_to_resample())

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_agent(agent, mode=run_mode)
        run_agent(agent, mode=run_mode)

    # Assert
    assert [warning.category for warning in caught] == [CachedResampleWarning]


class RunPickedModel(AgentMiddleware[Any, Any, Any]):
    """Answers each model call through a scripted model, as the provider a run configures."""

    def __init__(self, model: ScriptedChatModel) -> None:
        super().__init__()
        self.scripted = model

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(request.override(model=self.scripted))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(request.override(model=self.scripted))


CONFIGURABLE_MODELS = {
    "every-field-configurable": lambda: init_chat_model(configurable_fields="any"),
    "no-default-model": lambda: init_chat_model(temperature=0),
}
"""Models picked per run, whose `cache` raises when read: `AttributeError` and `TypeError`."""


@pytest.mark.parametrize(
    "build_configurable", CONFIGURABLE_MODELS.values(), ids=CONFIGURABLE_MODELS.keys()
)
def test_resampling_a_model_picked_per_run_neither_fails_nor_warns(
    run_mode: RunMode,
    build_configurable: Callable[[], Any],
) -> None:
    # Arrange: the monitor's request holds the configurable model; a later middleware answers
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        MonitorMiddleware(monitor=build_keyword_monitor(), protocol=build_defer_to_resample()),
        RunPickedModel(build_model()),
    ]
    agent = create_agent(
        model=cast("BaseChatModel", build_configurable()),
        tools=Workspace().build_tools(),
        middleware=middleware,
    )

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error", CachedResampleWarning)
        result = run_agent(agent, mode=run_mode)

    # Assert
    outcomes = [record["outcome"] for record in result["monitor_log"]]
    assert outcomes == ["resampled", "allowed", "allowed"]


@pytest.mark.usefixtures("global_cache")
@pytest.mark.parametrize(
    "wrap",
    [lambda model: model.bind(stop=["."]), lambda model: model.with_fallbacks([model])],
    ids=["bind", "with_fallbacks"],
)
def test_a_wrapped_model_is_read_through_its_wrapper(wrap: Callable[[Any], Any]) -> None:
    # Arrange
    wrapped = wrap(build_model())

    # Act
    is_active = is_response_cache_active(wrapped)

    # Assert
    assert is_active
