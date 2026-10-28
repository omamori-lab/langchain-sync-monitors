"""Monitors: each one configuration that builds hyoshigi's sync and async twins.

A monitor rates an action the agent proposes against its trajectory
[@hyoshigi2026]. hyoshigi gives every monitor as two classes, one that calls
its model synchronously and one that awaits it, and each configuration here
builds both from the same options, as `twins`, so one middleware serves
`invoke()` and `ainvoke()`. `LLMMonitor` and `GuardMonitor` call a LangChain
chat model through `ChatModelAdapter`, `DecisionMonitor` calls a decision
model's SDK, `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor`
wrap the twins of other monitors, and `MonitorTwins` holds a pair the user
wrote. Each option keeps hyoshigi's name and meaning. An option hyoshigi
refuses raises the library's `ConfigurationError` where the monitor is
built, and a warning hyoshigi raises then is raised once, at that line.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Final, TypedDict, assert_never

import hyoshigi
from hyoshigi import (
    DEFAULT_PROMPT,
    DEFAULT_STATEMENT,
    Aggregation,
    GuardLabels,
    MediaTypes,
    MonitorPrompt,
    View,
)
from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors.chat_model_adapter import ChatModelAdapter
from langchain_sync_monitors.errors import (
    ConfigurationError,
    MissingExtraError,
    build_missing_extra_message,
    raise_as_configuration_error,
)
from langchain_sync_monitors.model_calls import CachedResampleWarning, is_response_cache_active
from langchain_sync_monitors.openrouter_decisions import open_decision_models
from langchain_sync_monitors.options import check_instance_option, describe_option_value
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY

if TYPE_CHECKING:
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

LLM_CALL_SOURCE: Final = "llm_monitor"
"""The `lc_source` of an LLM monitor's model calls."""

GUARD_CALL_SOURCE: Final = "guard_model_monitor"
"""The `lc_source` of a guard monitor's model calls."""

CONSTRUCTOR_CALLER_LEVEL: Final = 3
"""The stack level, past the library's frames, of the line that built a library dataclass.

Level 2 is the dataclass's generated `__init__`, whose file is `<string>`.
"""


def build_twins_warning_once(build: Callable[[], MonitorTwins]) -> MonitorTwins:
    """Build twins, and raise each distinct warning the build raised once, at the caller.

    hyoshigi raises a warning once for each twin, from inside the library;
    each is raised again once, at the line that built the monitor.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        twins = build()
    for message, category in dict.fromkeys((str(item.message), item.category) for item in caught):
        warnings.warn(
            message,
            category,
            stacklevel=CONSTRUCTOR_CALLER_LEVEL,
            skip_file_prefixes=(LIBRARY_DIRECTORY,),
        )
    return twins


def set_twins(
    monitor: MonitorConfiguration,
    *,
    twins: MonitorTwins,
) -> None:
    """Keep the twins a configuration built, past the frozen dataclass's guard."""
    object.__setattr__(monitor, "twins", twins)


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorTwins:
    """A monitor written as hyoshigi's twins: one for `invoke()`, one for `ainvoke()`.

    `monitor` must be a `hyoshigi.Monitor` and `async_monitor` a
    `hyoshigi.AsyncMonitor`; anything else raises `ConfigurationError`. The
    two should rate alike, since the entry point alone picks which one runs.
    """

    monitor: hyoshigi.Monitor
    async_monitor: hyoshigi.AsyncMonitor

    def __post_init__(self) -> None:
        """Reject a twin of the wrong family."""
        check_instance_option(
            self.monitor,
            option_type=hyoshigi.Monitor,
            parameter_name="monitor",
            hint="It is the twin that rates under invoke(), whose rate is a plain method.",
        )
        check_instance_option(
            self.async_monitor,
            option_type=hyoshigi.AsyncMonitor,
            parameter_name="async_monitor",
            hint="It is the twin that rates under ainvoke(), whose rate is a coroutine.",
        )

    @property
    def twins(self) -> MonitorTwins:
        """The twins themselves, as every monitor configuration gives them."""
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMMonitor:
    """A generative model that reasons about an action, then scores it on an integer scale.

    `model` is a chat model or a provider string, which `init_chat_model`
    resolves. The other options are hyoshigi's `LLMMonitor`'s: what the
    monitor reads (`view`), its instructions (`prompt`), the ends of its
    scale, and how many more times it asks after an unreadable reply. Its
    calls carry the `lc_source` `llm_monitor`.
    """

    model: str | BaseChatModel
    view: View = field(default_factory=View)
    prompt: MonitorPrompt = DEFAULT_PROMPT
    lowest_score: int = 0
    highest_score: int = 10
    max_parse_retries: int = 2
    chat_model: BaseChatModel = field(init=False, repr=False, compare=False)
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build both twins on one chat model."""
        adapter = ChatModelAdapter(self.model, source=LLM_CALL_SOURCE)
        object.__setattr__(self, "chat_model", adapter.chat_model)
        options = {
            "view": self.view,
            "prompt": self.prompt,
            "lowest_score": self.lowest_score,
            "highest_score": self.highest_score,
            "max_parse_retries": self.max_parse_retries,
        }
        with raise_as_configuration_error():
            twins = MonitorTwins(
                monitor=hyoshigi.LLMMonitor(model=adapter.generative_model, **options),
                async_monitor=hyoshigi.AsyncLLMMonitor(
                    model=adapter.async_generative_model,
                    **options,
                ),
            )
        set_twins(self, twins=twins)


class GuardOptions(TypedDict):
    """The options a guard monitor gives both of hyoshigi's twins."""

    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int
    view: View


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardMonitor:
    """A guard model that labels an action against a policy, scored by its labels.

    `model` is a chat model or a provider string. The other options are
    hyoshigi's `GuardMonitor`'s: `prompt` states the policy, `labels` names
    the guard's labels, and the score is the share of `samples` replies
    labelled suspicious. One sample warns `HardLabelWarning` once, at the
    line that builds the monitor. `top_log_probabilities` is the number of
    alternatives to each token that the chat model is asked for, a whole
    number of at least 0. Above 0 the monitor asks once, and hyoshigi's guard
    scores the label from the alternatives [@hyoshigi2026choosing]. A chat
    model that rejects the request raises `ConfigurationError` when it is
    called. Its calls carry the `lc_source` `guard_model_monitor`.
    """

    model: str | BaseChatModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    top_log_probabilities: int = 0
    view: View = field(default_factory=View)
    chat_model: BaseChatModel = field(init=False, repr=False, compare=False)
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build both twins on one chat model."""
        adapter = ChatModelAdapter(
            self.model,
            source=GUARD_CALL_SOURCE,
            top_log_probabilities=self.top_log_probabilities,
        )
        object.__setattr__(self, "chat_model", adapter.chat_model)
        options = GuardOptions(
            prompt=self.prompt,
            labels=self.labels,
            samples=self.samples,
            view=self.view,
        )
        with raise_as_configuration_error():
            twins = build_twins_warning_once(
                lambda: MonitorTwins(
                    monitor=hyoshigi.GuardMonitor(model=adapter.generative_model, **options),
                    async_monitor=hyoshigi.AsyncGuardMonitor(
                        model=adapter.async_generative_model,
                        **options,
                    ),
                ),
            )
        set_twins(self, twins=twins)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionMonitor:
    """A decision model asked whether statements about the trajectory hold, scored by aggregation.

    `model` names a decision model on TypeSafe's API as `typesafe:<model id>`,
    such as `typesafe:jev-1.14`, whose clients read `TYPESAFE_API_KEY` and
    need the `typesafe` extra, or on OpenRouter's Decisions API as
    `openrouter:<model id>`, such as `openrouter:typesafe/jev-1.14`, which
    reads `OPENROUTER_API_KEY` and takes text only, as
    `openrouter_decisions` explains. The other options are hyoshigi's
    `DecisionMonitor`'s, and `perceives` the media the model takes. A model
    of another provider, a `perceives` that is not a `MediaTypes`, or a missing
    key raises `ConfigurationError`; any other decision model is a
    `MonitorTwins` of hyoshigi's `DecisionMonitor`.
    """

    model: str
    view: View = field(default_factory=View)
    statements: tuple[str, ...] = (DEFAULT_STATEMENT,)
    aggregation: Aggregation = Aggregation.MAX
    perceives: MediaTypes = field(default_factory=MediaTypes)
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build both twins on the model's decision models."""
        check_instance_option(self.model, option_type=str, parameter_name="model")
        check_instance_option(self.perceives, option_type=MediaTypes, parameter_name="perceives")
        model, async_model = build_decision_models(self.model, perceives=self.perceives)
        options = {
            "view": self.view,
            "statements": self.statements,
            "aggregation": self.aggregation,
        }
        with raise_as_configuration_error():
            twins = MonitorTwins(
                monitor=hyoshigi.DecisionMonitor(model=model, **options),
                async_monitor=hyoshigi.AsyncDecisionMonitor(model=async_model, **options),
            )
        set_twins(self, twins=twins)


def build_decision_models(
    model: str,
    *,
    perceives: MediaTypes,
) -> tuple[hyoshigi.DecisionModel, hyoshigi.AsyncDecisionModel]:
    """Return the sync and async decision models a `provider:<model id>` string names.

    Anything but `typesafe:` or `openrouter:` and a model id raises
    `ConfigurationError`, and so does an `openrouter:` model that perceives
    media, since its client sends text only.
    """
    provider, _, model_id = model.partition(":")
    match provider:
        case "typesafe" if model_id:
            sync_client, async_client = build_typesafe_clients()
            with raise_as_configuration_error():
                return (
                    hyoshigi.DecisionModel.from_typesafe(
                        sync_client,
                        model=model_id,
                        perceives=perceives,
                    ),
                    hyoshigi.AsyncDecisionModel.from_typesafe(
                        async_client,
                        model=model_id,
                        perceives=perceives,
                    ),
                )
        case "openrouter" if model_id:
            if perceives.patterns:
                message = "an 'openrouter:' decision model reads text only, so perceives no media"
                raise ConfigurationError(message)
            return open_decision_models(model_id)
        case _:
            message = (
                "model must name a decision model as 'typesafe:<model id>' or "
                f"'openrouter:<model id>', got {model!r}"
            )
            raise ConfigurationError(message)


def build_typesafe_clients() -> tuple[TypeSafeClient, AsyncTypeSafeClient]:
    """Return TypeSafe's sync and async clients, built from the environment.

    Without the `typesafe` extra this raises `MissingExtraError`, and a client
    TypeSafe's SDK refuses to build, such as one without a key, raises
    `ConfigurationError` with the SDK's message.
    """
    try:
        # Imported here because the SDK is the optional extra `typesafe`.
        from typesafe_sdk import (  # lanorme: ignore[IMPORT-001]
            AsyncTypeSafeClient,
            TypeSafeClient,
            TypeSafeError,
        )
    except ImportError as error:
        message = build_missing_extra_message("A 'typesafe:' decision model", extra="typesafe")
        raise MissingExtraError(message) from error
    try:
        return TypeSafeClient(), AsyncTypeSafeClient()
    except TypeSafeError as error:
        raise ConfigurationError(str(error)) from error


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMonitor:
    """A monitor asked `ensemble_size` times for the same action, reported as the mean.

    The twins are hyoshigi's `RepeatedMonitor` over `monitor`'s twins, with
    its rules for calls that failed closed.
    """

    monitor: MonitorConfiguration
    ensemble_size: int
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build both twins over the wrapped monitor's."""
        check_monitor_option(self.monitor, parameter_name="monitor")
        inner = self.monitor.twins
        with raise_as_configuration_error():
            twins = MonitorTwins(
                monitor=hyoshigi.RepeatedMonitor(
                    monitor=inner.monitor,
                    ensemble_size=self.ensemble_size,
                ),
                async_monitor=hyoshigi.AsyncRepeatedMonitor(
                    monitor=inner.async_monitor,
                    ensemble_size=self.ensemble_size,
                ),
            )
        set_twins(self, twins=twins)


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedMonitor:
    """A monitor's score reported as its smoothed conformal p-value among honest scores.

    The twins are hyoshigi's `CalibratedMonitor` over `monitor`'s twins.
    `honest_scores` is read once, as hyoshigi reads it, and both twins rank
    among the same scores.
    """

    monitor: MonitorConfiguration
    honest_scores: InitVar[Iterable[float]]
    random_seed: int | None = None
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self, honest_scores: Iterable[float]) -> None:
        """Build both twins over the wrapped monitor's."""
        check_monitor_option(self.monitor, parameter_name="monitor")
        inner = self.monitor.twins
        with raise_as_configuration_error():
            monitor = hyoshigi.CalibratedMonitor(
                monitor=inner.monitor,
                honest_scores=honest_scores,
                random_seed=self.random_seed,
            )
            twins = MonitorTwins(
                monitor=monitor,
                async_monitor=hyoshigi.AsyncCalibratedMonitor(
                    monitor=inner.async_monitor,
                    honest_scores=monitor.sorted_honest_scores,
                    random_seed=self.random_seed,
                ),
            )
        set_twins(self, twins=twins)


@dataclass(frozen=True, slots=True, kw_only=True)
class CascadeMonitor:
    """A fast monitor, and a careful one for the actions whose fast score is high.

    The twins are hyoshigi's `CascadeMonitor` over the two monitors' twins.
    """

    fast: MonitorConfiguration
    careful: MonitorConfiguration
    escalation_threshold: float
    twins: MonitorTwins = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build both twins over the two monitors'."""
        check_monitor_option(self.fast, parameter_name="fast")
        check_monitor_option(self.careful, parameter_name="careful")
        fast, careful = self.fast.twins, self.careful.twins
        with raise_as_configuration_error():
            twins = MonitorTwins(
                monitor=hyoshigi.CascadeMonitor(
                    fast=fast.monitor,
                    careful=careful.monitor,
                    escalation_threshold=self.escalation_threshold,
                ),
                async_monitor=hyoshigi.AsyncCascadeMonitor(
                    fast=fast.async_monitor,
                    careful=careful.async_monitor,
                    escalation_threshold=self.escalation_threshold,
                ),
            )
        set_twins(self, twins=twins)


type MonitorConfiguration = (
    LLMMonitor
    | GuardMonitor
    | DecisionMonitor
    | RepeatedMonitor
    | CalibratedMonitor
    | CascadeMonitor
    | MonitorTwins
)
"""A monitor the middleware and the wrappers take."""

MONITOR_CONFIGURATIONS: Final = (
    LLMMonitor,
    GuardMonitor,
    DecisionMonitor,
    RepeatedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    MonitorTwins,
)
"""The classes of `MonitorConfiguration`, for checks at run time."""


def check_monitor_option(value: object, *, parameter_name: str) -> None:
    """Raise `ConfigurationError` unless `value` is one of the monitor configurations."""
    if isinstance(value, MONITOR_CONFIGURATIONS):
        return
    names = ", ".join(monitor_class.__name__ for monitor_class in MONITOR_CONFIGURATIONS)
    message = (
        f"{parameter_name} must be one of {names}, got {describe_option_value(value)}. "
        "A monitor of your own is MonitorTwins(monitor=..., async_monitor=...), with "
        "hyoshigi's sync and async twins."
    )
    raise ConfigurationError(message)


def repeats_prompts_to_cached_model(monitor: MonitorConfiguration) -> bool:
    """Tell whether the monitor asks a cached chat model the same prompt more than once.

    An LLM monitor with parse retries asks again after an unreadable reply,
    a guard that samples more than once asks for each, and a repeated monitor
    of more than one call rates the same action again with a monitor that
    calls a cached model. Wrappers are read through. A user's own twins and a
    decision monitor call no LangChain model the library can see.
    """
    match monitor:
        case LLMMonitor() | GuardMonitor():
            return repeats_own_prompt(monitor)
        case RepeatedMonitor():
            return repeats_wrapped_rating(monitor)
        case CalibratedMonitor():
            return repeats_prompts_to_cached_model(monitor.monitor)
        case CascadeMonitor():
            return any(map(repeats_prompts_to_cached_model, (monitor.fast, monitor.careful)))
        case DecisionMonitor() | MonitorTwins():
            return False
        case unreachable:
            assert_never(unreachable)


def repeats_own_prompt(monitor: LLMMonitor | GuardMonitor) -> bool:
    """Tell whether one rating can ask the monitor's cached model the same prompt again."""
    match monitor:
        case LLMMonitor():
            repeats = monitor.max_parse_retries > 0
        case GuardMonitor():
            # A guard asks once to score log-probabilities, else once per sample.
            repeats = monitor.top_log_probabilities == 0 and monitor.samples > 1
        case unreachable:
            assert_never(unreachable)
    return repeats and is_response_cache_active(monitor.chat_model)


def repeats_wrapped_rating(monitor: RepeatedMonitor) -> bool:
    """Tell whether a repeated monitor asks a cached model again: by its calls, or within one."""
    if monitor.ensemble_size > 1:
        return has_cached_model(monitor.monitor)
    return repeats_prompts_to_cached_model(monitor.monitor)


def has_cached_model(monitor: MonitorConfiguration) -> bool:
    """Tell whether any chat model the monitor calls, wrapped ones included, is cached."""
    match monitor:
        case LLMMonitor() | GuardMonitor():
            return is_response_cache_active(monitor.chat_model)
        case RepeatedMonitor() | CalibratedMonitor():
            return has_cached_model(monitor.monitor)
        case CascadeMonitor():
            return has_cached_model(monitor.fast) or has_cached_model(monitor.careful)
        case DecisionMonitor() | MonitorTwins():
            return False
        case unreachable:
            assert_never(unreachable)


@functools.cache
def warn_about_cached_monitor_replies() -> None:
    """Emit the `CachedResampleWarning` for a monitor's model, on the first call in the process.

    The cache on this function keeps the warning to one per process, as
    `pending_steps.warn_about_cached_resamples` does for the agent's model.
    It points at the line that built the middleware, past its `__post_init__`.
    """
    warnings.warn(
        "A LangChain response cache is active for a monitor's model, so each repeat of its "
        "prompt returns a copy of the first reply: a guard's samples all carry the first "
        "label, and an LLM monitor asked again after an unreadable reply gets the same reply. "
        "Build the monitor's model with cache=False, or unset the global cache with "
        "set_llm_cache(None).",
        CachedResampleWarning,
        stacklevel=CONSTRUCTOR_CALLER_LEVEL,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )
