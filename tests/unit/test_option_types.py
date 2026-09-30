"""Constructors check the type of their options when they are built.

Protocols, fallbacks, the middleware's monitor, protocol and enum options, the
monitors, their decision models and `MonitorView` refuse an option of the
wrong type with `ConfigurationError`. The middleware's `label` and
`agent_name`, and `OpenRouterDecisionModel`'s `api_key`, are not checked here.
A limit given as a float, or a protocol given where a fallback belongs, used to
build without error and fail only at the first suspicious step: with a
`TypeError` or an `AttributeError`, during the attack the protocol exists for.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import FunctionType
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_sync_monitors.contracts import Channel, MonitorInput, MonitorView, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitors import (
    IS_THE_STEP_SUSPICIOUS,
    CalibratedMonitor,
    CascadeMonitor,
    DecisionModelMonitor,
    GuardModelMonitor,
    LLMMonitor,
    OpenRouterDecisionModel,
    RepeatedMonitor,
    TypeSafeDecisionModel,
    YesNoQuestion,
)
from langchain_sync_monitors.options import describe_option_value
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    DeferToTrustedModel,
    FeedbackTemplate,
    HaltRun,
    TrustedMonitoring,
)
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst
from tests.unit.monitors.doubles import CallPath, evaluate_on_path

WRONG_COUNTS = [2.5, 2.0, "3", True, None]
WRONG_COUNT_IDS = ["float", "whole-float", "string", "bool", "none"]


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of `int`."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


def build_trusted_model() -> FakeListChatModel:
    return FakeListChatModel(responses=["unused"])


def build_defer_to_trusted() -> DeferToTrusted:
    return DeferToTrusted(
        trusted_model=build_trusted_model(),
        defer_threshold=0.6,
        audit_threshold=0.9,
    )


def build_auto_mode(**options: Any) -> AutoMode:
    return AutoMode(block_threshold=0.6, **options)


def build_defer_to_resample(**options: Any) -> DeferToResample:
    settings: dict[str, Any] = {"fallback": HaltRun(), **options}
    return DeferToResample(defer_threshold=0.6, audit_threshold=0.9, **settings)


@pytest.mark.parametrize("value", WRONG_COUNTS, ids=WRONG_COUNT_IDS)
@pytest.mark.parametrize("parameter_name", ["max_consecutive_blocks", "max_total_blocks"])
def test_auto_mode_refuses_a_block_limit_that_is_not_a_whole_number(
    parameter_name: str,
    value: object,
) -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=f"{parameter_name} must be a whole number of at least 1",
    ):
        build_auto_mode(**{parameter_name: value})


@pytest.mark.parametrize("value", WRONG_COUNTS, ids=WRONG_COUNT_IDS)
def test_defer_to_resample_refuses_a_resample_count_that_is_not_a_whole_number(
    value: object,
) -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match="max_resamples must be a whole number of at least 0",
    ):
        build_defer_to_resample(max_resamples=value)


def test_the_smallest_counts_are_accepted() -> None:
    # Act
    auto_mode = build_auto_mode(max_consecutive_blocks=1, max_total_blocks=1)
    defer_to_resample = build_defer_to_resample(max_resamples=0)

    # Assert
    assert (auto_mode.max_consecutive_blocks, auto_mode.max_total_blocks) == (1, 1)
    assert defer_to_resample.max_resamples == 0


def test_integers_that_are_not_python_ints_are_accepted_as_ints() -> None:
    # Act
    auto_mode = build_auto_mode(
        max_consecutive_blocks=ArrayInteger(2),
        max_total_blocks=ArrayInteger(5),
    )
    defer_to_resample = build_defer_to_resample(max_resamples=ArrayInteger(3))

    # Assert
    assert (auto_mode.max_consecutive_blocks, auto_mode.max_total_blocks) == (2, 5)
    assert defer_to_resample.max_resamples == 3
    assert type(auto_mode.max_consecutive_blocks) is int
    assert type(defer_to_resample.max_resamples) is int


def test_infinity_lifts_the_total_block_limit() -> None:
    # Act
    auto_mode = build_auto_mode(max_total_blocks=math.inf)

    # Assert
    assert auto_mode.max_total_blocks == math.inf


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_consecutive_blocks": math.inf}, "max_consecutive_blocks must be a whole number"),
        ({"max_total_blocks": -math.inf}, "max_total_blocks must be a whole number"),
        ({"max_total_blocks": math.nan}, "or math.inf for no limit, got nan"),
        ({"max_total_blocks": 0}, "max_total_blocks must be a whole number of at least 1"),
    ],
    ids=["consecutive-infinity", "negative-infinity", "nan", "zero"],
)
def test_only_the_total_block_limit_accepts_infinity(
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build_auto_mode(**options)


def test_infinity_is_refused_as_a_resample_count() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError, match="max_resamples must be a whole number of at least 0, got inf"
    ):
        build_defer_to_resample(max_resamples=math.inf)


@pytest.mark.parametrize(
    "value", ["halt", None, build_trusted_model()], ids=["string", "none", "model"]
)
def test_auto_mode_refuses_a_limit_fallback_that_is_not_a_fallback(value: object) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="when_limit_reached must be a Fallback"):
        build_auto_mode(when_limit_reached=value)


def test_a_protocol_given_as_a_fallback_points_to_the_fallback_it_resembles() -> None:
    # Arrange
    protocol = build_defer_to_trusted()

    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"fallback must be a Fallback, got an instance of DeferToTrusted\. DeferToTrusted "
        r"is a control "
        r"protocol, .* use DeferToTrustedModel\(trusted_model=\.\.\.\)",
    ):
        build_defer_to_resample(fallback=protocol)


def test_a_protocol_given_as_the_limit_fallback_points_to_the_fallback_it_resembles() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"use DeferToTrustedModel\(trusted_model="):
        build_auto_mode(when_limit_reached=build_defer_to_trusted())


def test_a_fallback_given_as_the_protocol_points_to_the_protocol_it_resembles() -> None:
    # Arrange
    fallback = DeferToTrustedModel(trusted_model=build_trusted_model())

    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"protocol must be a ControlProtocol, got an instance of DeferToTrustedModel\. "
        r"DeferToTrustedModel is a fallback, .* use DeferToTrusted\(trusted_model=\.\.\.\)",
    ):
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=fallback)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"monitor": "openrouter:some/judge"},
            "monitor must be a Monitor, got 'openrouter:some/judge'",
        ),
        ({"protocol": "auto_mode"}, "protocol must be a ControlProtocol, got 'auto_mode'"),
    ],
    ids=["monitor", "protocol"],
)
def test_the_middleware_refuses_a_monitor_or_protocol_of_the_wrong_type(
    options: dict[str, object],
    message: str,
) -> None:
    # Arrange
    arguments: dict[str, Any] = {"monitor": KeywordMonitor(), "protocol": AcceptFirst(), **options}

    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        MonitorMiddleware(**arguments)


def test_auto_mode_refuses_feedback_given_as_plain_text() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"feedback must be a FeedbackTemplate, got 'Blocked: \{reason\}'\. Wrap the text",
    ):
        build_auto_mode(feedback="Blocked: {reason}")


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: FeedbackTemplate(text=42), "text must be a str, got 42"),  # ty: ignore[invalid-argument-type]
        (lambda: HaltRun(message=None), "message must be a str, got None"),  # ty: ignore[invalid-argument-type]
    ],
    ids=["feedback-text", "halt-message"],
)
def test_a_message_that_is_not_text_is_refused(build: Any, message: str) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build()


def test_infinity_that_is_not_a_float_is_refused_as_the_total_limit() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="max_total_blocks must be a whole number"):
        build_auto_mode(max_total_blocks=Decimal("Infinity"))


def test_a_foreign_type_that_shares_a_builtin_name_is_named_with_its_module() -> None:
    # Arrange: numpy's bool is not Python's, and is named bool too
    foreign_bool = type("bool", (), {"__module__": "numpy"})()

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"got an instance of numpy\.bool$"):
        build_auto_mode(max_consecutive_blocks=foreign_bool)


def build_protocol_with_threshold(parameter_name: str, value: object) -> object:
    thresholds: dict[str, Any] = {parameter_name: value}
    if parameter_name == "flag_threshold":
        return TrustedMonitoring(**thresholds)
    if parameter_name == "block_threshold":
        return AutoMode(**thresholds)
    settings: dict[str, Any] = {"defer_threshold": 0.6, "audit_threshold": 0.9, **thresholds}
    return DeferToResample(fallback=HaltRun(), **settings)


THRESHOLD_PARAMETERS = ["flag_threshold", "block_threshold", "defer_threshold", "audit_threshold"]


@pytest.mark.parametrize("parameter_name", THRESHOLD_PARAMETERS)
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0.6", "must be a number between 0 and 1, got '0.6'"),
        (True, "must be a number between 0 and 1, got True"),
        ([0.6], r"must be a number between 0 and 1, got an instance of list"),
        (math.nan, "must be between 0 and 1, got nan"),
        (1.5, "must be between 0 and 1, got 1.5"),
        (-0.1, "must be between 0 and 1, got -0.1"),
        (Decimal("sNaN"), r"must be between 0 and 1, got Decimal\('sNaN'\)"),
    ],
    ids=["string", "bool", "list", "nan", "above-one", "below-zero", "signalling-nan"],
)
def test_a_threshold_that_is_not_a_number_from_zero_to_one_is_refused(
    parameter_name: str,
    value: object,
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=f"{parameter_name} {message}"):
        build_protocol_with_threshold(parameter_name, value)


@pytest.mark.parametrize("parameter_name", THRESHOLD_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [0, 1, 0.6, Fraction(3, 5), Decimal("0.6")],
    ids=["zero", "one", "float", "fraction", "decimal"],
)
def test_a_threshold_given_as_any_real_number_from_zero_to_one_is_kept_as_a_float(
    parameter_name: str,
    value: float | Fraction | Decimal,
) -> None:
    # Act
    protocol = build_protocol_with_threshold(parameter_name, value)

    # Assert
    threshold = getattr(protocol, parameter_name)
    assert type(threshold) is float
    assert threshold == float(value)


@pytest.mark.parametrize("parameter_name", THRESHOLD_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000001"),
        Decimal("-1E-400"),
        Fraction(10**20 + 1, 10**20),
        Fraction(-1, 10**400),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        10**400,
        Fraction(10**400, 1),
        math.inf,
    ],
    ids=[
        "decimal-just-above-one",
        "decimal-just-below-zero",
        "fraction-just-above-one",
        "fraction-just-below-zero",
        "decimal-nan",
        "decimal-signalling-nan",
        "decimal-infinity",
        "huge-int",
        "huge-fraction",
        "infinity",
    ],
)
def test_a_threshold_just_outside_the_range_is_refused_before_it_is_rounded(
    parameter_name: str,
    value: object,
) -> None:
    # Act / Assert: a float would round the first four into [0, 1]
    with pytest.raises(ConfigurationError, match=f"{parameter_name} must be between 0 and 1"):
        build_protocol_with_threshold(parameter_name, value)


def test_a_negative_zero_threshold_is_read_as_zero() -> None:
    # Act
    protocol = TrustedMonitoring(flag_threshold=-0.0)

    # Assert
    threshold = protocol.flag_threshold
    assert threshold == 0.0
    assert math.copysign(1.0, threshold) == 1.0


@dataclass(frozen=True, kw_only=True, eq=False)
class RenamedMonitorMiddleware(MonitorMiddleware):
    """A subclass that names itself apart from its label and agent."""

    public_name: str = "outer"

    @property
    def name(self) -> str:
        return self.public_name


def test_a_monitor_named_apart_from_its_label_and_agent_is_refused() -> None:
    # Act / Assert: two such monitors could share a label and agent under different names
    with pytest.raises(
        ConfigurationError,
        match=r"a monitor's name must be label\[agent_name\], monitor\[main\], got 'outer'",
    ):
        RenamedMonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())


POLICY_PROMPT = ChatPromptTemplate.from_messages([("human", "{transcript}{proposed_step}")])
QUESTION_FIELDS: dict[str, Any] = {"key": "k", "text": "Leak?", "yes_means": "Y", "no_means": "N"}
RUNNABLE_CLASSIFIER = RunnableLambda(lambda request: request)
ONE_QUESTION = (IS_THE_STEP_SUSPICIOUS,)


@pytest.fixture(autouse=True)
def openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the decision model a key, so no refusal below is the missing key's."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")


def merge_options(defaults: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **options}


def build_llm_monitor(**options: Any) -> LLMMonitor:
    return LLMMonitor(**merge_options({"model": build_trusted_model()}, options))


def build_guard_monitor(**options: Any) -> GuardModelMonitor:
    labels = {"suspicious_labels": {"violation"}, "safe_labels": {"no_violation"}}
    defaults = {"model": build_trusted_model(), "policy_prompt": POLICY_PROMPT, **labels}
    return GuardModelMonitor(**merge_options(defaults, options))


def build_decision_model(**options: Any) -> OpenRouterDecisionModel:
    return OpenRouterDecisionModel(**merge_options({"model": "typesafe/jev-1.13"}, options))


def build_decision_monitor(**options: Any) -> DecisionModelMonitor:
    defaults = {"decision_model": build_decision_model()}
    return DecisionModelMonitor(**merge_options(defaults, options))


def build_typesafe_model(**options: Any) -> TypeSafeDecisionModel:
    pytest.importorskip("langchain_typesafe")
    return TypeSafeDecisionModel(**options)


def build_question(**options: Any) -> YesNoQuestion:
    return YesNoQuestion(**merge_options(QUESTION_FIELDS, options))


def build_repeated_monitor(**options: Any) -> RepeatedMonitor:
    return RepeatedMonitor(**merge_options({"monitor": KeywordMonitor(), "repeats": 2}, options))


def build_calibrated_monitor(**options: Any) -> CalibratedMonitor:
    defaults = {"monitor": KeywordMonitor(), "honest_scores": [0.1]}
    return CalibratedMonitor(**merge_options(defaults, options))


def build_cascade_monitor(**options: Any) -> CascadeMonitor:
    defaults = {"fast": KeywordMonitor(), "careful": KeywordMonitor(), "escalate_at": 0.5}
    return CascadeMonitor(**merge_options(defaults, options))


type Build = Callable[..., object]

WRONG_WHOLE_NUMBERS: list[object] = [True, 2.5, 2.0, "3", math.nan]
WRONG_THRESHOLDS: list[object] = [True, "0.5", None, math.nan, 1.5, -0.1, Decimal("sNaN")]
WRONG_OBJECTS: list[object] = ["text", True, math.nan, None]
WRONG_TEXTS: list[object] = [5, True, math.nan, None]
WRONG_DURATIONS: list[object] = [0, -1, math.inf, math.nan, 10**400, Decimal("1E-400")]
WHOLE = "be a whole number of at least"
BETWEEN_ZERO_AND_ONE = "be (a number )?between 0 and 1, got"
REFUSAL_RULES: list[tuple[Build, str, list[object], str]] = [
    (build_repeated_monitor, "repeats", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 1"),
    (build_repeated_monitor, "repeats", [0, -1], "be at least 1, got"),
    (build_guard_monitor, "samples", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 1"),
    (build_guard_monitor, "samples", [0], "be at least 1, got 0"),
    (build_llm_monitor, "max_parse_retries", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 0"),
    (build_llm_monitor, "max_parse_retries", [-1], "be at least 0, got -1"),
    (MonitorView, "most_recent_entries", WRONG_WHOLE_NUMBERS, f"{WHOLE} 1, or None"),
    (MonitorView, "most_recent_entries", [0, -1], "be at least 1, or None to keep every entry"),
    (build_llm_monitor, "lowest_score", [*WRONG_WHOLE_NUMBERS, None], "be an integer, got"),
    (build_llm_monitor, "highest_score", [*WRONG_WHOLE_NUMBERS, None], "be an integer, got"),
    (build_calibrated_monitor, "random_seed", WRONG_WHOLE_NUMBERS, "be an integer, got"),
    (build_cascade_monitor, "escalate_at", WRONG_THRESHOLDS, BETWEEN_ZERO_AND_ONE),
    (build_calibrated_monitor, "honest_scores", ["0.1", True, 0.5, None], "be an iterable of"),
    (build_calibrated_monitor, "honest_scores", [[]], "hold at least one score"),
    (build_decision_model, "timeout_seconds", [True, "30", None], "be a positive number, got"),
    (build_decision_model, "timeout_seconds", WRONG_DURATIONS, "be a positive, finite number"),
    (build_decision_model, "timeout_seconds", [Decimal("sNaN")], "be a positive, finite number"),
    (build_llm_monitor, "model", [True, math.nan, None], "be a LangChain chat model or a"),
    (build_llm_monitor, "prompt", WRONG_OBJECTS, "be a ChatPromptTemplate, got"),
    (build_guard_monitor, "policy_prompt", WRONG_OBJECTS, "be a ChatPromptTemplate, got"),
    (build_llm_monitor, "view", WRONG_OBJECTS, "be a MonitorView, got"),
    (build_guard_monitor, "view", WRONG_OBJECTS, "be a MonitorView, got"),
    (build_decision_monitor, "view", WRONG_OBJECTS, "be a MonitorView, got"),
    (build_guard_monitor, "suspicious_labels", [*WRONG_OBJECTS, ["x"]], "be a set of strings"),
    (build_guard_monitor, "suspicious_labels", [{5}], "hold only strings, got 5"),
    (build_guard_monitor, "safe_labels", [*WRONG_OBJECTS, ["x"]], "be a set of strings"),
    (build_guard_monitor, "safe_labels", [{None}], "hold only strings, got None"),
    (build_guard_monitor, "scoring", WRONG_OBJECTS, "be one of GuardScoring.AUTO"),
    (build_decision_monitor, "combine", WRONG_OBJECTS, "be one of Combine.MAX"),
    (build_decision_monitor, "decision_model", WRONG_OBJECTS, "be a DecisionModel, got"),
    (build_decision_monitor, "questions", [*WRONG_OBJECTS, *ONE_QUESTION], "be an iterable of"),
    (build_decision_monitor, "questions", [[], ONE_QUESTION * 2], "be non-empty, with unique"),
    (build_decision_model, "model", WRONG_TEXTS, "be a str, got"),
    (build_decision_model, "base_url", WRONG_TEXTS, "be a str, got"),
    (build_decision_model, "http_client", [httpx.AsyncClient(), "x"], "be a Client, got"),
    (build_decision_model, "async_http_client", [httpx.Client(), "x"], "be an AsyncClient"),
    (build_typesafe_model, "classifier", WRONG_OBJECTS, "be a Runnable, got"),
    (build_repeated_monitor, "monitor", [*WRONG_OBJECTS, build_trusted_model()], "be a Monitor"),
    (build_calibrated_monitor, "monitor", WRONG_OBJECTS, "be a Monitor, got"),
    (build_cascade_monitor, "fast", WRONG_OBJECTS, "be a Monitor, got"),
    (build_cascade_monitor, "careful", WRONG_OBJECTS, "be a Monitor, got"),
    (MonitorView, "channels", [*WRONG_OBJECTS, 1], "be a Channel, got"),
    (MonitorView, "delegation_tools", [*WRONG_OBJECTS, ["task"]], "be a set of strings, such as"),
    (MonitorView, "delegation_tools", [{5}], "hold only strings, got 5"),
    *[(build_question, field, WRONG_TEXTS, "be a str, got") for field in QUESTION_FIELDS],
]


def name_builder(build: Build) -> str:
    """Name a builder for a test id: a function or a class by its name."""
    return build.__name__ if isinstance(build, FunctionType | type) else repr(build)


REFUSAL_CASES = [
    pytest.param(
        build,
        {parameter: value},
        f"{parameter} must {message}",
        id=f"{name_builder(build)}-{parameter}-{describe_option_value(value)}",
    )
    for build, parameter, values, message in REFUSAL_RULES
    for value in values
]


@pytest.mark.parametrize(("build", "options", "message"), REFUSAL_CASES)
def test_a_monitor_or_view_refuses_an_option_of_the_wrong_type(
    build: Build,
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert: the message names the parameter and what it got
    with pytest.raises(ConfigurationError, match=message):
        build(**options)


ITEM_REFUSAL_CASES = [
    *[
        pytest.param(
            {"honest_scores": [0.2, value]},
            rf"honest_scores\[1\] must {BETWEEN_ZERO_AND_ONE}",
            id=f"honest-score-{describe_option_value(value)}",
        )
        for value in WRONG_THRESHOLDS
    ],
    pytest.param(
        {"questions": [IS_THE_STEP_SUSPICIOUS, "leaks"]},
        r"questions\[1\] must be a YesNoQuestion, got 'leaks'",
        id="question-string",
    ),
]


@pytest.mark.parametrize(("options", "message"), ITEM_REFUSAL_CASES)
def test_a_refused_item_is_named_by_its_position(options: dict[str, object], message: str) -> None:
    # Arrange
    build = build_calibrated_monitor if "honest_scores" in options else build_decision_monitor

    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build(**options)


def test_the_ends_of_a_scale_are_read_before_they_are_compared() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"lowest_score \(3\) must be below highest_score"):
        build_llm_monitor(lowest_score=ArrayInteger(3), highest_score=3)


SCORES_IN_ANY_FORM = iter([Decimal("0.5"), Fraction(1, 4), 1, 0, -0.0])
ACCEPTED_CASES: list[tuple[Build, dict[str, object], str, object]] = [
    (build_repeated_monitor, {"repeats": 1}, "repeats", 1),
    (build_repeated_monitor, {"repeats": ArrayInteger(3)}, "repeats", 3),
    (build_guard_monitor, {"samples": 1}, "samples", 1),
    (build_guard_monitor, {"samples": ArrayInteger(2)}, "samples", 2),
    (build_llm_monitor, {"max_parse_retries": 0}, "max_parse_retries", 0),
    (build_llm_monitor, {"max_parse_retries": ArrayInteger(1)}, "max_parse_retries", 1),
    (build_llm_monitor, {"lowest_score": -5, "highest_score": -4}, "highest_score", -4),
    (build_llm_monitor, {"lowest_score": ArrayInteger(9)}, "lowest_score", 9),
    (MonitorView, {"most_recent_entries": 1}, "most_recent_entries", 1),
    (MonitorView, {"most_recent_entries": ArrayInteger(2)}, "most_recent_entries", 2),
    (MonitorView, {"most_recent_entries": None}, "most_recent_entries", None),
    (MonitorView, {"channels": Channel.USER | Channel.REASONING}, "channels", Channel(3)),
    (MonitorView, {"delegation_tools": {"ask"}}, "delegation_tools", frozenset({"ask"})),
    (build_calibrated_monitor, {"random_seed": None}, "random_seed", None),
    (build_calibrated_monitor, {"random_seed": -3}, "random_seed", -3),
    (build_calibrated_monitor, {"random_seed": ArrayInteger(7)}, "random_seed", 7),
    (build_calibrated_monitor, {"honest_scores": [0.2, 1]}, "sorted_honest_scores", [0.2, 1.0]),
    (
        build_calibrated_monitor,
        {"honest_scores": SCORES_IN_ANY_FORM},
        "sorted_honest_scores",
        [0.0, 0.0, 0.25, 0.5, 1.0],
    ),
    (build_cascade_monitor, {"escalate_at": 0}, "escalate_at", 0.0),
    (build_cascade_monitor, {"escalate_at": 1}, "escalate_at", 1.0),
    (build_cascade_monitor, {"escalate_at": -0.0}, "escalate_at", 0.0),
    (build_cascade_monitor, {"escalate_at": Decimal("0.3")}, "escalate_at", 0.3),
    (build_decision_model, {"timeout_seconds": 30}, "timeout_seconds", 30.0),
    (build_decision_model, {"timeout_seconds": 5e-324}, "timeout_seconds", 5e-324),
    (build_decision_model, {"timeout_seconds": Decimal("0.5")}, "timeout_seconds", 0.5),
    (build_decision_model, {"async_http_client": None}, "async_http_client", None),
    (build_decision_monitor, {"questions": iter(ONE_QUESTION)}, "questions", ONE_QUESTION),
    (build_typesafe_model, {"classifier": RUNNABLE_CLASSIFIER}, "classifier", RUNNABLE_CLASSIFIER),
]


@pytest.mark.parametrize(("build", "options", "attribute", "expected"), ACCEPTED_CASES)
def test_a_valid_option_is_kept_as_a_plain_value(
    build: Build,
    options: dict[str, object],
    attribute: str,
    expected: object,
) -> None:
    # Act
    built = build(**options)

    # Assert: the repr tells an int from numpy's, a float from a Decimal and 0.0 from -0.0
    kept = getattr(built, attribute)
    assert kept == expected
    assert repr(kept) == repr(expected)


@pytest.mark.parametrize("call_path", ["async", "sync"])
async def test_a_decimal_escalation_threshold_escalates_the_float_score_it_names(
    call_path: CallPath,
) -> None:
    # Arrange: the float 0.3 lies below Decimal("0.3"), so a Decimal kept as given missed it
    step = MonitorInput(
        history=(HumanMessage("Summarise q3.md."),),
        proposal=AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {"path": "q3.md"}}],
        ),
        task_author=TaskAuthor.USER,
    )
    cascade = build_cascade_monitor(
        fast=KeywordMonitor(suspicion_by_keyword={"read_file": 0.3}),
        careful=KeywordMonitor(suspicion_by_keyword={"read_file": 0.9}),
        escalate_at=Decimal("0.3"),
    )

    # Act
    verdict = await evaluate_on_path(cascade, step, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.9
