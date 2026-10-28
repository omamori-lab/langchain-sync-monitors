"""Constructors and functions check the type of their options when they are called.

The middleware, the monitors, the feedback template, `monitor_subagents` and
`check_monitor_placement` refuse an option of the wrong type with
`ConfigurationError`. An option a monitor or a protocol passes to hyoshigi is
refused with hyoshigi's message, raised again as the library's error.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from operator import attrgetter
from pathlib import Path
from types import FunctionType
from typing import Any

import hyoshigi
import pytest
from hyoshigi import Action, Author, Observation, Step, TextPart, ToolCall, Trajectory
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors import (
    BlockAndFeedBack,
    CalibratedMonitor,
    CascadeMonitor,
    DecisionMonitor,
    DeferToResample,
    FailureResponse,
    GuardLabels,
    GuardMonitor,
    LLMMonitor,
    MonitorPrompt,
    MonitorTwins,
    RepeatedMonitor,
    Resampling,
)
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.options import describe_option_value
from langchain_sync_monitors.placement import check_monitor_placement
from langchain_sync_monitors.protocols import DEFAULT_FEEDBACK_TEMPLATE, FeedbackTemplate
from tests.support.monitors import KeywordMonitor, ScriptedGeneration
from tests.support.protocols import RUN_EVERY_STEP
from tests.support.redaction import mask_secret
from tests.unit.monitors.doubles import CallPath, rate_on_path


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of `int`."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


def build_trusted_model() -> FakeListChatModel:
    return FakeListChatModel(responses=["unused"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"monitor": "openrouter:some/monitor-model"},
            "monitor must be one of LLMMonitor, .*, got 'openrouter:some/monitor-model'",
        ),
        (
            {"protocol": "auto_mode"},
            "protocol must be one of DeferToResample, BlockAndFeedBack, got 'auto_mode'",
        ),
    ],
    ids=["monitor", "protocol"],
)
def test_the_middleware_refuses_a_monitor_or_protocol_of_the_wrong_type(
    options: dict[str, object],
    message: str,
) -> None:
    # Arrange
    arguments: dict[str, Any] = {"monitor": KeywordMonitor(), "protocol": RUN_EVERY_STEP, **options}

    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        MonitorMiddleware(**arguments)


def test_the_middleware_refuses_feedback_given_as_plain_text() -> None:
    # Act / Assert
    with pytest.raises(
        ConfigurationError,
        match=r"feedback must be a FeedbackTemplate, got 'Blocked: \{reason\}'\. Wrap the text",
    ):
        build_middleware(feedback="Blocked: {reason}")


@pytest.mark.xfail(strict=True, reason="PR 8 redacts the trajectory at entry")
@pytest.mark.parametrize(
    ("redaction", "message"),
    [
        (mask_secret, r"redaction must be a Redaction, got an instance of function\. Build one"),
        ("mask the token", r"redaction must be a Redaction, got 'mask the token'\. Build one"),
    ],
    ids=["a-function", "plain-text"],
)
def test_the_middleware_refuses_a_redaction_that_is_not_a_redaction(
    redaction: object,
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build_middleware(redaction=redaction)


def test_a_feedback_text_that_is_not_text_is_refused() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="text must be a str, got 42"):
        FeedbackTemplate(text=42)  # ty: ignore[invalid-argument-type]


def test_a_foreign_type_that_shares_a_builtin_name_is_named_with_its_module() -> None:
    # Arrange: numpy's bool is not Python's, and is named bool too
    foreign_bool = type("bool", (), {"__module__": "numpy"})()

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"got an instance of numpy\.bool$"):
        build_middleware(flag_threshold=foreign_bool)


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
def test_a_flag_threshold_that_is_not_a_number_from_zero_to_one_is_refused(
    value: object,
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=f"flag_threshold {message}"):
        build_middleware(flag_threshold=value)


@pytest.mark.parametrize(
    "value",
    [0, 1, 0.6, Fraction(3, 5), Decimal("0.6")],
    ids=["zero", "one", "float", "fraction", "decimal"],
)
def test_a_flag_threshold_given_as_any_real_number_from_zero_to_one_is_kept_as_a_float(
    value: float | Fraction | Decimal,
) -> None:
    # Act
    middleware = build_middleware(flag_threshold=value)

    # Assert
    threshold = middleware.flag_threshold
    assert type(threshold) is float
    assert threshold == float(value)


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
def test_a_flag_threshold_just_outside_the_range_is_refused_before_it_is_rounded(
    value: object,
) -> None:
    # Act / Assert: a float would round the first four into [0, 1]
    with pytest.raises(ConfigurationError, match="flag_threshold must be between 0 and 1"):
        build_middleware(flag_threshold=value)


def test_a_negative_zero_flag_threshold_is_read_as_zero() -> None:
    # Act
    middleware = build_middleware(flag_threshold=-0.0)

    # Assert
    threshold = middleware.flag_threshold
    assert threshold == 0.0
    assert threshold is not None
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
        RenamedMonitorMiddleware(monitor=KeywordMonitor(), protocol=RUN_EVERY_STEP)


POLICY_PROMPT = MonitorPrompt(system="", request="{trajectory}{proposed_action}")
LABELS = GuardLabels(suspicious=frozenset({"violation"}), safe=frozenset({"no_violation"}))


@pytest.fixture(autouse=True)
def typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the decision model's clients a key, so no refusal below is the missing key's."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-key")


@pytest.fixture(autouse=True)
def openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the `openrouter:` decision model a key, so no refusal below is the missing key's."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")


def merge_options(defaults: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **options}


def build_llm_monitor(**options: Any) -> LLMMonitor:
    return LLMMonitor(**merge_options({"model": build_trusted_model()}, options))


def build_guard_monitor(**options: Any) -> GuardMonitor:
    defaults = {"model": build_trusted_model(), "prompt": POLICY_PROMPT, "labels": LABELS}
    return GuardMonitor(**merge_options(defaults, options))


def build_decision_monitor(**options: Any) -> DecisionMonitor:
    # A typesafe: model needs the typesafe extra, which the job without extras leaves out.
    pytest.importorskip("typesafe_sdk")
    return DecisionMonitor(**merge_options({"model": "typesafe:jev-1.14"}, options))


def build_openrouter_decision_monitor(**options: Any) -> DecisionMonitor:
    return DecisionMonitor(**merge_options({"model": "openrouter:typesafe/jev-1.14"}, options))


def build_monitor_twins(**options: Any) -> MonitorTwins:
    twins = ScriptedGeneration(outcome=hyoshigi.Generation(text="unused")).build_twins()
    defaults = {"monitor": twins.monitor, "async_monitor": twins.async_monitor}
    return MonitorTwins(**merge_options(defaults, options))


def build_repeated_monitor(**options: Any) -> RepeatedMonitor:
    return RepeatedMonitor(
        **merge_options({"monitor": KeywordMonitor(), "ensemble_size": 2}, options)
    )


def build_calibrated_monitor(**options: Any) -> CalibratedMonitor:
    defaults = {"monitor": KeywordMonitor(), "honest_scores": [0.1]}
    return CalibratedMonitor(**merge_options(defaults, options))


def build_cascade_monitor(**options: Any) -> CascadeMonitor:
    defaults = {"fast": KeywordMonitor(), "careful": KeywordMonitor(), "escalation_threshold": 0.5}
    return CascadeMonitor(**merge_options(defaults, options))


WORKER_SPEC = {"name": "worker", "description": "Finds.", "system_prompt": "Find."}


def build_middleware(**options: Any) -> MonitorMiddleware:
    defaults = {"monitor": KeywordMonitor(), "protocol": RUN_EVERY_STEP}
    return MonitorMiddleware(**merge_options(defaults, options))


def build_monitored_subagents(**options: Any) -> list[Any]:
    return monitor_subagents(**merge_options({"middleware": build_middleware()}, options))


type Build = Callable[..., object]

WRONG_WHOLE_NUMBERS: list[object] = [True, 2.5, 2.0, "3", math.nan]
WRONG_THRESHOLDS: list[object] = [True, "0.5", None, math.nan, 1.5, -0.1, Decimal("sNaN")]
WRONG_OBJECTS: list[object] = ["text", True, math.nan, None]
WRONG_TEXTS: list[object] = [5, True, math.nan, None]
WHOLE = "be a whole number of at least"
BETWEEN_ZERO_AND_ONE = "be (a number )?between 0 and 1, got"
REFUSAL_RULES: list[tuple[Build, str, list[object], str]] = [
    (build_repeated_monitor, "ensemble_size", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 1"),
    (build_repeated_monitor, "ensemble_size", [0, -1], "be at least 1, got"),
    (build_guard_monitor, "samples", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 1"),
    (build_guard_monitor, "samples", [0], "be at least 1, got 0"),
    (build_llm_monitor, "max_parse_retries", [*WRONG_WHOLE_NUMBERS, None], f"{WHOLE} 0"),
    (build_llm_monitor, "max_parse_retries", [-1], "be at least 0, got -1"),
    (build_llm_monitor, "lowest_score", [*WRONG_WHOLE_NUMBERS, None], "be an integer, got"),
    (build_llm_monitor, "highest_score", [*WRONG_WHOLE_NUMBERS, None], "be an integer, got"),
    (build_calibrated_monitor, "random_seed", WRONG_WHOLE_NUMBERS, "be an integer, got"),
    (build_cascade_monitor, "escalation_threshold", WRONG_THRESHOLDS, BETWEEN_ZERO_AND_ONE),
    (build_calibrated_monitor, "honest_scores", ["0.1", True, 0.5, None], "be an iterable of"),
    (build_calibrated_monitor, "honest_scores", [[]], "hold at least one score"),
    (build_llm_monitor, "model", [True, math.nan, None], "be a LangChain chat model or a"),
    (build_llm_monitor, "prompt", WRONG_OBJECTS, "be a MonitorPrompt, got"),
    (build_guard_monitor, "prompt", WRONG_OBJECTS, "be a MonitorPrompt, got"),
    (build_llm_monitor, "view", WRONG_OBJECTS, "be a View, got"),
    (build_guard_monitor, "view", WRONG_OBJECTS, "be a View, got"),
    (build_decision_monitor, "view", WRONG_OBJECTS, "be a View, got"),
    (build_guard_monitor, "labels", WRONG_OBJECTS, "be a GuardLabels, got"),
    (build_decision_monitor, "aggregation", WRONG_OBJECTS, "be an Aggregation, got"),
    (build_decision_monitor, "statements", [*WRONG_OBJECTS, ["x"]], "be a tuple, got"),
    (build_decision_monitor, "statements", [()], "not be empty"),
    (
        build_openrouter_decision_monitor,
        "perceives",
        [*WRONG_OBJECTS, ["image/png"]],
        "be a MediaTypes, got",
    ),
    (build_decision_monitor, "model", WRONG_TEXTS, "be a str, got"),
    (
        build_decision_monitor,
        "model",
        ["anthropic:x", "jev-1.14", "typesafe:"],
        "name a decision model",
    ),
    (build_repeated_monitor, "monitor", [*WRONG_OBJECTS, build_trusted_model()], "be one of"),
    (build_calibrated_monitor, "monitor", WRONG_OBJECTS, "be one of LLMMonitor, "),
    (build_cascade_monitor, "fast", WRONG_OBJECTS, "be one of LLMMonitor, "),
    (build_cascade_monitor, "careful", WRONG_OBJECTS, "be one of LLMMonitor, "),
    (build_monitor_twins, "monitor", WRONG_OBJECTS, "be a Monitor, got"),
    (build_monitor_twins, "async_monitor", WRONG_OBJECTS, "be an AsyncMonitor, got"),
    (build_middleware, "label", ["", " ", *WRONG_TEXTS], "be a non-blank string, got"),
    (build_middleware, "label", ["team:a", "team|a"], "not contain '[:|]', which LangGraph"),
    (build_middleware, "agent_name", ["", *WRONG_TEXTS], "be a non-blank string, got"),
    (build_middleware, "agent_name", ["sub:agent", "sub|agent"], "not contain '[:|]'"),
    (build_monitored_subagents, "middleware", WRONG_OBJECTS, "be a MonitorMiddleware, got"),
    (build_monitored_subagents, "middleware", [KeywordMonitor()], "be a MonitorMiddleware"),
    (build_monitored_subagents, "subagents", ["x", b"", True, None, WORKER_SPEC], "be a list of"),
    (build_monitored_subagents, "overrides", [["x"], "x", True], "map subagent names to Mon"),
    (build_monitored_subagents, "overrides", [{1: KeywordMonitor()}], "be keyed by subagent"),
    (build_monitored_subagents, "skills", ["/skills/"], "be a list of skill source paths, not"),
    (build_monitored_subagents, "skills", [5, True, b"x"], "be a list of skill source paths, got"),
    (check_monitor_placement, "middleware", ["m", b"", True, None, iter([])], "be the list given"),
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


HONEST_SCORE_REFUSAL = rf"honest_scores\[1\] must {BETWEEN_ZERO_AND_ONE}"
SKILL_SOURCE_REFUSAL = (
    r"skills\[1\] must be a skill source path or a \(path, label\) pair of strings"
)
WRONG_SKILL_SOURCES: list[object] = [
    5,
    Path("/skills/"),
    ("/skills/",),
    ("/skills/", 5),
    ("/skills/", "Label", "More"),
    ["/skills/", "Label"],
]
ITEM_REFUSAL_CASES: list[tuple[Build, dict[str, Any], str]] = [
    *[
        (build_calibrated_monitor, {"honest_scores": [0.2, value]}, HONEST_SCORE_REFUSAL)
        for value in WRONG_THRESHOLDS
    ],
    (build_monitored_subagents, {"subagents": ["worker"]}, r"subagents\[0\] must be a Mapping"),
    *[
        (build_monitored_subagents, {"subagents": [{**WORKER_SPEC, "name": name}]}, message)
        for name, message in [
            ("sub:agent", r"subagents\[0\]\['name'\] must not contain ':'"),
            ("sub|agent", r"subagents\[0\]\['name'\] must not contain '\|'"),
            ("", r"subagents\[0\]\['name'\] must be a non-blank string, got ''"),
            (5, r"subagents\[0\]\['name'\] must be a non-blank string, got 5"),
        ]
    ],
    (build_monitored_subagents, {"subagents": [{"description": "Finds."}]}, r"\['name'\] must"),
    *[
        (build_monitored_subagents, {"skills": ["/skills/", source]}, SKILL_SOURCE_REFUSAL)
        for source in WRONG_SKILL_SOURCES
    ],
    (
        build_monitored_subagents,
        {"subagents": [WORKER_SPEC], "overrides": {"worker": KeywordMonitor()}},
        r"overrides\['worker'\] must be a MonitorMiddleware",
    ),
    (check_monitor_placement, {"middleware": [KeywordMonitor()]}, r"middleware\[0\] must be an Ag"),
]


@pytest.mark.parametrize(("build", "options", "message"), ITEM_REFUSAL_CASES)
def test_a_refused_item_is_named_by_its_position(
    build: Build,
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build(**options)


def test_the_ends_of_a_scale_are_read_before_they_are_compared() -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=r"lowest_score \(3\) must be below highest_score"):
        build_llm_monitor(lowest_score=ArrayInteger(3), highest_score=3)


SCORES_IN_ANY_FORM = iter([Decimal("0.5"), Fraction(1, 4), 1, 0, -0.0])
ACCEPTED_CASES: list[tuple[Build, dict[str, object], str, object]] = [
    (build_repeated_monitor, {"ensemble_size": 1}, "twins.monitor.ensemble_size", 1),
    (build_repeated_monitor, {"ensemble_size": ArrayInteger(3)}, "twins.monitor.ensemble_size", 3),
    (build_guard_monitor, {"samples": 1}, "twins.async_monitor.samples", 1),
    (build_guard_monitor, {"samples": ArrayInteger(2)}, "twins.async_monitor.samples", 2),
    (build_llm_monitor, {"max_parse_retries": 0}, "twins.monitor.max_parse_retries", 0),
    (
        build_llm_monitor,
        {"max_parse_retries": ArrayInteger(1)},
        "twins.async_monitor.max_parse_retries",
        1,
    ),
    (
        build_llm_monitor,
        {"lowest_score": -5, "highest_score": -4},
        "twins.monitor.highest_score",
        -4,
    ),
    (build_llm_monitor, {"lowest_score": ArrayInteger(9)}, "twins.monitor.lowest_score", 9),
    (build_calibrated_monitor, {"random_seed": None}, "twins.monitor.random_seed", None),
    (build_calibrated_monitor, {"random_seed": -3}, "twins.async_monitor.random_seed", -3),
    (build_calibrated_monitor, {"random_seed": ArrayInteger(7)}, "twins.monitor.random_seed", 7),
    (
        build_calibrated_monitor,
        {"honest_scores": [0.2, 1]},
        "twins.async_monitor.sorted_honest_scores",
        (0.2, 1.0),
    ),
    (
        build_calibrated_monitor,
        {"honest_scores": SCORES_IN_ANY_FORM},
        "twins.async_monitor.sorted_honest_scores",
        (0.0, 0.0, 0.25, 0.5, 1.0),
    ),
    (build_cascade_monitor, {"escalation_threshold": 0}, "twins.monitor.escalation_threshold", 0.0),
    (
        build_cascade_monitor,
        {"escalation_threshold": 1},
        "twins.async_monitor.escalation_threshold",
        1.0,
    ),
    (
        build_cascade_monitor,
        {"escalation_threshold": -0.0},
        "twins.monitor.escalation_threshold",
        0.0,
    ),
    (
        build_cascade_monitor,
        {"escalation_threshold": Decimal("0.3")},
        "twins.monitor.escalation_threshold",
        0.3,
    ),
    (build_middleware, {"label": "team[a"}, "name", "team[a[main]"),
    (build_middleware, {"agent_name": "worker[v2]"}, "name", "monitor[worker[v2]]"),
    (build_middleware, {"label": "team.a", "agent_name": "sub a"}, "name", "team.a[sub a]"),
]


@pytest.mark.filterwarnings("ignore::hyoshigi.HardLabelWarning")
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
    kept = attrgetter(attribute)(built)
    assert kept == expected
    assert repr(kept) == repr(expected)


@pytest.mark.parametrize("call_path", ["async", "sync"])
async def test_a_decimal_escalation_threshold_escalates_the_float_score_it_names(
    call_path: CallPath,
) -> None:
    # Arrange: the float 0.3 lies below Decimal("0.3"), so a Decimal kept as given missed it
    trajectory = Trajectory()
    trajectory.observe(Observation(author=Author.USER, parts=(TextPart(text="Summarise q3.md."),)))
    call = ToolCall(id="call-1", name="read_file", arguments='{"path": "q3.md"}')
    step = Step(trajectory=trajectory, action=Action(tool_calls=(call,)))
    cascade = build_cascade_monitor(
        fast=KeywordMonitor(suspicion_by_keyword={"read_file": 0.3}),
        careful=KeywordMonitor(suspicion_by_keyword={"read_file": 0.9}),
        escalation_threshold=Decimal("0.3"),
    )

    # Act
    score = await rate_on_path(cascade, step, call_path=call_path)

    # Assert
    assert score.value == 0.9


def test_a_bracketed_name_builds_an_agent() -> None:
    # Arrange: LangGraph refuses only ":" and "|" in the node names the hooks become
    monitor = build_middleware(label="team[a]", agent_name="main")

    # Act
    agent = create_agent(model=build_trusted_model(), tools=[], middleware=[monitor])

    # Assert
    assert "team[a][main].before_model" in agent.get_graph().nodes


def test_monitor_subagents_reads_generators_of_subagents_and_skills_once() -> None:
    # Arrange
    pytest.importorskip("deepagents")

    # Act
    specs = build_monitored_subagents(subagents=iter([WORKER_SPEC]), skills=iter(["/skills/"]))

    # Assert
    assert [spec["name"] for spec in specs] == ["worker", "general-purpose"]
    assert specs[1]["skills"] == ["/skills/"]
    assert specs[0]["middleware"][-1].name == "monitor[worker]"


def test_the_placement_check_takes_a_tuple_as_it_takes_a_list() -> None:
    # Act / Assert
    assert check_monitor_placement(middleware=(build_middleware(),)) == []


@pytest.mark.parametrize(
    ("build", "options", "message"),
    [
        (
            build_repeated_monitor,
            {"monitor": LLMMonitor},
            r"got the class LLMMonitor; pass an instance\. A monitor of your own",
        ),
        (
            build_monitor_twins,
            {"monitor": hyoshigi.LLMMonitor},
            "; pass an instance. It is the twin",
        ),
        (build_repeated_monitor, {"ensemble_size": int}, "got the class int; pass an instance$"),
    ],
    ids=["class-for-monitor", "class-with-hint", "class-for-count"],
)
def test_a_class_given_where_an_instance_belongs_is_named_as_a_class(
    build: Build,
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build(**options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"view": "all"}, r"view must be a View, got an instance of str$"),
        (
            {"model": None},
            r"got NoneType\. Pass the chat model itself, not a Runnable wrapped around it, and "
            r"set its retries through its own max_retries\.$",
        ),
    ],
    ids=["no-hint", "hint"],
)
def test_a_refusal_ends_with_a_full_stop_only_after_a_hint(
    options: dict[str, object],
    message: str,
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match=message):
        build_llm_monitor(**options)


HUGE = 10**5000
"""An integer of 5,001 digits: by default Python refuses to write out one of over 4,300."""
HUGE_OPTIONS = {
    "flag-threshold": lambda: build_middleware(flag_threshold=HUGE),
    "threshold-fraction": lambda: build_middleware(flag_threshold=Fraction(HUGE, 3)),
    "enum": lambda: MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=RUN_EVERY_STEP,
        task_author=HUGE,  # ty: ignore[invalid-argument-type]
    ),
    "highest-score": lambda: LLMMonitor(
        model=FakeListChatModel(responses=["ok"]), highest_score=HUGE
    ),
    "lowest-score-above-highest": lambda: LLMMonitor(
        model=FakeListChatModel(responses=["ok"]), lowest_score=HUGE, highest_score=0
    ),
    "overrides-key": lambda: build_monitored_subagents(overrides={HUGE: build_middleware()}),
}
"""Options each refusal of which once wrote the value out, and raised `ValueError` doing so;
the scale's ends were written into the prompt only at the first step."""


@pytest.mark.parametrize("build", HUGE_OPTIONS.values(), ids=HUGE_OPTIONS.keys())
def test_an_option_too_long_to_write_out_is_refused_by_its_kind(
    build: Callable[[], object],
) -> None:
    # Act / Assert
    with pytest.raises(ConfigurationError, match="too long to write out"):
        build()


def test_a_number_too_long_to_write_out_is_named_by_its_kind() -> None:
    # Act
    described = [describe_option_value(HUGE), describe_option_value(Fraction(HUGE, 3))]

    # Assert
    assert described == [
        "an integer too long to write out",
        "a fractions.Fraction too long to write out",
    ]


FULL_REFUSALS: dict[str, tuple[Build, dict[str, object], str]] = {
    "lowest-score-too-long": (
        build_llm_monitor,
        {"lowest_score": -HUGE},
        "lowest_score must be an integer Python can write out, got an integer too long to "
        "write out",
    ),
    "highest-score-too-long": (
        build_llm_monitor,
        {"highest_score": HUGE},
        "highest_score must be an integer Python can write out, got an integer too long to "
        "write out",
    ),
    "score-not-an-integer": (
        build_llm_monitor,
        {"lowest_score": 2.5},
        "lowest_score must be an integer, got 2.5",
    ),
    "honest-scores-not-an-iterable": (
        build_calibrated_monitor,
        {"honest_scores": "0.1"},
        "honest_scores must be an iterable of numbers between 0 and 1, got '0.1'",
    ),
    "no-honest-scores": (
        build_calibrated_monitor,
        {"honest_scores": []},
        "honest_scores must hold at least one score",
    ),
    "both-reserved-characters": (
        build_middleware,
        {"label": "team:a|b"},
        "label must not contain ':' or '|', which LangGraph refuses in the names of the graph "
        "nodes the monitor's hooks become, got 'team:a|b'",
    ),
}


@pytest.mark.parametrize(
    ("build", "options", "message"),
    FULL_REFUSALS.values(),
    ids=FULL_REFUSALS.keys(),
)
def test_a_refusal_reads_in_full(
    build: Build,
    options: dict[str, object],
    message: str,
) -> None:
    # Act
    with pytest.raises(ConfigurationError) as refusal:
        build(**options)

    # Assert: in full, so that no hint, example or refused value can go missing unseen
    assert str(refusal.value) == message


@pytest.mark.parametrize(
    ("option_type", "text", "member"),
    [
        (Resampling, "sequential", Resampling.SEQUENTIAL),
        (Resampling, "parallel", Resampling.PARALLEL),
    ],
)
def test_a_string_from_configuration_converts_to_the_member_it_names(
    option_type: type[StrEnum],
    text: str,
    member: StrEnum,
) -> None:
    # Act: as each refusal of a plain string advises, such as with Aggregation(value)
    converted = option_type(text)

    # Assert
    assert converted is member


SHARED_PRESETS: dict[str, tuple[object, str]] = {
    "default-feedback": (DEFAULT_FEEDBACK_TEMPLATE, "text"),
}


@pytest.mark.parametrize(
    ("preset", "field_name"),
    SHARED_PRESETS.values(),
    ids=SHARED_PRESETS.keys(),
)
def test_a_preset_every_monitor_or_protocol_shares_cannot_be_changed_in_place(
    preset: object,
    field_name: str,
) -> None:
    # Act / Assert: one instance serves every monitor or protocol built with the default
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(preset, field_name, "changed")


@pytest.mark.parametrize(
    "protocol",
    [
        BlockAndFeedBack(block_threshold=0.5),
        DeferToResample(defer_threshold=0.5, audit_threshold=0.5, max_resamples=0),
    ],
    ids=["block-and-feed-back", "defer-at-the-audit-threshold"],
)
def test_a_trusted_model_the_protocol_never_asks_for_is_refused(
    protocol: BlockAndFeedBack | DeferToResample,
) -> None:
    # Arrange
    trusted_model = FakeListChatModel(responses=["I will only summarise the file."])

    # Act / Assert
    with pytest.raises(ConfigurationError, match="trusted_model"):
        MonitorMiddleware(monitor=KeywordMonitor(), protocol=protocol, trusted_model=trusted_model)


@pytest.mark.parametrize(
    "protocol",
    [
        DeferToResample(defer_threshold=0.5, audit_threshold=0.9, max_resamples=0),
        DeferToResample(
            defer_threshold=0.5,
            audit_threshold=0.5,
            max_resamples=0,
            on_unavailable=FailureResponse.TRUSTED_STEP,
        ),
    ],
    ids=["defer-below-the-audit-threshold", "a-trusted-step-when-no-answer"],
)
def test_a_trusted_model_the_protocol_may_ask_for_is_kept(protocol: DeferToResample) -> None:
    # Arrange
    trusted_model = FakeListChatModel(responses=["I will only summarise the file."])

    # Act
    middleware = MonitorMiddleware(
        monitor=KeywordMonitor(),
        protocol=protocol,
        trusted_model=trusted_model,
    )

    # Assert
    assert middleware.trusted_model is trusted_model
