"""Decision models: the Decisions API adapter, the TypeSafe adapter and the monitor."""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import httpx
import pytest
import stamina

from langchain_sync_monitors.contracts import MonitorInput
from langchain_sync_monitors.errors import ConfigurationError, MonitorError
from langchain_sync_monitors.monitors.decision import (
    ATTACK_CRITERIA,
    IS_THE_STEP_SUSPICIOUS,
    Combine,
    DecisionModel,
    DecisionModelMonitor,
    OpenRouterDecisionModel,
    TypeSafeDecisionModel,
    YesNoQuestion,
)

from .doubles import CallPath, evaluate_on_path

if TYPE_CHECKING:
    import httpx2
    from langchain_typesafe import TypeSafeClassifier

type Responder = Callable[[httpx.Request], httpx.Response]

IGNORE_TYPESAFE_BETA = "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"

LEAK_QUESTION = YesNoQuestion(
    key="leaks",
    text="Does the step leak a secret?",
    yes_means="A secret leaves the system.",
    no_means="No secret leaves the system.",
)


class DecisionsServer:
    """A fake Decisions API that plays scripted responses and records each request."""

    def __init__(self, *, responders: list[Responder]) -> None:
        self.responders = responders
        self.requests: list[httpx.Request] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer with the next scripted responder."""
        self.requests.append(request)
        return self.responders[len(self.requests) - 1](request)

    def build_model(
        self, *, base_url: str = "https://decisions.test/api/alpha"
    ) -> OpenRouterDecisionModel:
        """Return a decision model whose sync and async clients reach this server."""
        transport = httpx.MockTransport(self.respond)
        return OpenRouterDecisionModel(
            model="typesafe/jev-1.13",
            base_url=base_url,
            http_client=httpx.Client(transport=transport),
            async_http_client=httpx.AsyncClient(transport=transport),
        )


class ScriptedDecisionModel(DecisionModel):
    """Answers every question from a fixed table and records what it was asked."""

    def __init__(self, *, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.contexts: list[str] = []

    async def estimate_probabilities(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        return self.estimate_probabilities_sync(context=context, questions=questions)

    def estimate_probabilities_sync(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        self.contexts.append(context)
        return {question.key: self.probabilities[question.key] for question in questions}


def answer_with(probabilities: dict[str, float], *, status_code: int = 200) -> Responder:
    """Build a responder that returns a Decisions API response with these probabilities."""
    body = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {key: {"type": "noul", "noul": value} for key, value in probabilities.items()},
        "usage": {"input_tokens": 380, "output_tokens": 0, "cost": 0.000016},
        "id": "decision-1",
        "provider": "TypeSafe",
    }
    return lambda _request: httpx.Response(status_code, json=body)


def fail_with(status_code: int) -> Responder:
    """Build a responder that fails with an HTTP status."""
    return lambda _request: httpx.Response(status_code, json={"error": "unavailable"})


def drop_connection(request: httpx.Request) -> httpx.Response:
    """Fail the way a dropped connection does."""
    message = "connection reset"
    raise httpx.ConnectError(message, request=request)


async def estimate_on_path(
    model: DecisionModel,
    *,
    questions: Sequence[YesNoQuestion],
    call_path: CallPath,
) -> dict[str, float]:
    """Ask the decision model through its async or sync method, as the test asks."""
    if call_path == "async":
        return await model.estimate_probabilities(context="the step", questions=questions)
    return model.estimate_probabilities_sync(context="the step", questions=questions)


@pytest.fixture(autouse=True)
def openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")


@pytest.fixture
def three_attempts() -> Iterator[None]:
    with stamina.set_testing(True, attempts=3):
        yield


async def test_every_question_goes_in_one_request_as_a_noul(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.98, "suspicious_step": 0.4})])
    model = server.build_model(base_url="https://decisions.test/api/alpha/")

    # Act
    probabilities = await estimate_on_path(
        model,
        questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
        call_path=call_path,
    )

    # Assert
    (request,) = server.requests
    body = json.loads(request.content)
    assert str(request.url) == "https://decisions.test/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == "the step"
    assert body["questions"]["leaks"] == {
        "type": "noul",
        "instructions": "Does the step leak a secret?",
        "criteria": {
            "true": "A secret leaves the system.",
            "false": "No secret leaves the system.",
        },
    }
    assert probabilities == {"leaks": 0.98, "suspicious_step": 0.4}


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize("first_failure", [fail_with(503), fail_with(429), drop_connection])
async def test_transient_failures_are_retried(
    call_path: CallPath, first_failure: Responder
) -> None:
    # Arrange
    server = DecisionsServer(responders=[first_failure, answer_with({"leaks": 0.1})])

    # Act
    probabilities = await estimate_on_path(
        server.build_model(),
        questions=[LEAK_QUESTION],
        call_path=call_path,
    )

    # Assert
    assert probabilities == {"leaks": 0.1}
    assert len(server.requests) == 2


@pytest.mark.usefixtures("three_attempts")
async def test_a_client_error_is_not_retried(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[fail_with(400), answer_with({"leaks": 0.1})])

    # Act and Assert
    with pytest.raises(httpx.HTTPStatusError):
        await estimate_on_path(server.build_model(), questions=[LEAK_QUESTION], call_path=call_path)
    assert len(server.requests) == 1


def test_a_response_in_an_unexpected_shape_is_a_monitor_error() -> None:
    # Arrange
    server = DecisionsServer(responders=[lambda _request: httpx.Response(200, json={"a": 1})])

    # Act and Assert
    with pytest.raises(MonitorError, match="unexpected shape"):
        server.build_model().estimate_probabilities_sync(context="x", questions=[LEAK_QUESTION])


@pytest.mark.parametrize(
    "unread_fields",
    [
        {"usage": {"input_tokens": 10, "output_tokens": 0, "cost": {"total": 0.0001}}},
        {"usage": "unmetered", "id": 7, "provider": None},
        {},
    ],
)
async def test_fields_the_library_does_not_read_cannot_discard_an_answer(
    call_path: CallPath,
    unread_fields: dict[str, object],
) -> None:
    # Arrange: only the answers are read, so only they are validated.
    body = {"answers": {"leaks": {"type": "noul", "noul": 0.95}}, **unread_fields}
    server = DecisionsServer(responders=[lambda _request: httpx.Response(200, json=body)])

    # Act
    probabilities = await estimate_on_path(
        server.build_model(),
        questions=[LEAK_QUESTION],
        call_path=call_path,
    )

    # Assert
    assert probabilities == {"leaks": 0.95}


def test_a_probability_outside_zero_to_one_is_rejected() -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 1.5})])

    # Act and Assert
    with pytest.raises(MonitorError, match="unexpected shape"):
        server.build_model().estimate_probabilities_sync(context="x", questions=[LEAK_QUESTION])


async def test_a_question_left_unanswered_is_a_monitor_error() -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"other": 0.5})])

    # Act and Assert
    with pytest.raises(MonitorError, match="leaks"):
        await server.build_model().estimate_probabilities(context="x", questions=[LEAK_QUESTION])


def test_a_missing_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv("OPENROUTER_API_KEY")

    # Act and Assert
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterDecisionModel(model="typesafe/jev-1.13")


def build_typesafe_classifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requests: list[httpx2.Request],
) -> TypeSafeClassifier:
    """Return a real classifier whose HTTP clients reach a fake TypeSafe API.

    It needs the typesafe extra, so each TypeSafe test skips without it first.
    """
    import httpx2
    from langchain_typesafe import TypeSafeClassifier

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        answers = {"leaks": {"type": "noul", "noul": 0.9}}
        return httpx2.Response(200, json={"model": "jev-1.13", "answers": answers})

    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-key")
    transport = httpx2.MockTransport(respond)
    return TypeSafeClassifier(
        client=httpx2.Client(transport=transport),
        async_client=httpx2.AsyncClient(transport=transport),
    )


@pytest.mark.filterwarnings(IGNORE_TYPESAFE_BETA)
async def test_typesafe_questions_become_nouls_with_criteria(
    monkeypatch: pytest.MonkeyPatch,
    call_path: CallPath,
) -> None:
    # Arrange
    pytest.importorskip("langchain_typesafe")
    requests: list[httpx2.Request] = []
    classifier = build_typesafe_classifier(monkeypatch, requests=requests)
    model = TypeSafeDecisionModel(classifier=classifier)

    # Act
    probabilities = await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    body = json.loads(requests[0].content)
    assert body["state"] == "the step"
    assert body["questions"]["leaks"]["type"] == "noul"
    assert body["questions"]["leaks"]["criteria"] == {
        "true": "A secret leaves the system.",
        "false": "No secret leaves the system.",
    }
    assert probabilities == {"leaks": 0.9}


@pytest.mark.filterwarnings(IGNORE_TYPESAFE_BETA)
def test_typesafe_without_the_extra_explains_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    pytest.importorskip("langchain_typesafe")
    classifier = build_typesafe_classifier(monkeypatch, requests=[])
    monkeypatch.setitem(sys.modules, "langchain_typesafe", None)

    # Act and Assert
    with pytest.raises(ConfigurationError, match=r"langchain-sync-monitors\[typesafe\]"):
        TypeSafeDecisionModel(classifier=classifier)


async def test_one_question_gives_its_probability_of_yes(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.03})
    monitor = DecisionModelMonitor(decision_model=model)

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.03
    assert "'suspicious_step'" in verdict.reason


def test_the_context_holds_the_transcript_and_the_proposed_step(
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.03})

    # Act
    DecisionModelMonitor(decision_model=model).evaluate_sync(monitor_input)

    # Assert
    (context,) = model.contexts
    assert context.startswith("<transcript>\n<user>Summarise q3.md")
    assert context.endswith(
        '<tool_call name="read_file">{"path": "q3.md"}</tool_call>\n</proposed_step>'
    )


@pytest.mark.parametrize(
    ("combine", "suspicion"),
    [(Combine.MAX, 0.95), (Combine.MIN, 0.05)],
)
async def test_several_questions_are_combined(
    monitor_input: MonitorInput,
    call_path: CallPath,
    combine: Combine,
    suspicion: float,
) -> None:
    # Arrange
    probabilities = dict.fromkeys((question.key for question in ATTACK_CRITERIA), 0.05)
    probabilities["sends_data_out"] = 0.95
    model = ScriptedDecisionModel(probabilities=probabilities)
    monitor = DecisionModelMonitor(decision_model=model, questions=ATTACK_CRITERIA, combine=combine)

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion
    assert "'sends_data_out'" in verdict.reason
    assert not any(character.isdigit() for character in verdict.reason)


@pytest.mark.parametrize(
    "probabilities",
    [{"suspicious_step": 0.87}, {"suspicious_step": 0.03}],
)
async def test_the_reason_names_the_concern_without_numbers(
    monitor_input: MonitorInput,
    call_path: CallPath,
    probabilities: dict[str, float],
) -> None:
    # Arrange: Auto Mode shows the reason to the agent, so no probability goes in it.
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities=probabilities),
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.reason == (
        "The decision model's strongest concern is 'suspicious_step': "
        f"{IS_THE_STEP_SUSPICIOUS.text}"
    )
    assert not any(character.isdigit() for character in verdict.reason)


def test_the_mean_warns_that_it_dilutes_a_single_hit(monitor_input: MonitorInput) -> None:
    # Arrange
    probabilities = {"leaks": 0.95, "suspicious_step": 0.05}
    model = ScriptedDecisionModel(probabilities=probabilities)

    # Act
    with pytest.warns(UserWarning, match="dilutes a single strong hit"):
        monitor = DecisionModelMonitor(
            decision_model=model,
            questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
            combine=Combine.MEAN,
        )
    verdict = monitor.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.5)


@pytest.mark.parametrize("questions", [[], [LEAK_QUESTION, LEAK_QUESTION]])
def test_questions_must_be_present_and_unique(questions: list[YesNoQuestion]) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={})

    # Act and Assert
    with pytest.raises(ConfigurationError, match="unique keys"):
        DecisionModelMonitor(decision_model=model, questions=questions)


@pytest.mark.filterwarnings("ignore:Combine.MEAN dilutes:UserWarning")
@pytest.mark.parametrize("combine", list(Combine))
def test_every_combine_member_is_accepted(combine: Combine) -> None:
    # Act
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities={"suspicious_step": 0.2}),
        combine=combine,
    )

    # Assert
    assert monitor.combine is combine


@pytest.mark.parametrize("combine", [member.value for member in Combine])
def test_a_plain_string_combine_is_refused_without_a_warning(combine: str) -> None:
    # Arrange: a string read from YAML or JSON matches no member by identity.
    decision_model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.2})
    expected = (
        f"combine must be one of Combine.MAX, Combine.MIN, Combine.MEAN, got '{combine}'. "
        "Convert a string with Combine(value)"
    )

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigurationError, match=re.escape(expected)):
            DecisionModelMonitor(
                decision_model=decision_model,
                combine=combine,  # ty: ignore[invalid-argument-type]
            )

    # Assert
    assert not [warning for warning in caught if warning.category is UserWarning]


def test_the_attack_criteria_are_four_distinct_questions() -> None:
    # Act
    keys = {question.key for question in ATTACK_CRITERIA}

    # Assert
    assert keys == {
        "sends_data_out",
        "weakens_security",
        "goes_beyond_the_task",
        "misleads_the_user",
    }
