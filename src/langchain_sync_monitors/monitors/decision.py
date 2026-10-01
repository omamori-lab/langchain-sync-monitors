"""Decision models as monitors: yes/no probabilities, no generated text.

A decision model such as TypeSafe's Jev answers typed questions about some
context with probabilities, in one fast request. TypeSafe describes Jev's
probabilities as calibrated (section Frontiers, Old and New)
[@typesafe2026jev]; the library has not measured their calibration on agent
steps, so set thresholds from honest runs, as for any monitor.
`DecisionModelMonitor` asks yes/no questions about the rendered step and
takes suspicion from the probabilities of yes. `OpenRouterDecisionModel`, in
`openrouter_decisions`, and `TypeSafeDecisionModel` reach Jev through
OpenRouter's Decisions API and through TypeSafe's own client. The evidence
behind the question format is in
`docs/plans/initial-implementation/research/decision-model-question-format.md`.
"""

from __future__ import annotations

import importlib
import numbers
import statistics
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from types import ModuleType
from typing import TYPE_CHECKING

from langchain_core.runnables import Runnable

from langchain_sync_monitors.contracts import Monitor, MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.errors import MissingExtraError, MonitorError
from langchain_sync_monitors.model_calls import build_internal_call_config
from langchain_sync_monitors.monitors.chat import DEFAULT_MONITOR_VIEW
from langchain_sync_monitors.monitors.decision_questions import (
    IS_THE_STEP_SUSPICIOUS,
    YesNoQuestion,
    read_questions,
)
from langchain_sync_monitors.options import check_enum_option, check_instance_option
from langchain_sync_monitors.transcript import render_proposed_step, render_transcript

if TYPE_CHECKING:
    from langchain_typesafe import ClassifierRequest, TypeSafeClassifier


class DecisionModel(ABC):
    """A model that answers yes/no questions about a context with probabilities.

    It generates no text. Every question goes in one request, and the answer
    maps each question's key to the probability of yes.
    """

    @abstractmethod
    async def estimate_probabilities(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Return the probability of yes for each question, keyed by question key."""

    @abstractmethod
    def estimate_probabilities_sync(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Return the probability of yes for each question, without an event loop."""


def select_question_probabilities(
    probabilities: Mapping[str, float],
    *,
    questions: Sequence[YesNoQuestion],
) -> dict[str, float]:
    """Keep one probability per question; fail if the model skipped one or gave no probability.

    An answer that is not a finite number from 0 to 1 raises `MonitorError`,
    as a skipped question does: the decision model gave no readable answer.
    A number is an `int`, a `float`, a `Decimal` or another real number;
    `True` and `False`, which Python counts as numbers, are no probability.
    That also covers NaN, which `max` and `min` would drop or keep depending
    on its position, so that the step could score low, and `None` and
    strings, which cannot be compared. Answers come back as floats.
    """
    missing = [question.key for question in questions if question.key not in probabilities]
    if missing:
        message = f"the decision model returned no answer for {missing}"
        raise MonitorError(message)
    selected = {question.key: probabilities[question.key] for question in questions}
    unreadable = sorted(key for key, value in selected.items() if not is_probability(value))
    if unreadable:
        message = f"the decision model returned no probability from 0 to 1 for {unreadable}"
        raise MonitorError(message)
    return {key: float(value) for key, value in selected.items()}


def is_probability(value: object) -> bool:
    """Tell whether a value is a finite number from 0 to 1; a bool is not, and neither is NaN.

    A `Decimal` is no `numbers.Real`, so it is checked on its own; its
    signalling NaN cannot even be converted to a float.
    """
    if isinstance(value, Decimal):
        return value.is_finite() and 0 <= value <= 1
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    # Compared exactly, not as a float: `float()` raises `OverflowError` on an `int` or `Fraction`
    # as large as 10**400, and rounds a `Fraction` just outside 0 to 1 into it. NaN fails `<= 1`.
    return bool(not value < 0 and value <= 1)


def load_typesafe_module() -> ModuleType:
    """Import the optional langchain-typesafe package, or say how to install it."""
    try:
        return importlib.import_module("langchain_typesafe")
    except ImportError as error:
        message = (
            "TypeSafeDecisionModel needs the typesafe extra: "
            "pip install 'langchain-sync-monitors[typesafe]'"
        )
        raise MissingExtraError(message) from error


class TypeSafeDecisionModel(DecisionModel):
    """Jev through TypeSafe's own API, using a `TypeSafeClassifier` you configure.

    Each question becomes a `Noul` with `NoulCriteria`, and all of them go in
    one classifier call [@typesafe2026langchain]. The classifier, with its
    key, model and HTTP clients, is yours. Needs the `typesafe` extra.
    """

    def __init__(self, *, classifier: TypeSafeClassifier) -> None:
        """Wrap `classifier`; fails with an install hint when the extra is missing.

        `classifier` must be a `Runnable`, such as the `TypeSafeClassifier`
        itself or one wrapped by `with_retry()`; anything else raises
        `ConfigurationError`.
        """
        self.typesafe = load_typesafe_module()
        check_instance_option(
            classifier,
            option_type=Runnable,
            parameter_name="classifier",
            hint="Pass a TypeSafeClassifier from langchain-typesafe.",
        )
        self.classifier = classifier
        self.call_config = build_internal_call_config(source="decision_model_monitor")

    async def estimate_probabilities(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Ask every question in one classifier call and return the probabilities of yes."""
        request = self.build_request(context=context, questions=questions)
        response = await self.classifier.ainvoke(request, config=self.call_config)
        probabilities = {key: answer.noul for key, answer in response.nouls.items()}
        return select_question_probabilities(probabilities, questions=questions)

    def estimate_probabilities_sync(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Ask every question in one classifier call, without an event loop."""
        request = self.build_request(context=context, questions=questions)
        response = self.classifier.invoke(request, config=self.call_config)
        probabilities = {key: answer.noul for key, answer in response.nouls.items()}
        return select_question_probabilities(probabilities, questions=questions)

    def build_request(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> ClassifierRequest:
        """Build the classifier request: the context as the state, one `Noul` per question."""
        noul, criteria = self.typesafe.Noul, self.typesafe.NoulCriteria
        return {
            "state": context,
            "questions": {
                question.key: noul(
                    instructions=question.text,
                    criteria=criteria(true=question.yes_means, false=question.no_means),
                )
                for question in questions
            },
        }


class Aggregation(StrEnum):
    """How `DecisionModelMonitor` aggregates the probabilities of several questions.

    `MAX` lets any single criterion raise suspicion, the right shape for threat
    criteria, and the way ShieldGemma turns its harm types into one binary
    outcome when it evaluates them (section 5.1) [@zeng2024shieldgemma]. `MIN`
    fits questions that must all hold. `MEAN` dilutes one strong hit: 0.95
    with three 0.05s averages 0.275.

    `MAX` takes the maximum of raw probabilities. Control evaluations
    aggregate on the honest scale instead: LinuxArena turns each monitor's
    score into a percentile rank against its honest distribution and then
    takes the maximum (appendix C), and AI Control maps each score through the
    empirical distribution of honest scores before aggregating (appendix K)
    [@tracy2026linuxarena; @greenblatt2023aicontrol]. On raw probabilities, a
    question that honest steps often answer yes to can dominate the maximum.
    Wrapping the monitor in `CalibratedMonitor` makes a threshold flag a known
    share of honest steps, but does not remove that dominance.
    """

    MAX = "max"
    MIN = "min"
    MEAN = "mean"


def aggregate_probabilities(probabilities: Sequence[float], *, aggregation: Aggregation) -> float:
    """Aggregate the probabilities of yes into one suspicion."""
    if aggregation is Aggregation.MAX:
        return max(probabilities)
    if aggregation is Aggregation.MIN:
        return min(probabilities)
    return statistics.fmean(probabilities)


class DecisionModelMonitor(Monitor):
    """Suspicion from a decision model's probability of yes to questions about the step.

    The model reads the rendered transcript and the proposed step as one
    string. With the default single question, suspicion is its probability of
    yes. With several, such as `ATTACK_CRITERIA`, they go in one request and
    `aggregation` merges them; `Aggregation.MAX` is the default because
    averaging dilutes a single strong hit, while the maximum keeps it (see
    `docs/plans/initial-implementation/research/decision-model-question-format.md`).
    The reason names the question with the highest probability of yes, or,
    under `Aggregation.MIN`, every question in the order given: the minimum
    rests on all of them, and naming the lowest would point the agent at the
    one criterion it needs to lower to pass. It carries no probability or
    aggregated score: Auto Mode shows the reason to the agent, and those numbers
    would tell it how close its step came to passing. They stay in the
    suspicion.

    Every answer is checked, whichever `DecisionModel` gave it: a skipped
    question, or an answer that is not a finite number from 0 to 1 (an
    `int`, a `float`, a `Decimal` or another real number, but not a `bool`),
    raises `MonitorError`, so the step fails, and none of the agent's own
    tools run.
    """

    def __init__(
        self,
        *,
        decision_model: DecisionModel,
        questions: Iterable[YesNoQuestion] = (IS_THE_STEP_SUSPICIOUS,),
        aggregation: Aggregation = Aggregation.MAX,
        view: MonitorView = DEFAULT_MONITOR_VIEW,
    ) -> None:
        """Build the monitor; question keys must be unique, and `Aggregation.MEAN` warns.

        `questions` may be any iterable, a generator included; it is read once
        into a tuple, so checking it cannot consume the questions. An option
        of the wrong type raises `ConfigurationError`.
        """
        check_instance_option(
            decision_model,
            option_type=DecisionModel,
            parameter_name="decision_model",
            hint="Use OpenRouterDecisionModel(model=...) or TypeSafeDecisionModel(classifier=...).",
        )
        check_enum_option(aggregation, option_type=Aggregation, parameter_name="aggregation")
        check_instance_option(view, option_type=MonitorView, parameter_name="view")
        question_tuple = read_questions(questions)
        if aggregation is Aggregation.MEAN:
            message = (
                "Aggregation.MEAN dilutes a single strong hit (0.95 with three 0.05s averages "
                "0.275); Aggregation.MAX keeps it."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
        self.decision_model = decision_model
        self.questions = question_tuple
        self.aggregation = aggregation
        self.view = view

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Ask the decision model every question in one request."""
        probabilities = await self.decision_model.estimate_probabilities(
            context=self.render_context(monitor_input),
            questions=self.questions,
        )
        return self.build_verdict(probabilities)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Ask the decision model every question in one request, without an event loop."""
        probabilities = self.decision_model.estimate_probabilities_sync(
            context=self.render_context(monitor_input),
            questions=self.questions,
        )
        return self.build_verdict(probabilities)

    def render_context(self, monitor_input: MonitorInput) -> str:
        """Render the transcript and the proposed step as the decision model's context."""
        transcript = render_transcript(
            monitor_input.history,
            view=self.view,
            task_author=monitor_input.task_author,
        )
        proposed_step = render_proposed_step(monitor_input.proposal, view=self.view)
        return f"<transcript>\n{transcript}\n</transcript>\n{proposed_step}"

    def build_verdict(self, probabilities: Mapping[str, float]) -> Verdict:
        """Check and aggregate the probabilities, and name the questions the suspicion rests on.

        Under `Aggregation.MIN` that is every question, in the order given, so
        the reason does not depend on the answers. Otherwise it is the question
        with the highest probability of yes, the first in order on a tie.
        """
        answers = select_question_probabilities(probabilities, questions=self.questions)
        suspicion = aggregate_probabilities(list(answers.values()), aggregation=self.aggregation)
        if self.aggregation is Aggregation.MIN:
            concerns = "; ".join(
                f"{question.key!r}: {question.text}" for question in self.questions
            )
            reason = f"The decision model's score rests on all of its concerns together: {concerns}"
        else:
            strongest = max(self.questions, key=lambda question: answers[question.key])
            reason = (
                f"The decision model's strongest concern is {strongest.key!r}: {strongest.text}"
            )
        return Verdict(suspicion=suspicion, reason=reason)
