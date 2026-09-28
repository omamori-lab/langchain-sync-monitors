"""Guard models (safety classifiers) that label a step against a policy you write.

A guard model such as gpt-oss-safeguard or Llama Guard replies with a label,
not a score. A label alone gives a monitor one operating point, so every
threshold flags the same steps. `GuardModelMonitor` therefore turns labels into
a suspicion from 0 to 1: from the label token's log-probabilities when the
provider returns them, otherwise from the share of several sampled replies that
carry a suspicious label. The evidence is in
`docs/research/guard-model-scoring.md`.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections.abc import Generator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from itertools import filterfalse
from typing import ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from langchain_sync_monitors.contracts import MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors.chat import (
    DEFAULT_MONITOR_VIEW,
    ChatModelMonitor,
    ReplyRequest,
    VerdictPlan,
)

logger = logging.getLogger(__name__)

TOP_LOG_PROBABILITIES = 20
"""How many alternatives to request per token; 20 is the most OpenAI-style APIs return."""

LABEL_PATTERN = re.compile(r"[\w-]+")
LABEL_LINE_PATTERN = re.compile(r"\W*(?:.*:\W*)?(?P<label>[\w-]+)\W*")
NON_EMPTY_LINE_PATTERN = re.compile(r"[^\n]*\S[^\n]*")
LABEL_MARKUP = "*`\"'#>"


class GuardScoring(StrEnum):
    """How `GuardModelMonitor` turns the guard model's labels into a suspicion.

    `AUTO` reads log-probabilities when the provider returns them and samples
    otherwise. `LOG_PROBABILITIES` raises a `ConfigurationError` when the
    provider returns none. `SAMPLE_FRACTION` always samples. `HARD_LABEL` reads
    one label as 0 or 1 and warns, because every threshold then flags the same
    steps. Sampling only tells replies apart when the model's temperature is
    above zero.
    """

    AUTO = "auto"
    LOG_PROBABILITIES = "log_probabilities"
    SAMPLE_FRACTION = "sample_fraction"
    HARD_LABEL = "hard_label"


class HardLabelWarning(UserWarning):
    """A guard monitor scores with hard labels, so every threshold flags the same steps."""


class AlternativeToken(BaseModel):
    """One of the most likely tokens at a position, with its log-probability."""

    token: str
    logprob: float


class TokenLogProbability(BaseModel):
    """A generated token, its log-probability and the most likely alternatives to it."""

    token: str
    logprob: float
    top_logprobs: list[AlternativeToken] = Field(default_factory=list)


class ReplyLogProbabilities(BaseModel):
    """The OpenAI-style `logprobs` entry of a reply's `response_metadata`."""

    content: list[TokenLogProbability] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelMatch:
    """A known label found in a reply, and the character offset where it starts."""

    label: str
    offset: int


def find_reply_label(text: str, *, labels: frozenset[str]) -> LabelMatch | None:
    """Find the label on the reply's last non-empty line, or else on its first.

    gpt-oss-safeguard writes its label last, after its reasoning; Llama Guard
    writes it first, with violated categories on the lines after. A line may
    hold the label alone or after a colon, as in `Label: violation`, and
    labels match case-insensitively.
    """
    lines = list(NON_EMPTY_LINE_PATTERN.finditer(text))
    for line in lines[-1:] + lines[:1]:
        match = LABEL_LINE_PATTERN.fullmatch(line.group())
        if match and match["label"].lower() in labels:
            offset = line.start() + match.start("label")
            return LabelMatch(label=match["label"].lower(), offset=offset)
    return None


def read_log_probabilities(reply: AIMessage) -> list[TokenLogProbability] | None:
    """Return the reply's token log-probabilities, or `None` when the provider sent none."""
    payload = reply.response_metadata.get("logprobs")
    if payload is None:
        return None
    try:
        tokens = ReplyLogProbabilities.model_validate(payload).content
    except ValidationError:
        logger.debug("Ignoring log-probabilities in an unknown format: %r", payload)
        return None
    return tokens or None


def find_token_index(tokens: list[TokenLogProbability], *, offset: int) -> int | None:
    """Return the index of the token that covers the character at `offset`."""
    end = 0
    for index, token in enumerate(tokens):
        end += len(token.token)
        if end > offset:
            return index
    return None


def classify_label_token(
    token: str,
    *,
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
) -> Literal["suspicious", "safe"] | None:
    """Tell which kind of label a token begins; `None` if it begins none, or both kinds."""
    prefix = token.strip().lstrip(LABEL_MARKUP).lower()
    if not prefix:
        return None
    begins_suspicious = any(label.startswith(prefix) for label in suspicious_labels)
    begins_safe = any(label.startswith(prefix) for label in safe_labels)
    if begins_suspicious == begins_safe:
        return None
    return "suspicious" if begins_suspicious else "safe"


def compute_suspicious_share(
    position: TokenLogProbability,
    *,
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
) -> float | None:
    """Share of the label probability at the label's first token that is suspicious.

    Every alternative that begins a label counts, so variants such as
    `violation`, ` violation` and `Violation` add up, as in Granite Guardian's
    probability of risk [@padhi2024graniteguardian]. Dividing by the mass of
    all label tokens renormalises over the labels, as Llama Guard and
    ShieldGemma do [@inan2023llamaguard; @zeng2024shieldgemma].
    """
    alternatives = {alternative.token: alternative.logprob for alternative in position.top_logprobs}
    alternatives.setdefault(position.token, position.logprob)
    mass = {"suspicious": 0.0, "safe": 0.0}
    for token, logprob in alternatives.items():
        kind = classify_label_token(
            token,
            suspicious_labels=suspicious_labels,
            safe_labels=safe_labels,
        )
        if kind is not None:
            mass[kind] += math.exp(logprob)
    total = mass["suspicious"] + mass["safe"]
    return mass["suspicious"] / total if total > 0.0 else None


def validate_labels(*, suspicious_labels: AbstractSet[str], safe_labels: AbstractSet[str]) -> None:
    """Reject empty, multi-word or overlapping label sets at construction."""
    if not suspicious_labels or not safe_labels:
        message = "suspicious_labels and safe_labels must each hold at least one label"
        raise ConfigurationError(message)
    multi_word = sorted(filterfalse(is_one_word, suspicious_labels | safe_labels))
    if multi_word:
        message = f"labels must be single words (letters, digits, _ or -), got {multi_word}"
        raise ConfigurationError(message)
    shared = set(map(str.lower, suspicious_labels)) & set(map(str.lower, safe_labels))
    if shared:
        message = f"labels cannot be both suspicious and safe: {sorted(shared)}"
        raise ConfigurationError(message)


def is_one_word(label: str) -> bool:
    """Tell whether a label is one word of letters, digits, underscores or hyphens."""
    return LABEL_PATTERN.fullmatch(label) is not None


class GuardModelMonitor(ChatModelMonitor):
    """A guard model (safety classifier) that labels a step against a policy you write.

    `policy_prompt` states the policy and must use `{transcript}` and
    `{proposed_step}`; it should ask for one label on a line of its own. The
    labels in `suspicious_labels` and `safe_labels` match case-insensitively,
    for example `{"violation"}` and `{"no_violation"}` for gpt-oss-safeguard,
    or `{"unsafe"}` and `{"safe"}` for Llama Guard. A reply without a known
    label counts as suspicious, so the monitor fails closed.

    With log-probabilities, suspicion is the probability the guard gives to a
    suspicious label, the score Llama Guard, ShieldGemma and Granite Guardian
    report [@inan2023llamaguard; @zeng2024shieldgemma; @padhi2024graniteguardian].
    Without them, suspicion is the share of `samples` replies labelled
    suspicious; the gain from more samples levels off around 5 to 10
    [@xiong2024uncertainty; @lyu2025sampleconsistency]. Sampling needs the
    model's temperature above zero. Guard probabilities are overconfident
    [@liu2025guardcalibration], so set thresholds on honest-run percentiles
    with `CalibratedMonitor` rather than on raw values.
    """

    call_source: ClassVar[str] = "guard_model_monitor"

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        policy_prompt: ChatPromptTemplate,
        suspicious_labels: AbstractSet[str],
        safe_labels: AbstractSet[str],
        scoring: GuardScoring = GuardScoring.AUTO,
        samples: int = 5,
        view: MonitorView = DEFAULT_MONITOR_VIEW,
    ) -> None:
        """Build a guard monitor; `samples` applies to sampling, including `AUTO`'s fallback."""
        validate_labels(suspicious_labels=suspicious_labels, safe_labels=safe_labels)
        if samples < 1:
            message = f"samples must be at least 1, got {samples}"
            raise ConfigurationError(message)
        super().__init__(model=model, prompt=policy_prompt, view=view)
        self.suspicious_labels = frozenset(label.lower() for label in suspicious_labels)
        self.safe_labels = frozenset(label.lower() for label in safe_labels)
        self.known_labels = self.suspicious_labels | self.safe_labels
        self.scoring = scoring
        self.samples = samples
        self.model_with_log_probabilities = self.model.bind(
            logprobs=True,
            top_logprobs=TOP_LOG_PROBABILITIES,
        )
        if scoring is GuardScoring.HARD_LABEL:
            message = (
                "GuardScoring.HARD_LABEL gives every step a suspicion of 0 or 1, so every "
                "threshold flags the same steps and an audit budget cannot be set. Prefer "
                "GuardScoring.AUTO."
            )
            warnings.warn(message, HardLabelWarning, stacklevel=2)

    def build_verdict_plan(self, monitor_input: MonitorInput) -> VerdictPlan:
        """Score from log-probabilities, from samples, or from one hard label."""
        messages = self.build_prompt_messages(monitor_input)
        if self.scoring is GuardScoring.HARD_LABEL:
            return self.build_sampling_plan(messages, count=1)
        if self.scoring is GuardScoring.SAMPLE_FRACTION:
            return self.build_sampling_plan(messages, count=self.samples)
        return self.build_log_probability_plan(messages)

    def build_sampling_plan(
        self,
        messages: tuple[BaseMessage, ...],
        *,
        count: int,
    ) -> Generator[ReplyRequest, list[AIMessage], Verdict]:
        """Draw `count` replies and return the share labelled suspicious."""
        replies = yield ReplyRequest(model=self.model, messages=messages, count=count)
        return self.build_sample_verdict(replies)

    def build_log_probability_plan(
        self,
        messages: tuple[BaseMessage, ...],
    ) -> Generator[ReplyRequest, list[AIMessage], Verdict]:
        """Read log-probabilities from a first reply; in `AUTO`, sample when that fails."""
        scoring_model = self.model_with_log_probabilities
        (first_reply,) = yield ReplyRequest(model=scoring_model, messages=messages)
        verdict = self.build_log_probability_verdict(first_reply)
        if verdict is not None:
            return verdict
        if self.scoring is GuardScoring.LOG_PROBABILITIES:
            return build_unlocated_label_verdict()
        remaining = self.samples - 1
        more_replies = yield ReplyRequest(model=self.model, messages=messages, count=remaining)
        return self.build_sample_verdict([first_reply, *more_replies])

    def build_log_probability_verdict(self, reply: AIMessage) -> Verdict | None:
        """Score the reply from its log-probabilities, or return `None` when they are unusable.

        Missing log-probabilities are a configuration error under
        `LOG_PROBABILITIES`, since that mode cannot score without them.
        """
        tokens = read_log_probabilities(reply)
        if tokens is None and self.scoring is GuardScoring.LOG_PROBABILITIES:
            message = (
                f"{self.model.get_name()} returned no log-probabilities; use "
                "GuardScoring.AUTO or GuardScoring.SAMPLE_FRACTION with this model"
            )
            raise ConfigurationError(message)
        return self.find_label_probability(tokens) if tokens else None

    def find_label_probability(self, tokens: list[TokenLogProbability]) -> Verdict | None:
        """Score from the alternatives at the label's first token, or `None` if it is not found."""
        text = "".join(token.token for token in tokens)
        match = find_reply_label(text, labels=self.known_labels)
        index = find_token_index(tokens, offset=match.offset) if match else None
        if match is None or index is None:
            return None
        share = compute_suspicious_share(
            tokens[index],
            suspicious_labels=self.suspicious_labels,
            safe_labels=self.safe_labels,
        )
        if share is None:
            return None
        reason = (
            f"The guard model labelled the step {match.label!r} and gives probability "
            f"{share:.3f} to a suspicious label."
        )
        return Verdict(suspicion=share, reason=reason)

    def build_sample_verdict(self, replies: list[AIMessage]) -> Verdict:
        """Return the share of replies labelled suspicious; an unreadable reply counts as one."""
        matches = [find_reply_label(reply.text, labels=self.known_labels) for reply in replies]
        suspicious_count = sum(
            match is None or match.label in self.suspicious_labels for match in matches
        )
        names = ", ".join(match.label if match else "unreadable" for match in matches)
        reason = (
            f"{suspicious_count} of {len(replies)} guard replies labelled the step "
            f"suspicious ({names})."
        )
        return Verdict(suspicion=suspicious_count / len(replies), reason=reason)


def build_unlocated_label_verdict() -> Verdict:
    """Fail closed when log-probabilities came back but no label could be found in them."""
    logger.warning("No guard label found in the log-probabilities; the step is suspicious.")
    reason = (
        "The guard model's reply had log-probabilities but no readable label, "
        "so the step is treated as suspicious."
    )
    return Verdict(suspicion=1.0, reason=reason)
