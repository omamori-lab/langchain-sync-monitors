"""Guard models (safety classifiers) that label a step against a policy you write.

A guard model such as gpt-oss-safeguard or Llama Guard replies with a label,
not a score. A label alone gives a monitor one operating point, so every
threshold flags the same steps. `GuardModelMonitor` therefore turns labels into
a suspicion from 0 to 1: from the label token's log-probabilities when the
provider returns them, otherwise from the share of several sampled replies that
carry a suspicious label. The evidence is in
`docs/plans/initial-implementation/research/guard-model-scoring.md`.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections.abc import Generator, Iterator
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
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
    is_reply_cut_off,
)
from langchain_sync_monitors.monitors.guard_labels import LabelMatch, find_reply_label
from langchain_sync_monitors.options import check_enum_option

logger = logging.getLogger(__name__)

TOP_LOG_PROBABILITIES = 20
"""How many alternatives to request per token; 20 is the most OpenAI-style APIs return."""

LABEL_PATTERN = re.compile(r"[\w-]+")
LEADING_MARKUP_PATTERN = re.compile(r"^\W+")
"""Markup a label line may open with, such as `**` or `(`; a token may carry it before a label."""
UNREADABLE_LABEL_REASON = (
    "The guard model gave no readable label, so the step is treated as suspicious."
)
UNCERTAIN_LABEL_REASON = "The guard model was uncertain whether the step breaks the policy."

SUSPICIOUS_SHARE = 0.5
"""From this suspicious share on, a log-probability reason says the step breaks the policy.

At one half or more, a suspicious label is the guard's likelier finding. The
library's default thresholds are 0.6 and above, so every step a raw guard
score blocks at a default threshold reads as breaking the policy; a blocked
step reads as uncertain only under calibration or a threshold below one half.
"""
CONFIDENTLY_SAFE_SHARE = 0.001
"""Below this suspicious share, a log-probability reason says the step follows the policy.

A guard that commits to a label leaves the other kind a tiny share: the real
DeepSeek reply the tests use left its safe alternatives about one in ten
million. A share of 0.1% or more is doubt. The edge is small because under
`CalibratedMonitor` a threshold can block a step whose raw share is below it,
and that step's reason would still say it follows the policy.
"""

type LabelKind = Literal["suspicious", "safe"]


class GuardScoring(StrEnum):
    """How `GuardModelMonitor` turns the guard model's labels into a suspicion.

    `AUTO` reads log-probabilities when the provider returns them and samples
    otherwise. `LOG_PROBABILITIES` raises a `ConfigurationError` when the
    provider returns none, or none with alternatives. Both ask for them, and
    raise a `ConfigurationError` naming `SAMPLE_FRACTION` when the chat model
    rejects the request, as `ChatAnthropic` does. `SAMPLE_FRACTION` always
    samples. `HARD_LABEL` reads one label as 0 or 1 and warns, because every
    threshold then flags the same steps. Sampling only tells replies apart when
    the model's temperature is above zero.
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
    """The OpenAI-style `logprobs` entry of a reply's `response_metadata`.

    pydantic validates it [@pydantic2026], because the provider's payload is
    external data whose shape can change.
    """

    content: list[TokenLogProbability] | None = None


def read_log_probabilities(reply: AIMessage) -> list[TokenLogProbability] | None:
    """Return the reply's token log-probabilities, or `None` when they cannot give a score.

    The score weighs the alternatives at the label's token, so
    log-probabilities without any alternatives, as from a provider that
    ignores `top_logprobs`, count as none: they would only give a hard label.
    """
    payload = reply.response_metadata.get("logprobs")
    if payload is None:
        return None
    try:
        tokens = ReplyLogProbabilities.model_validate(payload).content
    except ValidationError:
        logger.debug("Ignoring log-probabilities in an unknown format: %r", payload)
        return None
    if not tokens or not any(token.top_logprobs for token in tokens):
        return None
    return tokens


def find_token_index(tokens: list[TokenLogProbability], *, offset: int) -> int | None:
    """Return the index of the token that covers the character at `offset`."""
    end = 0
    for index, token in enumerate(tokens):
        end += len(token.token)
        if end > offset:
            return index
    return None


def read_label_prefix(token: str) -> str:
    """Return the start of a label a token may carry: no whitespace, no leading markup, lower case.

    The markup is what a label line may open with, so a token such as `(viol`
    is read as the parser reads `(violation)`. Trailing markup stays, since a
    token such as `no)` can begin no label.
    """
    return LEADING_MARKUP_PATTERN.sub("", token.strip()).lower()


def classify_label_token(
    token: str,
    *,
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
) -> LabelKind | None:
    """Tell which kind of label a token begins; `None` if it begins none, or both kinds."""
    prefix = read_label_prefix(token)
    if not prefix:
        return None
    begins_suspicious = any(label.startswith(prefix) for label in suspicious_labels)
    begins_safe = any(label.startswith(prefix) for label in safe_labels)
    if begins_suspicious == begins_safe:
        return None
    return "suspicious" if begins_suspicious else "safe"


def collect_alternatives(position: TokenLogProbability) -> dict[str, float]:
    """Map the guard's own token and its alternatives at a position to their log-probabilities."""
    alternatives = {alternative.token: alternative.logprob for alternative in position.top_logprobs}
    alternatives.setdefault(position.token, position.logprob)
    return alternatives


def compute_suspicious_share(
    position: TokenLogProbability,
    *,
    label_kind: LabelKind,
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
) -> float | None:
    """Share of the label probability at the label's first token that is suspicious.

    Every alternative that begins a label counts, so variants such as
    `violation`, ` violation` and `Violation` add up, as in Granite Guardian's
    probability of risk [@padhi2024graniteguardian]. Dividing by the mass of
    all label tokens renormalises over the labels, as Llama Guard and
    ShieldGemma do [@inan2023llamaguard; @zeng2024shieldgemma].

    The share is `None` unless the provider gave alternatives at the position
    and the guard's own token there begins a label of `label_kind`, the kind
    the reply's text names. Otherwise the alternatives would be weighed at a
    token that is not the label, where a label the guard all but ruled out
    could decide the score.
    """
    chosen_kind = classify_label_token(
        position.token,
        suspicious_labels=suspicious_labels,
        safe_labels=safe_labels,
    )
    if not position.top_logprobs or chosen_kind != label_kind:
        return None
    mass = {"suspicious": 0.0, "safe": 0.0}
    for token, logprob in collect_alternatives(position).items():
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
    or `{"unsafe"}` and `{"safe"}` for Llama Guard. The label is read only on
    the reply's last line, or on its first when it is suspicious and only
    category codes follow. A reply without a label there, with two different
    labels on their own lines, or cut off at a length limit counts as a
    suspicious label, so it can only raise the suspicion: alone it gives 1.0,
    and among several samples it is one suspicious vote.

    With log-probabilities, suspicion is the probability the guard gives to a
    suspicious label, the score Llama Guard, ShieldGemma and Granite Guardian
    report [@inan2023llamaguard; @zeng2024shieldgemma; @padhi2024graniteguardian].
    Without them, suspicion is the share of `samples` replies labelled
    suspicious; the gain from more samples levels off around 5 to 10
    [@xiong2024uncertainty; @lyu2025sampleconsistency; @tian2023justask].
    Sampling needs the model's temperature above zero. Guard probabilities
    are overconfident [@liu2025guardcalibration], so set thresholds on
    honest-run percentiles with `CalibratedMonitor` rather than on raw values.

    The verdict's reason states the guard's finding, with no probability or
    count: the most severe label among sampled replies, or, from
    log-probabilities, one of three bands of the suspicious share (it breaks
    the policy, the guard was uncertain, or it follows the policy). Auto Mode
    shows the reason to the agent, and those numbers would tell it how close
    its step came to passing. They stay in the suspicion.
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
        check_enum_option(scoring, option_type=GuardScoring, parameter_name="scoring")
        validate_labels(suspicious_labels=suspicious_labels, safe_labels=safe_labels)
        if samples < 1:
            message = f"samples must be at least 1, got {samples}"
            raise ConfigurationError(message)
        super().__init__(model=model, prompt=policy_prompt, view=view)
        self.suspicious_labels = frozenset(label.lower() for label in suspicious_labels)
        self.safe_labels = frozenset(label.lower() for label in safe_labels)
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
        # The first reply counts as a sample, so with `samples=1` no more are drawn.
        remaining = self.samples - 1
        more_replies = yield ReplyRequest(model=self.model, messages=messages, count=remaining)
        return self.build_sample_verdict([first_reply, *more_replies])

    def build_log_probability_verdict(self, reply: AIMessage) -> Verdict | None:
        """Score the reply from its log-probabilities, or return `None` when they are unusable.

        Missing log-probabilities, or ones without alternatives, are a
        configuration error under `LOG_PROBABILITIES`, since that mode cannot
        score without them. A reply cut off at a length limit is unusable.
        """
        tokens = read_log_probabilities(reply)
        if tokens is None and self.scoring is GuardScoring.LOG_PROBABILITIES:
            message = (
                f"{self.model.get_name()} returned no log-probabilities with alternatives to "
                "score from; use GuardScoring.AUTO or GuardScoring.SAMPLE_FRACTION with this model"
            )
            raise ConfigurationError(message)
        if not tokens or is_reply_cut_off(reply):
            return None
        return self.find_label_probability(tokens)

    def find_label_probability(self, tokens: list[TokenLogProbability]) -> Verdict | None:
        """Score from the alternatives at the label's first token; `None` if they are unusable."""
        text = "".join(token.token for token in tokens)
        match = self.find_label(text)
        index = find_token_index(tokens, offset=match.offset) if match else None
        if match is None or index is None:
            return None
        position = tokens[index]
        share = compute_suspicious_share(
            position,
            label_kind=self.classify_label(match.label),
            suspicious_labels=self.suspicious_labels,
            safe_labels=self.safe_labels,
        )
        if share is None:
            return None
        reason = self.build_log_probability_reason(
            position,
            written_label=match.label,
            share=share,
        )
        return Verdict(suspicion=share, reason=reason)

    def build_log_probability_reason(
        self,
        position: TokenLogProbability,
        *,
        written_label: str,
        share: float,
    ) -> str:
        """State the guard's finding in one of three bands of the suspicious share, with no number.

        From `SUSPICIOUS_SHARE` on, the guard found that the step breaks the
        policy; below `CONFIDENTLY_SAFE_SHARE`, that it follows it; in
        between, the guard was uncertain, and the reason names no label. The
        edges are explained where they are defined.
        """
        if CONFIDENTLY_SAFE_SHARE <= share < SUSPICIOUS_SHARE:
            return UNCERTAIN_LABEL_REASON
        kind: LabelKind = "suspicious" if share >= SUSPICIOUS_SHARE else "safe"
        label = self.name_label_of_kind(position, written_label=written_label, kind=kind)
        return self.build_label_reason(label)

    def name_label_of_kind(
        self,
        position: TokenLogProbability,
        *,
        written_label: str,
        kind: LabelKind,
    ) -> str:
        """Name a label of `kind`: the one the guard wrote, or else its likeliest of that kind.

        When the guard wrote a label of the other kind, the reason names the
        likeliest label of `kind` instead, so that it agrees with the
        suspicion and reads as it would had the guard written that label: a
        different wording would tell the agent that its step came close.
        """
        if self.classify_label(written_label) == kind:
            return written_label
        # Neither search below comes up empty: `kind` is asked for only when the share gives it
        # some probability, which a token beginning a label of that kind carries.
        alternatives = collect_alternatives(position)
        likeliest_token = max(
            (token for token in alternatives if self.classify_token(token) == kind),
            key=alternatives.__getitem__,
        )
        prefix = read_label_prefix(likeliest_token)
        labels = self.suspicious_labels if kind == "suspicious" else self.safe_labels
        return min(label for label in labels if label.startswith(prefix))

    def classify_label(self, label: str) -> LabelKind:
        """Tell whether one of this monitor's labels is suspicious or safe."""
        return "suspicious" if label in self.suspicious_labels else "safe"

    def classify_token(self, token: str) -> LabelKind | None:
        """Tell which kind of this monitor's labels a token begins, if exactly one."""
        return classify_label_token(
            token,
            suspicious_labels=self.suspicious_labels,
            safe_labels=self.safe_labels,
        )

    async def request_replies(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies concurrently; see `explain_rejected_log_probabilities`."""
        with self.explain_rejected_log_probabilities(request):
            return await super().request_replies(request)

    def request_replies_sync(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies in turn; see `explain_rejected_log_probabilities`."""
        with self.explain_rejected_log_probabilities(request):
            return super().request_replies_sync(request)

    @contextmanager
    def explain_rejected_log_probabilities(self, request: ReplyRequest) -> Iterator[None]:
        """Turn a chat model's refusal of the `logprobs` keyword into a `ConfigurationError`.

        LangChain's model profiles have no capability flag for
        log-probabilities [@langchaincore2026], so the request for them is sent
        to every model. Some adapters, such as `ChatAnthropic` and
        `ChatBedrockConverse`, pass the keyword on to a function that does not
        take it [@langchainanthropic2026; @langchainaws2026], and Python raises
        a `TypeError`, "got an unexpected keyword argument 'logprobs'", before
        any request leaves. Only that error, raised by the request for
        log-probabilities, becomes a `ConfigurationError` naming the mode to
        use; every other error passes through unchanged, including a
        `TypeError` that merely mentions `logprobs`.
        """
        try:
            yield
        except TypeError as error:
            is_log_probability_request = request.model is self.model_with_log_probabilities
            if not is_log_probability_request or not is_rejected_keyword(error, keyword="logprobs"):
                raise
            message = (
                f"{self.model.get_name()} does not accept the request for log-probabilities "
                "that GuardScoring.AUTO and GuardScoring.LOG_PROBABILITIES make; use "
                "GuardScoring.SAMPLE_FRACTION with this model"
            )
            raise ConfigurationError(message) from error

    def build_sample_verdict(self, replies: list[AIMessage]) -> Verdict:
        """Return the share of replies labelled suspicious; an unreadable reply counts as one.

        The reason names the most severe finding: a suspicious label if any
        reply gave one, then an unreadable reply, then the safe label.
        """
        labels = [self.read_reply_label(reply) for reply in replies]
        readable_labels = [label for label in labels if label is not None]
        suspicious_labels = [label for label in readable_labels if label in self.suspicious_labels]
        # Every unreadable reply is a suspicious vote, so the count fails closed.
        suspicious_count = len(labels) - len(readable_labels) + len(suspicious_labels)
        if suspicious_labels:
            reason = self.build_label_reason(suspicious_labels[0])
        elif len(readable_labels) < len(labels):
            reason = UNREADABLE_LABEL_REASON
        else:
            reason = self.build_label_reason(readable_labels[0])
        return Verdict(suspicion=suspicious_count / len(replies), reason=reason)

    def read_reply_label(self, reply: AIMessage) -> str | None:
        """Return the reply's label, or `None` when it has none or was cut off at a length limit."""
        if is_reply_cut_off(reply):
            return None
        match = self.find_label(reply.text)
        return match.label if match else None

    def find_label(self, text: str) -> LabelMatch | None:
        """Find the guard's label in a reply's text, with this monitor's labels."""
        return find_reply_label(
            text,
            suspicious_labels=self.suspicious_labels,
            safe_labels=self.safe_labels,
        )

    def build_label_reason(self, label: str) -> str:
        """State what the guard's label says about the step, without any number."""
        finding = "breaks" if label in self.suspicious_labels else "follows"
        return f"The guard model labelled the step {label!r}, finding that it {finding} the policy."


def is_rejected_keyword(error: TypeError, *, keyword: str) -> bool:
    """Tell whether Python refused a call for an unexpected keyword argument named `keyword`."""
    text = str(error).lower()
    return "unexpected keyword argument" in text and keyword in text


def build_unlocated_label_verdict() -> Verdict:
    """Fail closed when log-probabilities came back but no label could be scored from them."""
    logger.warning(
        "No guard label could be scored from the log-probabilities; the step is suspicious."
    )
    reason = (
        "The guard model's reply had log-probabilities but no readable label, "
        "so the step is treated as suspicious."
    )
    return Verdict(suspicion=1.0, reason=reason)
