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
from langchain_sync_monitors.monitors.guard_labels import LABEL_WORD, LabelMatch, find_reply_label
from langchain_sync_monitors.options import (
    check_enum_option,
    check_string_set_option,
    read_count_option,
)

logger = logging.getLogger(__name__)

TOP_LOG_PROBABILITIES = 20
"""How many alternatives to request per token; 20 is the most OpenAI-style APIs return."""

LABEL_PATTERN = re.compile(LABEL_WORD)
"""A label the reply parser can read: letters or digits at both ends, where `_` or `-` is markup."""
LEADING_MARKUP_PATTERN = re.compile(r"^[\W_]+")
"""Markup a label line may open with, such as `**`, `__` or `(`; a token may carry it."""
UNREADABLE_LABEL_REASON = (
    "The guard model gave no readable label, so the step is treated as suspicious."
)
UNCERTAIN_LABEL_REASON = "The guard model was uncertain whether the step breaks the policy."

SUSPICIOUS_PROBABILITY = 0.5
"""From this probability on, a log-probability reason says the step breaks the policy.

The probability is that of a suspicious label, and from one half on, a
suspicious label is the guard's likelier finding. The library's default
thresholds are 0.6 and above, so every step a raw guard score blocks at a
default threshold reads as breaking the policy; a blocked step reads as
uncertain only under calibration or a threshold below one half.
"""
CONFIDENTLY_SAFE_PROBABILITY = 0.001
"""Below this probability, a log-probability reason says the step follows the policy.

The probability is that of a suspicious label. A guard that commits to a
label leaves the other kind a tiny probability: the real DeepSeek reply the
tests use left its safe alternatives about one in ten million. A probability
of 0.1% or more is doubt. The edge is small because under `CalibratedMonitor`
a threshold can block a step whose raw probability is below it, and that
step's reason would still say it follows the policy.
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
    the model's temperature is above zero and no response cache answers it.
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
    Log-probabilities in an unknown format are logged by their type alone,
    since their tokens are the guard's reply, which quotes the transcript.
    """
    payload = reply.response_metadata.get("logprobs")
    if payload is None:
        return None
    try:
        tokens = ReplyLogProbabilities.model_validate(payload).content
    except ValidationError:
        payload_type = type(payload).__name__
        logger.debug("Ignoring log-probabilities in an unknown format, of type %s.", payload_type)
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
    or `__viol` is read as the parser reads `(violation)` or `__violation__`.
    Trailing markup stays, since a token such as `no)` can begin no label.
    """
    return LEADING_MARKUP_PATTERN.sub("", token.strip()).lower()


def collect_alternatives(position: TokenLogProbability) -> dict[str, float]:
    """Map the guard's own token and its alternatives at a position to their log-probabilities."""
    alternatives = {alternative.token: alternative.logprob for alternative in position.top_logprobs}
    alternatives.setdefault(position.token, position.logprob)
    return alternatives


def validate_labels(*, suspicious_labels: AbstractSet[str], safe_labels: AbstractSet[str]) -> None:
    """Reject label sets that are not sets of strings, or are empty, overlapping or unreadable."""
    validate_label_set(
        suspicious_labels,
        parameter_name="suspicious_labels",
        example="{'violation'}",
    )
    validate_label_set(safe_labels, parameter_name="safe_labels", example="{'no_violation'}")
    shared = sorted(set(map(str.lower, suspicious_labels)) & set(map(str.lower, safe_labels)))
    if shared:
        message = f"a label cannot be in both suspicious_labels and safe_labels, got {shared}"
        raise ConfigurationError(message)


def validate_label_set(labels: AbstractSet[str], *, parameter_name: str, example: str) -> None:
    """Reject one label set that is not a set of strings, is empty, or holds an unreadable label.

    A label the reply parser could never read, such as `not safe` or
    `violation_`, would make every reply unreadable, so it is refused here.
    """
    check_string_set_option(labels, parameter_name=parameter_name, example=example)
    if not labels:
        message = f"{parameter_name} must hold at least one label"
        raise ConfigurationError(message)
    multi_word = sorted(filterfalse(is_one_word, labels))
    if multi_word:
        message = (
            f"{parameter_name} must hold only single words of letters, digits, _ or -, each "
            f"beginning and ending with a letter or digit, got {multi_word}"
        )
        raise ConfigurationError(message)


def is_one_word(label: str) -> bool:
    """Tell whether a label is one readable word: letters and digits, joined by `_` or `-`.

    The reply parser reads a `_` or `-` at either end of a word as markup.
    """
    return LABEL_PATTERN.fullmatch(label) is not None


class GuardModelMonitor(ChatModelMonitor):
    """A guard model (safety classifier) that labels a step against a policy you write.

    `policy_prompt` states the policy and must use `{transcript}` and
    `{proposed_step}`; it should ask for one label on a line of its own. The
    labels in `suspicious_labels` and `safe_labels` match case-insensitively,
    for example `{"violation"}` and `{"no_violation"}` for gpt-oss-safeguard,
    or `{"unsafe"}` and `{"safe"}` for Llama Guard; each is one word that
    begins and ends with a letter or digit. The label is read only on the
    reply's last line, or on its first when it is suspicious and only
    category codes follow, with any markup such as `**` or `__` around it or
    its key; a label with no key in a list item or a quote, as in
    `2. no_violation`, is no verdict. A reply without a label there, with two
    different labels on their own lines, or cut off at a length limit counts
    as a suspicious label, so it can only raise the suspicion: alone it gives
    1.0, and among several samples it is one suspicious vote.

    With log-probabilities, suspicion is the probability the guard gives to a
    suspicious label, the score Llama Guard, ShieldGemma and Granite Guardian
    report [@inan2023llamaguard; @zeng2024shieldgemma; @padhi2024graniteguardian].
    Without them, suspicion is the share of `samples` replies labelled
    suspicious. Studies of sampled answer confidence, none of them on guard
    models, find that the gain from more samples levels off: Lyu et al. saw
    calibration saturate after 15 to 20 samples [@lyu2025sampleconsistency],
    Xiong et al. saw it converge within 13 [@xiong2024uncertainty], and Tian
    et al. found no meaningful gain from 20 samples over 10 [@tian2023justask].
    Sampling needs the model's temperature above zero, and no response
    cache: LangChain's cache answers each sample with the first one's reply,
    and the monitor emits a `CachedResampleWarning` when it samples under
    one. Guard probabilities are overconfident [@liu2025guardcalibration], so
    set thresholds on honest-run percentiles with `CalibratedMonitor` rather
    than on raw values.

    The verdict's reason states the guard's finding, with no probability or
    count: the most severe label among sampled replies, or, from
    log-probabilities, one of three bands of the probability of a suspicious
    label (it breaks the policy, the guard was uncertain, or it follows the
    policy). Auto Mode shows the reason to the agent, and those numbers would
    tell it how close its step came to passing. They stay in the suspicion.
    """

    call_source: ClassVar[str] = "guard_model_monitor"
    prompt_parameter_name: ClassVar[str] = "policy_prompt"

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
        """Build a guard monitor; `samples` applies to sampling, including `AUTO`'s fallback.

        `samples` must be a whole number of at least 1, and each label set a
        set of strings; anything else raises `ConfigurationError`.
        """
        check_enum_option(scoring, option_type=GuardScoring, parameter_name="scoring")
        validate_labels(suspicious_labels=suspicious_labels, safe_labels=safe_labels)
        self.samples = read_count_option(samples, parameter_name="samples", minimum=1)
        super().__init__(model=model, prompt=policy_prompt, view=view)
        self.suspicious_labels = frozenset(label.lower() for label in suspicious_labels)
        self.safe_labels = frozenset(label.lower() for label in safe_labels)
        self.scoring = scoring
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
            return build_unscored_label_verdict()
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
        probability = self.compute_suspicious_probability(
            position, label_kind=self.classify_label(match.label)
        )
        if probability is None:
            return None
        reason = self.build_log_probability_reason(
            position,
            written_label=match.label,
            probability=probability,
        )
        return Verdict(suspicion=probability, reason=reason)

    def compute_suspicious_probability(
        self,
        position: TokenLogProbability,
        *,
        label_kind: LabelKind,
    ) -> float | None:
        """Probability of a suspicious label at the label's first token, among the labels.

        Every alternative that begins a label counts, so variants such as
        `violation`, ` violation` and `Violation` add up, and dividing by the mass
        of all label tokens renormalises over the labels, as in Granite Guardian's
        probability of risk [@padhi2024graniteguardian]. ShieldGemma renormalises
        over exactly `Yes` and `No` [@zeng2024shieldgemma]; Llama Guard reads its
        first token's probability as it is [@inan2023llamaguard].

        The probability is `None` unless the provider gave alternatives at the
        position and the guard's own token there begins a label of `label_kind`,
        the kind the reply's text names. Otherwise the alternatives would be
        weighed at a token that is not the label, where a label the guard all
        but ruled out could decide the score.
        """
        if not position.top_logprobs or self.classify_token(position.token) != label_kind:
            return None
        mass = {"suspicious": 0.0, "safe": 0.0}
        for token, logprob in collect_alternatives(position).items():
            kind = self.classify_token(token)
            if kind is not None:
                mass[kind] += math.exp(logprob)
        total = mass["suspicious"] + mass["safe"]
        return mass["suspicious"] / total if total > 0.0 else None

    def build_log_probability_reason(
        self,
        position: TokenLogProbability,
        *,
        written_label: str,
        probability: float,
    ) -> str:
        """State the guard's finding in one of three bands of the probability, with no number.

        From `SUSPICIOUS_PROBABILITY` on, the guard found that the step breaks the
        policy; below `CONFIDENTLY_SAFE_PROBABILITY`, that it follows it; in
        between, the guard was uncertain, and the reason names no label. The
        edges are explained where they are defined.
        """
        if CONFIDENTLY_SAFE_PROBABILITY <= probability < SUSPICIOUS_PROBABILITY:
            return UNCERTAIN_LABEL_REASON
        kind: LabelKind = "suspicious" if probability >= SUSPICIOUS_PROBABILITY else "safe"
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
        # Neither search below comes up empty: `kind` is asked for only when its labels have some
        # probability, which a token beginning a label of that kind carries.
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
        """Tell which kind of this monitor's labels a token begins; `None` for none, or both."""
        prefix = read_label_prefix(token)
        if not prefix:
            return None
        begins_suspicious = any(label.startswith(prefix) for label in self.suspicious_labels)
        begins_safe = any(label.startswith(prefix) for label in self.safe_labels)
        if begins_suspicious == begins_safe:
            return None
        return "suspicious" if begins_suspicious else "safe"

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


def build_unscored_label_verdict() -> Verdict:
    """Fail closed when log-probabilities came back but no label could be scored from them."""
    logger.warning(
        "No guard label could be scored from the log-probabilities; the step is suspicious."
    )
    reason = (
        "The guard model's reply had log-probabilities, but no label could be scored "
        "from them, so the step is treated as suspicious."
    )
    return Verdict(suspicion=1.0, reason=reason)
