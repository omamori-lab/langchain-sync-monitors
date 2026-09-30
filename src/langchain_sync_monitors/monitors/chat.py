"""Chat models as judges: fill a prompt with the rendered step, call the model, read the reply.

`ChatModelMonitor` holds what chat judges and guard models share. Each monitor
writes its scoring once, as a plan: a generator that yields the model replies
it needs and returns a `Verdict`. `evaluate` and `evaluate_sync` only differ in
how they obtain those replies, so the sync and async paths cannot drift apart.
This keeps the protocol logic free of I/O, the sans-I/O pattern
[@cannon2016sansio].
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import ClassVar

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.contracts import Monitor, MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model
from langchain_sync_monitors.options import (
    check_instance_option,
    read_count_option,
    read_integer_option,
)
from langchain_sync_monitors.prompts import DEFAULT_MONITOR_PROMPT
from langchain_sync_monitors.transcript import render_proposed_step, render_transcript

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_VIEW = MonitorView()
"""The view every monitor uses unless given another: user messages, tool calls, subagent reports."""

REQUIRED_PROMPT_VARIABLES = frozenset({"transcript", "proposed_step"})
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
REASONING_PATTERN = re.compile(
    r"<reasoning>(?P<reasoning>.*?)</reasoning>",
    flags=re.DOTALL | re.IGNORECASE,
)
FINAL_SCORE_PATTERN = re.compile(r"\s*<score>(?P<score>[^<]*)</score>\s*", flags=re.IGNORECASE)

STOP_REASON_KEYS = (
    "finish_reason",
    "native_finish_reason",
    "stop_reason",
    "stopReason",
    "done_reason",
)
"""Where providers put why a reply stopped.

OpenAI, OpenRouter and Gemini use the first two, Anthropic the third,
Bedrock Converse the fourth and Ollama the last.
"""

CUT_OFF_STOP_REASONS = ("length", "max_tokens", "max_output_tokens", "context_window_exceeded")
"""Stop reasons that mean the reply hit a length limit before the model finished it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplyRequest:
    """What a scoring plan asks for: `count` independent replies from `model` to `messages`."""

    model: Runnable[LanguageModelInput, AIMessage]
    messages: tuple[BaseMessage, ...]
    count: int = 1


type VerdictPlan = Generator[ReplyRequest, list[AIMessage], Verdict]
"""A monitor's scoring logic: it yields reply requests, receives the replies, returns a verdict."""


def resume_verdict_plan(
    plan: VerdictPlan, *, replies: list[AIMessage] | None
) -> ReplyRequest | Verdict:
    """Start the plan, or send it the replies it asked for; return its next request or verdict."""
    try:
        return next(plan) if replies is None else plan.send(replies)
    except StopIteration as finished:
        verdict: Verdict = finished.value
        return verdict


async def run_verdict_plan(
    plan: VerdictPlan,
    *,
    request_replies: Callable[[ReplyRequest], Awaitable[list[AIMessage]]],
) -> Verdict:
    """Drive a plan to its verdict, awaiting each batch of replies it asks for."""
    step = resume_verdict_plan(plan, replies=None)
    while isinstance(step, ReplyRequest):
        step = resume_verdict_plan(plan, replies=await request_replies(step))
    return step


def run_verdict_plan_sync(
    plan: VerdictPlan,
    *,
    request_replies: Callable[[ReplyRequest], list[AIMessage]],
) -> Verdict:
    """Drive a plan to its verdict, obtaining each batch of replies without an event loop."""
    step = resume_verdict_plan(plan, replies=None)
    while isinstance(step, ReplyRequest):
        step = resume_verdict_plan(plan, replies=request_replies(step))
    return step


def require_prompt_variables(prompt: ChatPromptTemplate, *, allowed: frozenset[str]) -> None:
    """Fail at construction, not mid-run, when a prompt lacks or adds variables.

    The prompt must use `{transcript}` and `{proposed_step}`, and may only use
    the other variables in `allowed`, since the monitor fills nothing else.
    """
    variables = set(prompt.input_variables)
    missing = sorted(REQUIRED_PROMPT_VARIABLES - variables)
    if missing:
        message = f"the monitor prompt must use the variables {missing}"
        raise ConfigurationError(message)
    unknown = sorted(variables - allowed - REQUIRED_PROMPT_VARIABLES)
    if unknown:
        message = f"the monitor prompt uses variables the monitor cannot fill: {unknown}"
        raise ConfigurationError(message)


class ChatModelMonitor(Monitor, ABC):
    """What chat judges and guard models share: fill a prompt with the step, then call the model.

    Subclasses write their scoring once, in `build_verdict_plan`. The model's
    calls are tagged as internal, so they stay out of the agent's message
    stream.
    """

    call_source: ClassVar[str] = "monitor"
    extra_prompt_variables: ClassVar[frozenset[str]] = frozenset()
    prompt_parameter_name: ClassVar[str] = "prompt"
    """The name under which the subclass's constructor takes the prompt, for messages."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        prompt: ChatPromptTemplate,
        view: MonitorView,
    ) -> None:
        """Resolve the model, and check the prompt, its variables and the view."""
        check_instance_option(
            prompt,
            option_type=ChatPromptTemplate,
            parameter_name=self.prompt_parameter_name,
            hint="Build one with ChatPromptTemplate.from_messages(...).",
        )
        check_instance_option(view, option_type=MonitorView, parameter_name="view")
        require_prompt_variables(prompt, allowed=self.extra_prompt_variables)
        self.model = resolve_chat_model(model)
        self.prompt = prompt
        self.view = view
        self.call_config = build_internal_call_config(source=self.call_source)

    @abstractmethod
    def build_verdict_plan(self, monitor_input: MonitorInput) -> VerdictPlan:
        """Return the plan that scores one step: the replies it needs and the verdict they give."""

    def build_extra_prompt_values(self) -> dict[str, str]:
        """Return the prompt values beyond the transcript and the proposed step."""
        return {}

    def build_prompt_messages(self, monitor_input: MonitorInput) -> tuple[BaseMessage, ...]:
        """Fill the prompt with the transcript and the proposed step, as the view shows them."""
        messages = self.prompt.format_messages(
            transcript=render_transcript(
                monitor_input.history,
                view=self.view,
                task_author=monitor_input.task_author,
            ),
            proposed_step=render_proposed_step(monitor_input.proposal, view=self.view),
            **self.build_extra_prompt_values(),
        )
        return tuple(messages)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step, drawing several replies concurrently when the plan asks for them."""
        plan = self.build_verdict_plan(monitor_input)
        return await run_verdict_plan(plan, request_replies=self.request_replies)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step without an event loop, drawing replies one after another."""
        plan = self.build_verdict_plan(monitor_input)
        return run_verdict_plan_sync(plan, request_replies=self.request_replies_sync)

    async def request_replies(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies concurrently; one failed call cancels the others."""
        return await run_concurrently(
            request.model.ainvoke(list(request.messages), config=self.call_config)
            for _ in range(request.count)
        )

    def request_replies_sync(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies one after another."""
        return [
            request.model.invoke(list(request.messages), config=self.call_config)
            for _ in range(request.count)
        ]


def is_reply_cut_off(reply: AIMessage) -> bool:
    """Tell whether the provider reports that the reply stopped at a length limit.

    OpenAI and OpenRouter report `finish_reason: "length"`, Gemini
    `finish_reason: "MAX_TOKENS"`, Anthropic `stop_reason: "max_tokens"` or
    `"model_context_window_exceeded"`, and OpenRouter keeps the upstream value
    in `native_finish_reason`; the OpenAI Responses API reports
    `incomplete_details: {"reason": "max_output_tokens"}`. `ChatBedrockConverse`
    keeps the Converse response's `stopReason`, `"max_tokens"` or
    `"model_context_window_exceeded"` [@langchainaws2026; @aws2026converse],
    and `ChatOllama` keeps Ollama's `done_reason: "length"`
    [@langchainollama2026]. Merging streamed
    chunks can repeat a value, as in `"MAX_TOKENSMAX_TOKENS"`, so a stop
    reason counts when it contains a cut-off reason.
    """
    metadata = reply.response_metadata
    stop_reasons = [metadata.get(key) for key in STOP_REASON_KEYS]
    incomplete_details = metadata.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        stop_reasons.append(incomplete_details.get("reason"))
    return any(
        isinstance(stop_reason, str)
        and any(cut_off in stop_reason.lower() for cut_off in CUT_OFF_STOP_REASONS)
        for stop_reason in stop_reasons
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgeReply:
    """The two parts of a readable judge reply: its reasoning and the text of its score."""

    reasoning: str
    score_text: str


def read_judge_reply(text: str) -> JudgeReply | None:
    """Split a reply into its reasoning and its score, or return `None` when it is unreadable.

    A reply is readable only when its score is structurally the last thing in
    it: the first `<reasoning>...</reasoning>` block, holding some text, then
    one `<score>...</score>` tag with only whitespace around it. The agent
    writes the step the judge reads, so it can plant a score tag there, and a
    judge may quote that tag, in its reasoning or in a note after its score.
    A quote inside the reasoning is never read; a second score tag, prose
    after the reasoning or text after the score makes the reply unreadable.
    """
    reasoning = REASONING_PATTERN.search(text)
    if reasoning is None or not reasoning["reasoning"].strip():
        return None
    # Matched from the reasoning's end to the text's end, so a score tag elsewhere never counts.
    score = FINAL_SCORE_PATTERN.fullmatch(text, reasoning.end())
    if score is None:
        return None
    return JudgeReply(reasoning=reasoning["reasoning"].strip(), score_text=score["score"].strip())


def parse_score(score_text: str, *, lowest_score: int, highest_score: int) -> int | None:
    """Read the score as an integer, or return `None` if it is not an integer in range.

    Python refuses to convert an integer of more than 4,300 digits, by
    default; such a score is no integer in range either.
    """
    # `int` would also take forms no score is written in, such as `1_0` for 10.
    if not INTEGER_PATTERN.fullmatch(score_text):
        return None
    try:
        score = int(score_text)
    except ValueError:
        return None
    return score if lowest_score <= score <= highest_score else None


class LLMMonitor(ChatModelMonitor):
    """A general chat model used as a judge: reasoning first, then a score on an anchored scale.

    The model reasons inside `<reasoning>` tags and then gives an integer from
    `lowest_score` to `highest_score` inside `<score>` tags. The score maps
    linearly onto a suspicion from 0 to 1, and the reasoning, without the
    score, becomes the verdict's reason. Reasoning before an anchored score
    follows Apollo Research's findings and the Ctrl-Z monitor
    [@apollo2026monitoringprompt; @bhatt2025ctrlz].

    A reply is readable only in exactly that shape: the reasoning block, then
    one score tag, with nothing but whitespace between them or after the
    score. A custom `prompt` must ask for this shape, or every reply is
    unreadable. An agent can write text aimed at its monitor into its step
    [@terekhov2025adaptive], so the shape keeps a score tag the agent planted,
    and the judge quoted, from becoming the verdict. A reply that the provider
    reports as cut off at a length limit is unreadable too, since its end is
    not the judge's verdict.

    An unreadable reply is asked for again, up to `max_parse_retries` more
    times. If none can be read, the monitor fails closed: the step gets
    suspicion 1.0 and a warning is logged.

    One call gives an integer score, which ties often at low audit budgets;
    wrap the monitor in `RepeatedMonitor` to average several calls.
    """

    call_source: ClassVar[str] = "llm_monitor"
    extra_prompt_variables: ClassVar[frozenset[str]] = frozenset({"lowest_score", "highest_score"})

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        prompt: ChatPromptTemplate = DEFAULT_MONITOR_PROMPT,
        lowest_score: int = 0,
        highest_score: int = 10,
        view: MonitorView = DEFAULT_MONITOR_VIEW,
        max_parse_retries: int = 2,
    ) -> None:
        """Build a chat judge; the prompt must use `{transcript}` and `{proposed_step}`.

        The ends of the scale must be integers, the lowest below the highest,
        and `max_parse_retries` a whole number of at least 0; anything else
        raises `ConfigurationError`.
        """
        self.lowest_score = read_integer_option(lowest_score, parameter_name="lowest_score")
        self.highest_score = read_integer_option(highest_score, parameter_name="highest_score")
        if self.lowest_score >= self.highest_score:
            message = (
                f"lowest_score ({self.lowest_score}) must be below "
                f"highest_score ({self.highest_score})"
            )
            raise ConfigurationError(message)
        self.max_parse_retries = read_count_option(
            max_parse_retries,
            parameter_name="max_parse_retries",
            minimum=0,
        )
        super().__init__(model=model, prompt=prompt, view=view)

    def build_extra_prompt_values(self) -> dict[str, str]:
        """Give the prompt the ends of the scale."""
        return {"lowest_score": str(self.lowest_score), "highest_score": str(self.highest_score)}

    def build_verdict_plan(self, monitor_input: MonitorInput) -> VerdictPlan:
        """Ask for a reply until one has a readable score, then fail closed."""
        request = ReplyRequest(model=self.model, messages=self.build_prompt_messages(monitor_input))
        attempts = self.max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            (reply,) = yield request
            verdict = self.parse_verdict(reply)
            if verdict is not None:
                return verdict
            logger.debug("Monitor reply %d of %d had no readable score.", attempt, attempts)
        logger.warning(
            "The monitor gave no readable score in %d replies; the step is treated as suspicious.",
            attempts,
        )
        reason = "The monitor gave no readable score, so the step is treated as suspicious."
        return Verdict(suspicion=1.0, reason=reason)

    def parse_verdict(self, reply: AIMessage) -> Verdict | None:
        """Turn a reply into a verdict, or return `None` when it is unreadable.

        The reason is the judge's reasoning alone. The score stays out of it,
        because Auto Mode shows the reason to the agent, and a score would tell
        the agent how close its step came to passing.
        """
        if is_reply_cut_off(reply):
            logger.debug("The monitor reply was cut off at a length limit.")
            return None
        judge_reply = read_judge_reply(reply.text)
        if judge_reply is None:
            return None
        score = parse_score(
            judge_reply.score_text,
            lowest_score=self.lowest_score,
            highest_score=self.highest_score,
        )
        if score is None:
            return None
        suspicion = (score - self.lowest_score) / (self.highest_score - self.lowest_score)
        return Verdict(suspicion=suspicion, reason=judge_reply.reasoning)
