"""Chat models as monitors: fill a prompt with the rendered step, call the model, read the reply.

`ChatModelMonitor` holds what LLM monitors and guard models share. Each monitor
writes its scoring once, as a plan: a generator that yields the model replies
it needs and returns a `Verdict`. `evaluate` and `evaluate_sync` only differ in
how they obtain those replies, so the sync and async paths cannot drift apart.
This keeps the protocol logic free of I/O, the sans-I/O pattern
[@cannon2016sansio].
"""

from __future__ import annotations

import functools
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, replace
from typing import ClassVar

import httpx
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.contracts import Monitor, MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.model_calls import (
    CachedResampleWarning,
    build_internal_call_config,
    is_response_cache_active,
    resolve_chat_model,
)
from langchain_sync_monitors.options import (
    check_instance_option,
    read_count_option,
    read_integer_option,
    write_integer_option,
)
from langchain_sync_monitors.prompts import DEFAULT_MONITOR_PROMPT
from langchain_sync_monitors.retries import call_with_retries, call_with_retries_sync
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY
from langchain_sync_monitors.transcript import render_proposed_step, render_transcript

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_VIEW = MonitorView()
"""The view every monitor uses unless given another: user messages, tool calls, subagent reports."""

REQUIRED_PROMPT_VARIABLES = frozenset({"transcript", "proposed_step"})
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
REASONING_START_TAG_PATTERN = re.compile(r"<reasoning>", flags=re.IGNORECASE)
REASONING_END_TAG_PATTERN = re.compile(r"</reasoning>", flags=re.IGNORECASE)
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

RATE_LIMIT_ATTEMPTS = 4
"""How many times a monitor's call is tried in all while the provider answers HTTP 429."""
RATE_LIMIT_FIRST_WAIT_SECONDS = 1.0
"""The wait before the first retry of a rate-limited call; each later wait doubles, up to 5 s."""


def is_rate_limit_error(error: Exception) -> bool:
    """Tell whether a chat model's call failed on a rate limit, HTTP 429.

    A chat model's own `max_retries` does not always cover one:
    `ChatOpenRouter` hands its retries to the OpenRouter SDK
    [@langchainopenrouter2026], which retries a chat completion on HTTP 5xx
    and network errors alone, and whose retry settings name no status code
    [@openrouterpythonsdk2026]. Provider SDKs put the status on their errors
    as `status_code`, OpenRouter's, OpenAI's and Anthropic's among them, and
    the last two retry a 429 themselves too
    [@openaipythonsdk2026; @anthropicpythonsdk2026]; httpx puts the status
    on the error's response [@httpx2024].
    """
    if isinstance(error, httpx.HTTPStatusError):
        status: object = error.response.status_code
    else:
        status = getattr(error, "status_code", None)
    return status == httpx.codes.TOO_MANY_REQUESTS


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplyRequest:
    """What a scoring plan asks for: `count` independent replies from `model` to `messages`.

    `repeated` says the plan asked for a reply to the same messages before.
    """

    model: Runnable[LanguageModelInput, AIMessage]
    messages: tuple[BaseMessage, ...]
    count: int = 1
    repeated: bool = False


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
    request_or_verdict = resume_verdict_plan(plan, replies=None)
    while isinstance(request_or_verdict, ReplyRequest):
        request_or_verdict = resume_verdict_plan(
            plan, replies=await request_replies(request_or_verdict)
        )
    return request_or_verdict


def run_verdict_plan_sync(
    plan: VerdictPlan,
    *,
    request_replies: Callable[[ReplyRequest], list[AIMessage]],
) -> Verdict:
    """Drive a plan to its verdict, obtaining each batch of replies without an event loop."""
    request_or_verdict = resume_verdict_plan(plan, replies=None)
    while isinstance(request_or_verdict, ReplyRequest):
        request_or_verdict = resume_verdict_plan(plan, replies=request_replies(request_or_verdict))
    return request_or_verdict


@functools.cache
def warn_about_cached_monitor_replies() -> None:
    """Emit the `CachedResampleWarning` for a monitor's model, on the first call in the process.

    The cache on this function keeps the warning to one per process, as
    `pending_steps.warn_about_cached_resamples` does for the agent's model.
    """
    warnings.warn(
        "A LangChain response cache is active for a monitor's model, so each repeat of its "
        "prompt returns a copy of the first reply: a guard's samples all carry the first "
        "label, and an LLM monitor asked again after an unreadable reply gets the same reply. "
        "Build the monitor's model with cache=False, or unset the global cache with "
        "set_llm_cache(None).",
        CachedResampleWarning,
        stacklevel=2,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )


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
    """What LLM monitors and guard models share: fill a prompt with the step, then call the model.

    Subclasses write their scoring once, in `build_verdict_plan`. The model's
    calls carry LangChain's internal-call metadata, which drops them from
    `stream_events(version="v3")`; the monitor middleware's `nostream` tag
    keeps them out of `stream_mode="messages"`. A call the provider answers
    with HTTP 429 is made again, as `request_reply` says; every other error
    is left to the chat model's own `max_retries`.
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
        self.warn_when_replies_are_cached(request)
        return await run_concurrently(self.request_reply(request) for _ in range(request.count))

    def request_replies_sync(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies one after another."""
        self.warn_when_replies_are_cached(request)
        return [self.request_reply_sync(request) for _ in range(request.count)]

    def warn_when_replies_are_cached(self, request: ReplyRequest) -> None:
        """Warn, once per process, when a request repeats a prompt under a response cache.

        A request repeats one when it asks for several replies at once, or
        when the plan asked for a reply to the same messages before. The cache
        is checked at each draw, so one set after the monitor was built counts.
        """
        if (request.count > 1 or request.repeated) and is_response_cache_active(self.model):
            warn_about_cached_monitor_replies()

    async def request_reply(self, request: ReplyRequest) -> AIMessage:
        """Draw one reply, calling the model again after a rate limit.

        A call the provider answers with HTTP 429 is tried again with stamina
        [@schlawack2026stamina], after a growing, jittered wait, up to
        `RATE_LIMIT_ATTEMPTS` attempts in all; any other error is raised at
        once, and so is the provider's last 429. `call_with_retries` retries
        the call, so no retry hook is handed the prompt, or the provider's
        error, which can hold its whole reply: stamina's retry log holds the
        wait and a `RetriedCallError`'s repr, which names the provider error's
        type and its status alone. The call is a nested function, whose repr
        names it alone, since a hook can read the retried block's repr from
        the stand-in's traceback.
        """
        messages = list(request.messages)

        async def call_model() -> AIMessage:
            return await request.model.ainvoke(messages, config=self.call_config)

        return await call_with_retries(
            call_model,
            is_retried=is_rate_limit_error,
            attempts=RATE_LIMIT_ATTEMPTS,
            wait_initial=RATE_LIMIT_FIRST_WAIT_SECONDS,
        )

    def request_reply_sync(self, request: ReplyRequest) -> AIMessage:
        """Draw one reply without an event loop, calling the model again after a rate limit."""
        messages = list(request.messages)

        def call_model() -> AIMessage:
            return request.model.invoke(messages, config=self.call_config)

        return call_with_retries_sync(
            call_model,
            is_retried=is_rate_limit_error,
            attempts=RATE_LIMIT_ATTEMPTS,
            wait_initial=RATE_LIMIT_FIRST_WAIT_SECONDS,
        )


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
class ReasoningBlock:
    """The first `<reasoning>...</reasoning>` block of a reply: its text, and where it ends."""

    text: str
    end: int


def find_reasoning_block(text: str) -> ReasoningBlock | None:
    """Return the reply's first reasoning block, or `None` when it has none.

    The block runs from the first start tag to the first end tag after it,
    the block a lazy search for a start tag, any text and an end tag finds.
    When the first start tag has no end tag after it, no later one has, so
    the reply has no block. Searched that way, a reply of many start tags and
    no end tag took time quadratic in its length, since the search started
    again at each start tag; this takes two linear scans.
    """
    start_tag = REASONING_START_TAG_PATTERN.search(text)
    if start_tag is None:
        return None
    end_tag = REASONING_END_TAG_PATTERN.search(text, start_tag.end())
    if end_tag is None:
        return None
    return ReasoningBlock(text=text[start_tag.end() : end_tag.start()], end=end_tag.end())


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgeReply:
    """The two parts of a readable reply: its reasoning and the text of its score."""

    reasoning: str
    score_text: str


def read_judge_reply(text: str) -> JudgeReply | None:
    """Split a reply into its reasoning and its score, or return `None` when it is unreadable.

    A reply is readable only when its score is structurally the last thing in
    it: the first `<reasoning>...</reasoning>` block, holding some text, then
    one `<score>...</score>` tag with only whitespace around it. The agent
    writes the step the monitor reads, so it can plant a score tag there, and
    the model may quote that tag, in its reasoning or in a note after its score.
    A quote inside the reasoning is never read; a second score tag, prose
    after the reasoning or text after the score makes the reply unreadable.
    """
    reasoning = find_reasoning_block(text)
    if reasoning is None or not reasoning.text.strip():
        return None
    # Matched from the reasoning's end to the text's end, so a score tag elsewhere never counts.
    score = FINAL_SCORE_PATTERN.fullmatch(text, reasoning.end)
    if score is None:
        return None
    return JudgeReply(reasoning=reasoning.text.strip(), score_text=score["score"].strip())


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
    """A general chat model used as a monitor: reasoning first, then a score on an anchored scale.

    The model reasons inside `<reasoning>` tags and then gives an integer from
    `lowest_score` to `highest_score` inside `<score>` tags. The score maps
    linearly onto a suspicion from 0 to 1, and the reasoning, without the
    score, becomes the verdict's reason. Reasoning before an anchored score
    follows Apollo Research's findings (section Summary) and the Ctrl-Z monitor
    (appendix D) [@apollo2026monitoringprompt; @bhatt2025ctrlz]. The default
    scale of 0 to 10 is the one BashArena (section 4.1), LinuxArena (section
    5.1) and Koran et al. (section 3.1) use
    [@kaufman2025basharena; @tracy2026linuxarena; @koran2026ensemble]; Ctrl-Z's
    monitor scores 0 to 9 and Apollo Research's monitor 1 to 10.

    A reply is readable only in exactly that shape: the reasoning block, then
    one score tag, with nothing but whitespace between them or after the
    score. A custom `prompt` must ask for this shape, or every reply is
    unreadable. An agent can write text aimed at its monitor into its step
    [@terekhov2025adaptive], so the shape keeps a score tag the agent planted,
    and the model quoted, from becoming the verdict. A reply that the provider
    reports as cut off at a length limit is unreadable too, since its end is
    not the model's verdict.

    An unreadable reply is asked for again, up to `max_parse_retries` more
    times. If none can be read, the monitor fails closed: the step gets
    suspicion 1.0 and a warning is logged. A response cache on the model
    answers each request again with the first reply, so the monitor emits a
    `CachedResampleWarning` when it asks again under one; build the model
    with `cache=False`.

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
        """Build an LLM monitor; the prompt must use `{transcript}` and `{proposed_step}`.

        The ends of the scale must be integers Python can write out, since
        the prompt shows them, the lowest below the highest, and
        `max_parse_retries` a whole number of at least 0; anything else
        raises `ConfigurationError`.
        """
        self.lowest_score = read_integer_option(lowest_score, parameter_name="lowest_score")
        self.highest_score = read_integer_option(highest_score, parameter_name="highest_score")
        # Written once here, so an end the prompt cannot show fails now, not at the first step.
        self.scale_ends = {
            "lowest_score": write_integer_option(self.lowest_score, parameter_name="lowest_score"),
            "highest_score": write_integer_option(
                self.highest_score, parameter_name="highest_score"
            ),
        }
        if self.lowest_score >= self.highest_score:
            message = (
                f"lowest_score ({self.scale_ends['lowest_score']}) must be below "
                f"highest_score ({self.scale_ends['highest_score']})"
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
        return dict(self.scale_ends)

    def build_verdict_plan(self, monitor_input: MonitorInput) -> VerdictPlan:
        """Ask for a reply until one has a readable score, then fail closed."""
        request = ReplyRequest(model=self.model, messages=self.build_prompt_messages(monitor_input))
        attempts = self.max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            (reply,) = yield replace(request, repeated=attempt > 1)
            verdict = self.parse_verdict(reply)
            if verdict is not None:
                return verdict
            logger.debug("Monitor reply %d of %d had no readable score.", attempt, attempts)
        logger.warning(
            "The monitor gave no readable score in %d %s; the step is treated as suspicious.",
            attempts,
            "reply" if attempts == 1 else "replies",
        )
        reason = "The monitor gave no readable score, so the step is treated as suspicious."
        return Verdict(suspicion=1.0, reason=reason)

    def parse_verdict(self, reply: AIMessage) -> Verdict | None:
        """Turn a reply into a verdict, or return `None` when it is unreadable.

        The reason is the model's reasoning alone. The score stays out of it,
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
