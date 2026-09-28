"""Chat models as judges: fill a prompt with the rendered step, call the model, read the reply.

`ChatModelMonitor` holds what chat judges and guard models share. Each monitor
writes its scoring once, as a plan: a generator that yields the model replies
it needs and returns a `Verdict`. `evaluate` and `evaluate_sync` only differ in
how they obtain those replies, so the sync and async paths cannot drift apart.
This keeps the protocol logic free of I/O, the sans-I/O pattern
[@cannon2016sansio].
"""

from __future__ import annotations

import asyncio
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

from langchain_sync_monitors.contracts import Monitor, MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model
from langchain_sync_monitors.prompts import DEFAULT_MONITOR_PROMPT
from langchain_sync_monitors.transcript import render_proposed_step, render_transcript

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_VIEW = MonitorView()
"""The view every monitor uses unless given another: user messages, tool calls, subagent reports."""

REQUIRED_PROMPT_VARIABLES = frozenset({"transcript", "proposed_step"})
INTEGER_PATTERN = re.compile(r"[+-]?\d+")


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

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        prompt: ChatPromptTemplate,
        view: MonitorView,
    ) -> None:
        """Resolve the model and check the prompt's variables."""
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
        """Draw the requested replies concurrently."""
        replies = await asyncio.gather(
            *(
                request.model.ainvoke(list(request.messages), config=self.call_config)
                for _ in range(request.count)
            ),
        )
        return list(replies)

    def request_replies_sync(self, request: ReplyRequest) -> list[AIMessage]:
        """Draw the requested replies one after another."""
        return [
            request.model.invoke(list(request.messages), config=self.call_config)
            for _ in range(request.count)
        ]


def find_last_tag_text(text: str, *, tag: str) -> str | None:
    """Return the stripped content of the last `<tag>...</tag>` in `text`, if any."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches: list[str] = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def parse_score(text: str, *, lowest_score: int, highest_score: int) -> int | None:
    """Read the integer in the last `<score>` tag, or `None` if it is missing or out of range."""
    score_text = find_last_tag_text(text, tag="score")
    if score_text is None or not INTEGER_PATTERN.fullmatch(score_text):
        return None
    score = int(score_text)
    return score if lowest_score <= score <= highest_score else None


class LLMMonitor(ChatModelMonitor):
    """A general chat model used as a judge: reasoning first, then a score on an anchored scale.

    The model reasons inside `<reasoning>` tags and then gives an integer from
    `lowest_score` to `highest_score` inside `<score>` tags; the last score tag
    counts. The score maps linearly onto a suspicion from 0 to 1, and the
    reasoning becomes the verdict's reason. Reasoning before an anchored score
    follows Apollo Research's findings and the Ctrl-Z monitor
    [@apollo2026monitoringprompt; @bhatt2025ctrlz].

    A reply without a readable score is asked for again, up to
    `max_parse_retries` more times. If none can be read, the monitor fails
    closed: the step gets suspicion 1.0 and a warning is logged.

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
        """Build a chat judge; the prompt must use `{transcript}` and `{proposed_step}`."""
        if lowest_score >= highest_score:
            message = f"lowest_score ({lowest_score}) must be below highest_score ({highest_score})"
            raise ConfigurationError(message)
        if max_parse_retries < 0:
            message = f"max_parse_retries must not be negative, got {max_parse_retries}"
            raise ConfigurationError(message)
        super().__init__(model=model, prompt=prompt, view=view)
        self.lowest_score = lowest_score
        self.highest_score = highest_score
        self.max_parse_retries = max_parse_retries

    def build_extra_prompt_values(self) -> dict[str, str]:
        """Give the prompt the ends of the scale."""
        return {"lowest_score": str(self.lowest_score), "highest_score": str(self.highest_score)}

    def build_verdict_plan(self, monitor_input: MonitorInput) -> VerdictPlan:
        """Ask for a reply until one has a readable score, then fail closed."""
        request = ReplyRequest(model=self.model, messages=self.build_prompt_messages(monitor_input))
        attempts = self.max_parse_retries + 1
        for attempt in range(1, attempts + 1):
            (reply,) = yield request
            verdict = self.parse_verdict(reply.text)
            if verdict is not None:
                return verdict
            logger.debug("Monitor reply %d of %d had no readable score.", attempt, attempts)
        logger.warning(
            "The monitor gave no readable score in %d replies; the step is treated as suspicious.",
            attempts,
        )
        reason = (
            f"The monitor gave no readable score in {attempts} replies, "
            "so the step is treated as suspicious."
        )
        return Verdict(suspicion=1.0, reason=reason)

    def parse_verdict(self, text: str) -> Verdict | None:
        """Turn a reply into a verdict, or return `None` when it has no score in range."""
        score = parse_score(text, lowest_score=self.lowest_score, highest_score=self.highest_score)
        if score is None:
            return None
        suspicion = (score - self.lowest_score) / (self.highest_score - self.lowest_score)
        reasoning = find_last_tag_text(text, tag="reasoning")
        return Verdict(suspicion=suspicion, reason=reasoning or text.strip())
