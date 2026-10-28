"""Scripted monitors, written as hyoshigi's twins, whose scores depend on what they read."""

from __future__ import annotations

import asyncio
import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import hyoshigi
from hyoshigi import (
    Action,
    AsyncGenerativeModel,
    Author,
    Event,
    GenerationOutcome,
    GenerativeModel,
    Observation,
    Prompt,
    Rationale,
    SuspicionScore,
    TextPart,
    Trajectory,
    View,
)
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from pydantic import Field

from langchain_sync_monitors import LLMMonitor, MonitorTwins
from langchain_sync_monitors.model_calls import build_internal_call_config

BENIGN_SUSPICION = 0.1
MONITOR_CALL_SOURCE = "fake_monitor"
KEYWORD_MODEL = "keyword"


def read_action_text(action: Action) -> str:
    """Return the text a keyword is looked for in: the action's text, then each call's name and
    arguments.
    """
    texts = [part.text for part in (*action.thought, *action.parts) if isinstance(part, TextPart)]
    calls = [f"{call.name} {call.arguments}" for call in action.tool_calls]
    return "\n".join([*texts, *calls])


def score_by_keyword(text: str, *, suspicion_by_keyword: Mapping[str, float]) -> SuspicionScore:
    """Score text by the first keyword it holds, or as benign when it holds none."""
    for keyword, suspicion in suspicion_by_keyword.items():
        if keyword in text:
            return SuspicionScore(
                value=suspicion,
                explanation=f"mentions {keyword}",
                evidence=Rationale(model=KEYWORD_MODEL),
            )
    return SuspicionScore(
        value=BENIGN_SUSPICION,
        explanation="nothing suspicious",
        evidence=Rationale(model=KEYWORD_MODEL),
    )


@dataclass(frozen=True, kw_only=True)
class KeywordRating:
    """What a keyword monitor rated: the action, the history before it, and the score it gave."""

    action: Action
    history: tuple[Event, ...]
    score: SuspicionScore

    @property
    def task_author(self) -> Author | None:
        """The author of the history's first task entry: the user or the delegating agent."""
        task_authors = (Author.USER, Author.DELEGATOR)
        return next(
            (
                event.author
                for event in self.history
                if isinstance(event, Observation) and event.author in task_authors
            ),
            None,
        )


def build_monitor_call_config(call_tags: Sequence[str]) -> RunnableConfig:
    """Return the config of a monitor's own model call, with tags of its own when given."""
    config = build_internal_call_config(source=MONITOR_CALL_SOURCE)
    if call_tags:
        config["tags"] = list(call_tags)
    return config


@dataclass(frozen=True, kw_only=True)
class KeywordRater(hyoshigi.Monitor):
    """Rates an action by the first keyword it holds, keeping each rating in `ratings`."""

    suspicion_by_keyword: Mapping[str, float]
    ratings: list[KeywordRating]
    model: BaseChatModel | None
    call_tags: Sequence[str]

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        if self.model is not None:
            self.model.invoke("Judge this step.", config=build_monitor_call_config(self.call_tags))
        score = score_by_keyword(
            read_action_text(action),
            suspicion_by_keyword=self.suspicion_by_keyword,
        )
        self.ratings.append(KeywordRating(action=action, history=trajectory.history, score=score))
        return score


@dataclass(frozen=True, kw_only=True)
class AsyncKeywordRater(hyoshigi.AsyncMonitor):
    """Rates an action as `KeywordRater` does, awaiting the model call."""

    suspicion_by_keyword: Mapping[str, float]
    ratings: list[KeywordRating]
    model: BaseChatModel | None
    call_tags: Sequence[str]

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        if self.model is not None:
            await self.model.ainvoke(
                "Judge this step.",
                config=build_monitor_call_config(self.call_tags),
            )
        score = score_by_keyword(
            read_action_text(action),
            suspicion_by_keyword=self.suspicion_by_keyword,
        )
        self.ratings.append(KeywordRating(action=action, history=trajectory.history, score=score))
        return score


@dataclass(frozen=True, kw_only=True)
class KeywordMonitor(MonitorTwins):
    """Twins that score an action by the first keyword it holds, so ratings drawn at once agree.

    Keying the score on the action keeps concurrent sampling deterministic.
    With `model`, every rating also makes one tagged internal model call, as
    a real monitor would; with `call_tags`, that call carries tags of its
    own. Both twins keep their ratings in `ratings`.
    """

    suspicion_by_keyword: Mapping[str, float] = field(default_factory=dict)
    model: BaseChatModel | None = None
    call_tags: Sequence[str] = ()
    ratings: list[KeywordRating] = field(default_factory=list)
    monitor: hyoshigi.Monitor = field(init=False)
    async_monitor: hyoshigi.AsyncMonitor = field(init=False)

    def __post_init__(self) -> None:
        options: dict[str, Any] = {
            "suspicion_by_keyword": self.suspicion_by_keyword,
            "ratings": self.ratings,
            "model": self.model,
            "call_tags": self.call_tags,
        }
        object.__setattr__(self, "monitor", KeywordRater(**options))
        object.__setattr__(self, "async_monitor", AsyncKeywordRater(**options))
        super().__post_init__()


@dataclass(frozen=True, kw_only=True)
class MonitorReading:
    """What a rendering monitor's model read for one rating, and the score it gave."""

    trajectory: str
    proposed_action: str
    suspicion: float
    reason: str


TRAJECTORY_PATTERN = re.compile(r"<trajectory>\n(.*?)\n</trajectory>", re.DOTALL)
PROPOSED_ACTION_PATTERN = re.compile(r"<proposed_action>\n(.*?)\n</proposed_action>", re.DOTALL)


def read_element(pattern: re.Pattern[str], text: str) -> str:
    """Return the content of the one element the pattern finds, or an empty string."""
    found = pattern.search(text)
    return found[1] if found else ""


class KeywordScoringChatModel(BaseChatModel):
    """An LLM monitor's chat model that scores the proposed action by keyword, out of 100.

    It reads the request hyoshigi's default prompt builds, keeps what it read
    in `readings`, and replies in the shape the LLM monitor reads.
    """

    suspicion_by_keyword: Mapping[str, float]
    readings: list[MonitorReading] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "keyword-scoring"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        request = messages[-1].text
        proposed_action = read_element(PROPOSED_ACTION_PATTERN, request)
        score = score_by_keyword(
            html.unescape(proposed_action),
            suspicion_by_keyword=self.suspicion_by_keyword,
        )
        self.readings.append(
            MonitorReading(
                trajectory=read_element(TRAJECTORY_PATTERN, request),
                proposed_action=proposed_action,
                suspicion=score.value,
                reason=score.explanation,
            ),
        )
        score_text = round(score.value * 100)
        reply = f"<reasoning>{score.explanation}</reasoning>\n<score>{score_text}</score>"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return self._generate(messages, stop=stop)


@dataclass(frozen=True, kw_only=True)
class RenderingMonitor(MonitorTwins):
    """The twins of an `LLMMonitor` whose model scores what it reads by keyword.

    The monitor's model receives exactly what an `LLMMonitor` with the same
    view sends, and `readings` keeps the trajectory and the proposed action
    of each request. The score comes from the first keyword the proposed
    action holds, so an action the monitor cannot read cannot be flagged.
    """

    view: View = field(default_factory=View)
    suspicion_by_keyword: Mapping[str, float] = field(default_factory=dict)
    readings: list[MonitorReading] = field(init=False)
    monitor: hyoshigi.Monitor = field(init=False)
    async_monitor: hyoshigi.AsyncMonitor = field(init=False)

    def __post_init__(self) -> None:
        model = KeywordScoringChatModel(suspicion_by_keyword=self.suspicion_by_keyword)
        twins = LLMMonitor(model=model, view=self.view, highest_score=100).twins
        object.__setattr__(self, "readings", model.readings)
        object.__setattr__(self, "monitor", twins.monitor)
        object.__setattr__(self, "async_monitor", twins.async_monitor)
        super().__post_init__()

    def find_reading(self, *, tool_name: str) -> MonitorReading:
        """Return the first reading whose proposed action calls `tool_name`."""
        return next(
            reading for reading in self.readings if f'name="{tool_name}"' in reading.proposed_action
        )


def read_tagged_entries(rendering: str, *, tag: str) -> list[str]:
    """Return the unescaped content of every `tag` element in a rendering, in order."""
    pattern = re.compile(rf"<{tag}(?: [^<>]*)?>(.*?)</{tag}>", re.DOTALL)
    return [html.unescape(content) for content in pattern.findall(rendering)]


@dataclass(frozen=True, kw_only=True)
class GatedRater(hyoshigi.AsyncMonitor):
    """Holds each rating until `release` is set, and sets `judging` when one begins."""

    judging: asyncio.Event
    release: asyncio.Event

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        self.judging.set()
        await self.release.wait()
        return score_by_keyword("", suspicion_by_keyword={})


@dataclass(frozen=True, kw_only=True)
class CalmRater(hyoshigi.Monitor):
    """Rates every action as benign."""

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        return score_by_keyword("", suspicion_by_keyword={})


@dataclass(frozen=True, kw_only=True)
class GatedMonitor(MonitorTwins):
    """Twins whose async rating waits for `release`, and sets `judging` when one begins.

    A test can then act while a rating is in flight, such as closing its step.
    """

    judging: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    monitor: hyoshigi.Monitor = field(init=False)
    async_monitor: hyoshigi.AsyncMonitor = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "monitor", CalmRater())
        object.__setattr__(
            self,
            "async_monitor",
            GatedRater(judging=self.judging, release=self.release),
        )
        super().__post_init__()


@dataclass(kw_only=True)
class ScriptedGeneration:
    """hyoshigi's generative model on both sides, answering every prompt with one outcome.

    `sides` records which side each call came from, and `requests` the text of
    each prompt's content.
    """

    outcome: GenerationOutcome
    sides: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)

    def record_prompt(self, prompt: Prompt, *, side: str) -> GenerationOutcome:
        self.sides.append(side)
        self.requests.append(
            "".join(part.text for part in prompt.content if isinstance(part, TextPart)),
        )
        return self.outcome

    def generate_sync(self, prompt: Prompt) -> GenerationOutcome:
        return self.record_prompt(prompt, side="sync")

    async def generate(self, prompt: Prompt) -> GenerationOutcome:
        return self.record_prompt(prompt, side="async")

    def build_twins(self) -> MonitorTwins:
        """Return hyoshigi's LLM monitor twins over this model, as a user's own monitor."""
        return MonitorTwins(
            monitor=hyoshigi.LLMMonitor(
                model=GenerativeModel(name="scripted", generate=self.generate_sync),
            ),
            async_monitor=hyoshigi.AsyncLLMMonitor(
                model=AsyncGenerativeModel(name="scripted", generate=self.generate),
            ),
        )
