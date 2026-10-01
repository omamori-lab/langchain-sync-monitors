"""Fake monitors whose verdict depends on what a proposal says, not on call order."""

from __future__ import annotations

import asyncio
import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from langchain_sync_monitors.contracts import Monitor, MonitorInput, MonitorView, Verdict
from langchain_sync_monitors.model_calls import build_internal_call_config
from langchain_sync_monitors.transcript import (
    render_proposal_for_audit,
    render_proposed_step,
    render_transcript,
)

BENIGN_SUSPICION = 0.1
JUDGE_SOURCE = "fake_monitor"


@dataclass(kw_only=True)
class KeywordMonitor(Monitor):
    """Scores a proposal by the first keyword its audit rendering contains.

    Keying the verdict on the proposal keeps concurrent sampling deterministic,
    since `asyncio.gather` does not preserve call order. With `judge_model`,
    every verdict also makes one tagged internal model call, as a real monitor
    would. With `judge_tags`, that call carries tags of its own, which replace
    the ones it would inherit, as a user's monitor may set them.
    """

    suspicion_by_keyword: Mapping[str, float] = field(default_factory=dict)
    judge_model: BaseChatModel | None = None
    judge_tags: Sequence[str] = ()
    inputs: list[MonitorInput] = field(default_factory=list)

    def build_judge_config(self) -> RunnableConfig:
        config = build_internal_call_config(source=JUDGE_SOURCE)
        if self.judge_tags:
            config["tags"] = list(self.judge_tags)
        return config

    def score(self, monitor_input: MonitorInput) -> Verdict:
        self.inputs.append(monitor_input)
        rendered = render_proposal_for_audit(monitor_input.proposal)
        for keyword, suspicion in self.suspicion_by_keyword.items():
            if keyword in rendered:
                return Verdict(suspicion=suspicion, reason=f"mentions {keyword}")
        return Verdict(suspicion=BENIGN_SUSPICION, reason="nothing suspicious")

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            await self.judge_model.ainvoke("Judge this step.", config=self.build_judge_config())
        return self.score(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            self.judge_model.invoke("Judge this step.", config=self.build_judge_config())
        return self.score(monitor_input)


@dataclass(frozen=True, kw_only=True)
class MonitorReading:
    """What a rendering monitor read for one judgement, and the verdict it gave."""

    monitor_input: MonitorInput
    transcript: str
    proposed_step: str
    verdict: Verdict


@dataclass(kw_only=True)
class RenderingMonitor(Monitor):
    """Reads each input through its view, as `LLMMonitor` fills its prompt, and scores that text.

    The transcript and the proposed step are rendered with the library's own
    `render_transcript` and `render_proposed_step`, so `readings` holds exactly
    the text a chat judge with the same view would receive. The verdict comes
    from the first keyword the rendered proposed step contains, so a step the
    view hides cannot be flagged.
    """

    view: MonitorView = field(default_factory=MonitorView)
    suspicion_by_keyword: Mapping[str, float] = field(default_factory=dict)
    readings: list[MonitorReading] = field(default_factory=list)

    def score(self, monitor_input: MonitorInput) -> Verdict:
        transcript = render_transcript(
            monitor_input.history,
            view=self.view,
            task_author=monitor_input.task_author,
        )
        proposed_step = render_proposed_step(monitor_input.proposal, view=self.view)
        verdict = Verdict(suspicion=BENIGN_SUSPICION, reason="nothing suspicious")
        for keyword, suspicion in self.suspicion_by_keyword.items():
            if keyword in proposed_step:
                verdict = Verdict(suspicion=suspicion, reason=f"mentions {keyword}")
                break
        self.readings.append(
            MonitorReading(
                monitor_input=monitor_input,
                transcript=transcript,
                proposed_step=proposed_step,
                verdict=verdict,
            ),
        )
        return verdict

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return self.score(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return self.score(monitor_input)

    def find_reading(self, *, tool_name: str) -> MonitorReading:
        """Return the first reading whose proposal calls `tool_name`."""
        return next(
            reading
            for reading in self.readings
            if any(call["name"] == tool_name for call in reading.monitor_input.proposal.tool_calls)
        )


def read_tagged_entries(transcript: str, *, tag: str) -> list[str]:
    """Return the unescaped content of every `tag` entry in a rendered transcript, in order."""
    pattern = re.compile(rf"<{tag}(?: [^<>]*)?>(.*?)</{tag}>", re.DOTALL)
    return [html.unescape(content) for content in pattern.findall(transcript)]


@dataclass(kw_only=True)
class GatedMonitor(Monitor):
    """Holds each async judgement until `release` is set, and sets `judging` when one begins.

    A test can then act while a judgement is in flight, such as closing its step.
    """

    judging: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        self.judging.set()
        await self.release.wait()
        return self.evaluate_sync(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return Verdict(suspicion=BENIGN_SUSPICION, reason="nothing suspicious")
