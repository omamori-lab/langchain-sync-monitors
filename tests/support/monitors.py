"""A fake monitor whose verdict depends on what a proposal says, not on call order."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.model_calls import build_internal_call_config
from langchain_sync_monitors.transcript import render_proposal_for_audit

BENIGN_SUSPICION = 0.1
JUDGE_SOURCE = "fake_monitor"


@dataclass(kw_only=True)
class KeywordMonitor(Monitor):
    """Scores a proposal by the first keyword its audit rendering contains.

    Keying the verdict on the proposal keeps concurrent sampling deterministic,
    since `asyncio.gather` does not preserve call order. With `judge_model`,
    every verdict also makes one tagged internal model call, as a real monitor
    would.
    """

    suspicion_by_keyword: Mapping[str, float] = field(default_factory=dict)
    judge_model: BaseChatModel | None = None
    inputs: list[MonitorInput] = field(default_factory=list)

    def score(self, monitor_input: MonitorInput) -> Verdict:
        self.inputs.append(monitor_input)
        rendered = render_proposal_for_audit(monitor_input.proposal)
        for keyword, suspicion in self.suspicion_by_keyword.items():
            if keyword in rendered:
                return Verdict(suspicion=suspicion, reason=f"mentions {keyword}")
        return Verdict(suspicion=BENIGN_SUSPICION, reason="nothing suspicious")

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            config = build_internal_call_config(source=JUDGE_SOURCE)
            await self.judge_model.ainvoke("Judge this step.", config=config)
        return self.score(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            config = build_internal_call_config(source=JUDGE_SOURCE)
            self.judge_model.invoke("Judge this step.", config=config)
        return self.score(monitor_input)
