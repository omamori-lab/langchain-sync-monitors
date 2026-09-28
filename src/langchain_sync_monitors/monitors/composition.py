"""Monitors that wrap other monitors: averaging, calibration and cascades.

Each wrapper is a `Monitor` itself, so it stacks around any monitor family, and
around other wrappers, without the control protocols noticing. A common stack
averages a few calls first, then expresses the result as an honest-run
percentile: `CalibratedMonitor(monitor=RepeatedMonitor(monitor=judge, repeats=3),
honest_scores=scores)`.
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.errors import ConfigurationError


class RepeatedMonitor(Monitor):
    """Ask the same monitor several times and report the mean suspicion.

    A single call of a chat judge gives an integer score, and integer scores tie
    so often that a low audit budget cannot be set: the budget falls inside a
    group of tied steps. Averaging repeated calls adds resolution, which is why
    Ctrl-Z averages 4 calls, BashArena 3 and LinuxArena 2
    [@bhatt2025ctrlz; @kaufman2025basharena; @tracy2026linuxarena]; a plain mean
    works as well as more elaborate aggregators [@koran2026ensemble]. The calls
    only differ when the wrapped monitor samples, so its model needs a
    temperature above zero (Ctrl-Z used 1.1).

    The reason comes from the most suspicious call, since that is the one an
    auditor needs to read. `evaluate` makes the calls concurrently.
    """

    def __init__(self, *, monitor: Monitor, repeats: int) -> None:
        """Wrap `monitor`, calling it `repeats` times for each step."""
        if repeats < 1:
            message = f"repeats must be at least 1, got {repeats}"
            raise ConfigurationError(message)
        self.monitor = monitor
        self.repeats = repeats

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Call the wrapped monitor `repeats` times concurrently and average."""
        verdicts = await asyncio.gather(
            *(self.monitor.evaluate(monitor_input) for _ in range(self.repeats)),
        )
        return build_mean_verdict(verdicts)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Call the wrapped monitor `repeats` times in turn and average."""
        verdicts = [self.monitor.evaluate_sync(monitor_input) for _ in range(self.repeats)]
        return build_mean_verdict(verdicts)


def build_mean_verdict(verdicts: Sequence[Verdict]) -> Verdict:
    """Average the suspicions and keep the reason of the most suspicious verdict."""
    most_suspicious = max(verdicts, key=lambda verdict: verdict.suspicion)
    mean_suspicion = statistics.fmean(verdict.suspicion for verdict in verdicts)
    reason = (
        f"Mean suspicion over {len(verdicts)} calls. The most suspicious call "
        f"({most_suspicious.suspicion:.2f}) said: {most_suspicious.reason}"
    )
    return Verdict(suspicion=mean_suspicion, reason=reason)


class CalibratedMonitor(Monitor):
    """Report suspicion as a percentile of the scores the monitor gave honest runs.

    Control evaluations set thresholds as percentiles of honest-run scores,
    so an audit budget of 1% means flagging the top 1% of honest steps. After
    this wrapper a threshold of 0.99 means exactly that. A score tied with some
    honest scores lands at a random point within the tie, so a threshold can
    still hit its budget when many scores are equal; this is the tie-breaking
    of Makins and Angelini and of LinuxArena
    [@makins2026multiagent; @tracy2026linuxarena].

    Collect `honest_scores` by running the wrapped monitor, with the same
    settings, on steps from runs you trust. Pass `random_seed` for reproducible
    tie-breaking.
    """

    def __init__(
        self,
        *,
        monitor: Monitor,
        honest_scores: Sequence[float],
        random_seed: int | None = None,
    ) -> None:
        """Wrap `monitor` and rank its scores among `honest_scores`."""
        if not honest_scores:
            message = "honest_scores must hold at least one score"
            raise ConfigurationError(message)
        invalid_scores = [
            score for score in honest_scores if math.isnan(score) or not 0.0 <= score <= 1.0
        ]
        if invalid_scores:
            message = f"honest_scores must lie between 0 and 1, got {invalid_scores[:3]!r}"
            raise ConfigurationError(message)
        self.monitor = monitor
        self.sorted_honest_scores = sorted(honest_scores)
        self.random = random.Random(random_seed)

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step with the wrapped monitor, then rank the score."""
        return self.build_calibrated_verdict(await self.monitor.evaluate(monitor_input))

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step with the wrapped monitor without an event loop, then rank it."""
        return self.build_calibrated_verdict(self.monitor.evaluate_sync(monitor_input))

    def build_calibrated_verdict(self, verdict: Verdict) -> Verdict:
        """Replace the raw suspicion with its honest-run percentile.

        The percentile counts the honest scores below the raw score, plus a
        uniformly random share of the honest scores equal to it.
        """
        scores = self.sorted_honest_scores
        below = bisect_left(scores, verdict.suspicion)
        tied = bisect_right(scores, verdict.suspicion) - below
        percentile = (below + self.random.uniform(0.0, tied)) / len(scores)
        reason = (
            f"Raw suspicion {verdict.suspicion:.3f} is at the {percentile:.3f} "
            f"percentile of honest runs. {verdict.reason}"
        )
        return Verdict(suspicion=percentile, reason=reason)


class CascadeMonitor(Monitor):
    """Run a fast monitor on every step and a careful one only when it is needed.

    Steps the fast monitor scores below `escalate_at` keep its verdict; the
    rest go to the careful monitor, whose verdict replaces it. This follows the
    two stages of Claude Code's auto mode, a fast filter tuned for recall in
    front of a reasoning classifier [@hughes2026automode], and the idea of
    calling an expensive monitor only when a cheap one asks for it
    [@hua2025combining]. A decision model makes a good fast stage and a chat
    judge a good careful one.
    """

    def __init__(self, *, fast: Monitor, careful: Monitor, escalate_at: float) -> None:
        """Escalate from `fast` to `careful` when the fast suspicion reaches `escalate_at`."""
        if math.isnan(escalate_at) or not 0.0 <= escalate_at <= 1.0:
            message = f"escalate_at must lie between 0 and 1, got {escalate_at!r}"
            raise ConfigurationError(message)
        self.fast = fast
        self.careful = careful
        self.escalate_at = escalate_at

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score with the fast monitor, and with the careful one if that escalates."""
        fast_verdict = await self.fast.evaluate(monitor_input)
        if not self.is_escalated(fast_verdict):
            return fast_verdict
        careful_verdict = await self.careful.evaluate(monitor_input)
        return build_escalated_verdict(fast_verdict=fast_verdict, careful_verdict=careful_verdict)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score with the fast monitor, and with the careful one if that escalates."""
        fast_verdict = self.fast.evaluate_sync(monitor_input)
        if not self.is_escalated(fast_verdict):
            return fast_verdict
        careful_verdict = self.careful.evaluate_sync(monitor_input)
        return build_escalated_verdict(fast_verdict=fast_verdict, careful_verdict=careful_verdict)

    def is_escalated(self, fast_verdict: Verdict) -> bool:
        """Tell whether the fast verdict is suspicious enough for the careful monitor."""
        return fast_verdict.suspicion >= self.escalate_at


def build_escalated_verdict(*, fast_verdict: Verdict, careful_verdict: Verdict) -> Verdict:
    """Keep the careful suspicion, and say in the reason why the step was escalated."""
    reason = (
        f"The fast monitor scored {fast_verdict.suspicion:.2f}, so the careful monitor "
        f"judged the step: {careful_verdict.reason}"
    )
    return Verdict(suspicion=careful_verdict.suspicion, reason=reason)
