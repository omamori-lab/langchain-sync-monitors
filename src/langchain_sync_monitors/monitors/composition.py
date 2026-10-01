"""Monitors that wrap other monitors: averaging, calibration and cascades.

Each wrapper is a `Monitor` itself, so it stacks around any monitor family, and
around other wrappers, without the control protocols noticing. A common stack
averages a few calls first, then expresses the result as an honest-run
percentile: `CalibratedMonitor(monitor=RepeatedMonitor(monitor=judge,
ensemble_size=3), honest_scores=scores)`.

The wrappers change the suspicion and pass on the wrapped monitor's reason word
for word. Auto Mode shows the reason to the agent, so it carries no scores:
the numbers stay in the step's record, which is for the auditor.
"""

from __future__ import annotations

import random
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence

from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.contracts import Monitor, MonitorInput, Verdict
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.options import (
    check_instance_option,
    describe_option_value,
    read_count_option,
    read_integer_option,
)
from langchain_sync_monitors.thresholds import read_threshold_value
from langchain_sync_monitors.transcript import render_proposal_for_audit


class RepeatedMonitor(Monitor):
    """Ask the same monitor several times and report the mean suspicion.

    This is a homogeneous ensemble: the mean of independent calls of one
    monitor [@koran2026ensemble]. `ensemble_size` is the number of calls,
    which LinuxArena calls the ensemble size (appendix C)
    [@tracy2026linuxarena].

    A single call of a chat judge gives an integer score, and integer scores tie
    so often that a low audit budget cannot be set: the budget falls inside a
    group of tied steps. Averaging repeated calls adds resolution
    [@kaufman2025basharena]. BashArena averages three calls and LinuxArena
    two, and Ctrl-Z samples its monitor several times at temperature 1.1
    [@kaufman2025basharena; @tracy2026linuxarena; @bhatt2025ctrlz]; a plain
    mean works as well as more elaborate aggregators [@koran2026ensemble]. The
    calls only differ when the wrapped monitor samples, so its model needs a
    temperature above zero, and no LangChain response cache, which would
    answer every call after the first with a copy of it.

    The reason is the most suspicious call's own reason, since that is the one
    an auditor needs to read. `evaluate` makes the calls concurrently, and one
    failed call cancels the others.
    """

    def __init__(self, *, monitor: Monitor, ensemble_size: int) -> None:
        """Wrap `monitor`, calling it `ensemble_size` times for each step.

        `ensemble_size` must be a whole number of at least 1; anything else, or
        a `monitor` that is not a `Monitor`, raises `ConfigurationError`.
        """
        check_instance_option(monitor, option_type=Monitor, parameter_name="monitor")
        self.monitor = monitor
        self.ensemble_size = read_count_option(
            ensemble_size, parameter_name="ensemble_size", minimum=1
        )

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Call the wrapped monitor `ensemble_size` times concurrently and average."""
        verdicts = await run_concurrently(
            self.monitor.evaluate(monitor_input) for _ in range(self.ensemble_size)
        )
        return build_mean_verdict(verdicts)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Call the wrapped monitor `ensemble_size` times in turn and average."""
        verdicts = [self.monitor.evaluate_sync(monitor_input) for _ in range(self.ensemble_size)]
        return build_mean_verdict(verdicts)


MEAN_DECIMAL_PLACES = 12
"""Decimal places a mean suspicion is rounded to, so equal means compare equal.

Scores such as 0.1, 0.0, 0.2 and 0.0, 0.0, 0.3 have the same mean, but their
float averages differ in the last bits. Unrounded, `CalibratedMonitor` would
order such steps by rounding error instead of breaking the tie at random.
"""


def build_mean_verdict(verdicts: Sequence[Verdict]) -> Verdict:
    """Average the suspicions and keep the reason of the most suspicious verdict."""
    most_suspicious = max(verdicts, key=lambda verdict: verdict.suspicion)
    mean_suspicion = statistics.fmean(verdict.suspicion for verdict in verdicts)
    return Verdict(
        suspicion=round(mean_suspicion, MEAN_DECIMAL_PLACES),
        reason=most_suspicious.reason,
    )


def read_honest_scores(honest_scores: object) -> list[float]:
    """Return the honest scores as floats, raising `ConfigurationError` unless they are usable.

    They must be an iterable other than a string, holding at least one score,
    and each score is read as a threshold is: a number from 0 to 1, not a
    `bool` and not NaN. A refused score is named by its position.
    """
    if isinstance(honest_scores, str | bytes) or not isinstance(honest_scores, Iterable):
        message = (
            "honest_scores must be an iterable of numbers between 0 and 1, "
            f"got {describe_option_value(honest_scores)}"
        )
        raise ConfigurationError(message)
    scores = [
        read_threshold_value(score, parameter_name=f"honest_scores[{position}]")
        for position, score in enumerate(honest_scores)
    ]
    if not scores:
        message = "honest_scores must hold at least one score"
        raise ConfigurationError(message)
    return scores


def render_tie_break_key(monitor_input: MonitorInput) -> str:
    """Render what a step says, and nothing random, as the key of its tie-break draw.

    Message ids are left out, because the monitor's own messages get fresh
    random ids on every run.
    """
    history = "\n".join(f"{message.type}: {message.text}" for message in monitor_input.history)
    proposal = render_proposal_for_audit(monitor_input.proposal)
    return f"{monitor_input.task_author}\n{history}\n{proposal}"


class CalibratedMonitor(Monitor):
    """Report suspicion as the rank of a step's score among the scores of honest runs.

    Control evaluations set thresholds as percentiles of honest-run scores, so
    an audit budget of 1% means flagging for review the top 1% of honest
    steps. This wrapper reports the smoothed conformal rank of the raw score
    among the `n` honest scores [@vovk2005algorithmic]:

        (below + U * (tied + 1)) / (n + 1)

    where `below` counts the honest scores lower than the raw score, `tied`
    counts those equal to it, and `U` is uniform on [0, 1). The step takes one
    of `n + 1` rank slots, and `U` spreads it evenly over its own slot and
    those of the honest scores it ties with, the tie-breaking of Makins and
    Angelini and of LinuxArena [@makins2026multiagent; @tracy2026linuxarena].
    A score above every honest score lands between `n / (n + 1)` and 1.

    What a threshold means: when a step's raw score is exchangeable with the
    honest scores, as for a step of the same kind of honest run scored by the
    same monitor with the same settings, its calibrated suspicion is uniform
    on [0, 1]. A threshold `t` then flags exactly a share `1 - t` of such
    steps, so 0.99 flags 1% of honest steps. The share is exact on average
    over the honest scores collected; for one fixed set of `n` scores it
    varies around `1 - t` with a standard deviation of about
    `sqrt(t * (1 - t) / n)`, so collect enough of them: at `t = 0.99`, a
    thousand scores give a spread of about 0.3 percentage points.

    Collect `honest_scores` by running the wrapped monitor, with the same
    settings, on steps from runs you trust, and pass them as Python floats or
    numpy float64s: each is read as a float, so a float32 0.1 becomes
    0.10000000149011612 and no longer ties with a score of 0.1. With
    `random_seed`, `U` is derived from the seed and the step's content, so a
    step always gets the same suspicion, whatever else the instance has
    scored and however many runs share it. Without a seed, `U` is drawn afresh
    on every call. The reason is the wrapped monitor's, unchanged.
    """

    def __init__(
        self,
        *,
        monitor: Monitor,
        honest_scores: Iterable[float],
        random_seed: int | None = None,
    ) -> None:
        """Wrap `monitor` and rank its scores among `honest_scores`, which are read once.

        Reading them into a list first keeps a generator's scores: checking
        and then sorting it would consume it and leave nothing to rank among.
        Each score is read as a threshold is, so it becomes a float, and
        `random_seed` must be an integer or `None`; anything else raises
        `ConfigurationError`.
        """
        check_instance_option(monitor, option_type=Monitor, parameter_name="monitor")
        self.monitor = monitor
        self.sorted_honest_scores = sorted(read_honest_scores(honest_scores))
        self.random_seed = (
            None
            if random_seed is None
            else read_integer_option(random_seed, parameter_name="random_seed")
        )

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step with the wrapped monitor, then rank the score."""
        verdict = await self.monitor.evaluate(monitor_input)
        return self.build_calibrated_verdict(verdict, monitor_input=monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score the step with the wrapped monitor without an event loop, then rank it."""
        verdict = self.monitor.evaluate_sync(monitor_input)
        return self.build_calibrated_verdict(verdict, monitor_input=monitor_input)

    def build_calibrated_verdict(self, verdict: Verdict, *, monitor_input: MonitorInput) -> Verdict:
        """Replace the raw suspicion with its smoothed conformal rank among the honest scores."""
        scores = self.sorted_honest_scores
        # A conformal rank with ties: every honest score below counts, and the tied ones count
        # for a random share.
        below = bisect_left(scores, verdict.suspicion)
        tied = bisect_right(scores, verdict.suspicion) - below
        tie_share = self.draw_tie_share(monitor_input)
        percentile = (below + tie_share * (tied + 1)) / (len(scores) + 1)
        return Verdict(suspicion=percentile, reason=verdict.reason)

    def draw_tie_share(self, monitor_input: MonitorInput) -> float:
        """Draw `U`, uniform on [0, 1), from the seed and the step when a seed is set.

        A string seed goes through SHA-512 in Python's `random`, so the draw is
        the same in every process.
        """
        if self.random_seed is None:
            return random.random()
        key = f"{self.random_seed}\n{render_tie_break_key(monitor_input)}"
        return random.Random(key).random()


class CascadeMonitor(Monitor):
    """Run a fast monitor on every step and a careful one only when it is needed.

    Steps the fast monitor scores below `escalation_threshold` keep its
    verdict; the rest go to the careful monitor, whose verdict, reason
    included, replaces it. This follows the two stages of Claude Code's auto
    mode, a fast filter tuned for recall in front of a reasoning classifier
    [@hughes2026automode], and the idea of calling an expensive monitor only
    when a cheap one asks for it [@hua2025combining]. A decision model makes a
    good fast stage and a chat judge a good careful one.
    """

    def __init__(self, *, fast: Monitor, careful: Monitor, escalation_threshold: float) -> None:
        """Escalate from `fast` to `careful` once the fast suspicion reaches the threshold.

        `escalation_threshold` is read as any threshold is: a number from 0 to
        1, kept as a float. Anything else, or a stage that is not a `Monitor`,
        raises `ConfigurationError`.
        """
        check_instance_option(fast, option_type=Monitor, parameter_name="fast")
        check_instance_option(careful, option_type=Monitor, parameter_name="careful")
        self.fast = fast
        self.careful = careful
        self.escalation_threshold = read_threshold_value(
            escalation_threshold, parameter_name="escalation_threshold"
        )

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        """Score with the fast monitor, and with the careful one if that escalates."""
        fast_verdict = await self.fast.evaluate(monitor_input)
        if not self.is_escalated(fast_verdict):
            return fast_verdict
        return await self.careful.evaluate(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        """Score with the fast monitor, and with the careful one if that escalates."""
        fast_verdict = self.fast.evaluate_sync(monitor_input)
        if not self.is_escalated(fast_verdict):
            return fast_verdict
        return self.careful.evaluate_sync(monitor_input)

    def is_escalated(self, fast_verdict: Verdict) -> bool:
        """Tell whether the fast verdict is suspicious enough for the careful monitor."""
        return fast_verdict.suspicion >= self.escalation_threshold
