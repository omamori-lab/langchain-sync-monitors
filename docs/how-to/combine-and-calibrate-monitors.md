# Combine and calibrate monitors

This guide shows how to wrap a monitor in `RepeatedMonitor`,
`CalibratedMonitor` and `CascadeMonitor`, how to collect honest scores from
`monitor_log`, and how to set thresholds that mean something.

[TOC]

## Set every threshold yourself

Every control protocol compares the monitor's suspicion with a threshold. Each
threshold has a default, 0.6 to defer, block or flag and 0.9 to audit, and a
protocol built with a default warns. This is the warning Auto Mode gives:

```text
UncalibratedThresholdWarning: block_threshold uses the uncalibrated default 0.6. Suspicion scores differ between monitors and agents, so a fixed default may block honest work or miss attacks. Run the agent on honest tasks, collect the suspicion scores in monitor_log, and set the threshold at a high percentile of them (for example the 98th), or wrap the monitor in CalibratedMonitor so that its scores are honest-run percentiles. Pass block_threshold explicitly to silence this warning.
```

Treat the defaults as placeholders. Each monitor spreads its scores
differently, and an agent's honest work looks more or less unusual depending
on its task and tools. Control evaluations set each threshold from the scores
the monitor gives to honest runs, so that it spends a known audit budget
[@greenblatt2023aicontrol; @bhatt2025ctrlz].

A threshold may be any real number from 0 to 1, a `Fraction` or a `Decimal`
included. It is checked exactly, before it becomes a float, so a `Decimal`
just above 1 is refused rather than rounded into range. A `bool`, a string
such as `"0.6"` and NaN raise `ConfigurationError`, and `-0.0` reads as 0.

Passing a number silences the warning, but only a number chosen from your own
honest runs does what it asks. There are two ways to choose one:

- Take a high percentile of the raw honest scores. This is simple, but a
  percentile that falls among tied scores flags every step in the tie.
- Wrap the monitor in `CalibratedMonitor`. Its scores are then honest-run
  percentiles, so a threshold of 0.99 flags about 1% of honest steps.

The rest of this guide takes the second way.

## The three wrappers

Each wrapper is a monitor itself, so it stacks around any monitor family and
around the other wrappers, and every protocol accepts it.

| Wrapper | What it reports | Calls per judged sample | Reason |
|---|---|---|---|
| `RepeatedMonitor(monitor=..., repeats=...)` | The mean suspicion of `repeats` calls | `repeats` times the wrapped monitor's | The most suspicious call's |
| `CalibratedMonitor(monitor=..., honest_scores=..., random_seed=...)` | The rank of the score among honest scores | The wrapped monitor's, and no more | The wrapped monitor's |
| `CascadeMonitor(fast=..., careful=..., escalate_at=...)` | The fast verdict, or the careful one when the fast suspicion reaches `escalate_at` | The fast monitor's, plus the careful monitor's for escalated samples | The verdict's own |

The wrappers never add a score to a reason. Auto Mode shows the reason to the
agent, and a number would tell the agent how close its step came to passing.

Protocols multiply these costs, because they judge every sample they draw.
Trusted Monitoring and Defer to Trusted judge one sample per step; Defer to
Resample and Auto Mode judge up to three by default. So Defer to Resample
around a `RepeatedMonitor` of three chat judge calls makes three judge calls on
an ordinary step and up to nine on a suspicious one, counting one readable
reply per call. A chat judge asks again for an unreadable reply, up to
`max_parse_retries` more times.

## Average repeated calls

A chat judge gives an integer score, and integer scores tie so often that a
small audit budget falls inside a group of tied scores
[@kaufman2025basharena; @koran2026ensemble]. Averaging several calls adds
resolution [@kaufman2025basharena]. BashArena averages three calls and
LinuxArena two, and Ctrl-Z samples its monitor several times at temperature
1.1 [@kaufman2025basharena; @tracy2026linuxarena; @bhatt2025ctrlz]. For
ensembles of different monitors, Koran et al. found a plain mean as good as
other aggregators, and found that three calls of one monitor gain less than
three different monitors [@koran2026ensemble].

```python
from langchain.chat_models import init_chat_model

from langchain_sync_monitors import LLMMonitor, RepeatedMonitor

# The calls only differ when the judge samples, so its temperature is above zero.
judge_model = init_chat_model("openrouter:xiaomi/mimo-v2.6-pro", temperature=1.0)
monitor = RepeatedMonitor(monitor=LLMMonitor(model=judge_model), repeats=3)
```

Give the model no LangChain response cache, which would answer every repeat
with a copy of the first call. The mean is rounded to 12 decimal places, so
that equal means tie exactly instead of differing in their last bits.
Under `ainvoke()` the calls run concurrently, and one that fails cancels the
others; under `invoke()` they run one after another. `repeats` below 1 raises
`ConfigurationError`.

## Stack the wrappers in order

Average first, then calibrate: `CalibratedMonitor` goes outside
`RepeatedMonitor`. The calibrated monitor ranks exactly the score its honest
scores measured, here the mean of three calls, and the mean adds the
resolution that ranking needs. The other way round would average percentiles,
and a mean of percentiles is not itself a percentile.

```mermaid
flowchart TD
    step["Proposed step"] --> calls["Three LLMMonitor scores"]
    calls --> mean["RepeatedMonitor: the mean"]
    mean --> rank["CalibratedMonitor: its rank"]
    rank --> verdict["An honest-run percentile"]
```

## Collect honest scores

Honest scores are the scores your monitor gives to steps of runs you trust.
Collect them with exactly the monitor you will calibrate: the same model,
prompt, view, number of repeats and temperature.

```mermaid
flowchart TD
    tasks["Honest tasks"] --> run["Trusted Monitoring runs"]
    run --> log[("monitor_log")]
    log --> scores["Honest scores"]
    scores --> calibrate["CalibratedMonitor"]
```

Run the agent under `TrustedMonitoring`, which never changes a step, so the
run goes as it would unmonitored and every step is judged once. A protocol
that resamples would judge extra samples only for suspicious steps, and skew
the scores. Keep only the records of the agent and the monitor you calibrate:
a record names both, and a Deep Agent's log also holds its subagents' records.

```python
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_sync_monitors import (
    LLMMonitor,
    MonitorMiddleware,
    RepeatedMonitor,
    TrustedMonitoring,
)


def read_file(path: str) -> str:
    """Read a file from the workspace."""
    return Path(path).read_text(encoding="utf-8")


HONEST_TASKS = [
    "Summarise q3.md for the team.",
    "List the open questions in notes.md.",
]

# Exactly the monitor you will calibrate.
judge_model = init_chat_model("openrouter:xiaomi/mimo-v2.6-pro", temperature=1.0)
monitor = RepeatedMonitor(monitor=LLMMonitor(model=judge_model), repeats=3)
agent = create_agent(
    "openrouter:deepseek/deepseek-v4.1-flash",
    tools=[read_file],
    middleware=[
        MonitorMiddleware(monitor=monitor, protocol=TrustedMonitoring(flag_threshold=0.9)),
    ],
)

honest_samples = []
for task in HONEST_TASKS:
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    # Keep the main agent's samples under this monitor; skip subagents' records.
    honest_samples.extend(
        sample
        for record in result["monitor_log"]
        if record["agent"] == "main" and record["monitor"] == "monitor"
        for sample in record["samples"]
    )

honest_scores = [sample["suspicion"] for sample in honest_samples]
Path("honest_scores.json").write_text(json.dumps(honest_scores), encoding="utf-8")

# Read the reasons behind the five highest honest scores before you calibrate.
for sample in sorted(honest_samples, key=lambda sample: sample["suspicion"])[-5:]:
    print(sample["suspicion"], sample["reason"])
```

Each run here has no checkpointer, so its log holds only its own steps. On a
checkpointed thread, `monitor_log` holds every turn of the thread, so collect
it once per thread.

Read the highest honest scores before you calibrate. A chat judge whose reply
could not be read gives 1.0 with the reason "The monitor gave no readable
score, so the step is treated as suspicious.". That score measures nothing
about the step, and a 1.0 in the honest set ties with every later 1.0: a step
that fails closed, and a step the judge genuinely scores at the top of its
scale. Either then lands at a random place among the ties instead of above
every honest score. Fix the cause, such as too small a `max_tokens`, and
collect again.

Collect more scores than you think you need. With `n` honest scores, a step
that scores above all of them is sure to pass a threshold `t` only when
`n / (n + 1)` is at or above `t`, which for 0.99 takes at least 99 scores. A
thousand make the flagged share stable, as [What a calibrated threshold
means](#what-a-calibrated-threshold-means) explains.

## Calibrate the monitor

Wrap the same monitor in `CalibratedMonitor` with the honest scores, then set
thresholds as shares of honest steps:

```python
import json
from pathlib import Path

from langchain.chat_models import init_chat_model

from langchain_sync_monitors import (
    CalibratedMonitor,
    LLMMonitor,
    MonitorMiddleware,
    RepeatedMonitor,
    TrustedMonitoring,
)

honest_scores = json.loads(Path("honest_scores.json").read_text(encoding="utf-8"))
judge_model = init_chat_model("openrouter:xiaomi/mimo-v2.6-pro", temperature=1.0)
monitor = CalibratedMonitor(
    monitor=RepeatedMonitor(monitor=LLMMonitor(model=judge_model), repeats=3),
    honest_scores=honest_scores,
    # A fixed seed makes the tie-breaking replayable.
    random_seed=2026,
)
# Flag about 1% of honest steps.
middleware = MonitorMiddleware(
    monitor=monitor,
    protocol=TrustedMonitoring(flag_threshold=0.99),
)
```

Every other protocol's thresholds read the same way: with a calibrated
monitor, a defer threshold of 0.98 sends about 2% of honest steps to
resampling. `honest_scores` can be a list or any other iterable, such as a
generator, which the constructor reads once. It must hold at least one score,
each from 0 to 1 and none NaN, or the constructor raises `ConfigurationError`.

## What a calibrated threshold means

`CalibratedMonitor` replaces the wrapped monitor's score with its smoothed
conformal rank among the `n` honest scores [@vovk2005algorithmic]:

```text
calibrated suspicion = (below + U * (tied + 1)) / (n + 1)
```

The step takes one of `n + 1` places in the sorted list of honest scores.
`below` counts the honest scores lower than the step's score, and `tied`
counts those equal to it. `U` is a random number from 0 up to 1, which spreads
the step evenly over its own place and the places of the honest scores it ties
with. Breaking ties at random follows Makins et al. and LinuxArena
[@makins2026multiagent; @tracy2026linuxarena].

The guarantee holds when the step's raw score is exchangeable with the honest
scores: the step comes from the same kind of honest run, scored by the same
monitor with the same settings. Its calibrated suspicion is then spread evenly
from 0 to 1, so a threshold `t` flags a share `1 - t` of such steps: 0.99 flags
1% of honest steps, and 0.95 flags 5%.

That share is exact on average over the honest scores you might have
collected. For the one set you did collect, it varies around `1 - t`. Its
standard deviation is about `sqrt(t * (1 - t) / n)`, the spread of a share
estimated from `n` draws; this approximation is ours, not a cited result, and
a simulation matches it. At `t = 0.99` it is about 0.3 percentage points for a
thousand honest scores, and about 1 point for a hundred.

| Consequence | What it means |
|---|---|
| Scores above every honest score | Such a step lands between `n / (n + 1)` and 1, never at 1. With 49 honest scores it passes a threshold of 0.99 only half the time; with 99 or more, always. |
| Ties | A step that ties with many honest scores lands anywhere across their places. If 400 of 500 honest scores are 0, a step that scores 0 lands anywhere from 0 to about 0.8. |
| What calibration cannot do | It does not make the monitor better at telling attacks from honest work. An attack scores high only if its raw score is above most honest scores. |

## Fix the tie-breaking with a seed

Without `random_seed`, `U` is drawn afresh on every call, so a step whose
score ties with honest scores can land on either side of a threshold from one
run to the next.

With `random_seed`, `U` comes from the seed and the step's content: the task
author, the type and text of each earlier message, and the proposal with all
its parts. The earlier messages' tool calls and ids are left out, since the
monitor's own messages get fresh random ids on every run; two steps whose
histories differ only in tool calls get the same draw. The same step always gets the same draw, in every process, so the same
raw score always gives the same calibrated suspicion, and an evaluation can be
replayed exactly.

## Put a fast monitor in front of a careful one

`CascadeMonitor` runs a fast monitor on every step and sends a step to a
careful one only when the fast suspicion reaches `escalate_at`. Below it, the
fast verdict stands; at or above it, the careful verdict, reason included,
replaces it. This follows the two stages of Claude Code's auto mode, a fast
filter in front of a reasoning classifier [@hughes2026automode]. Hua et
al.'s best two-monitor protocol calls the expensive monitor only when the
cheap one's score falls in an uncertain range [@hua2025combining];
`CascadeMonitor` uses the simpler rule of one threshold.

```mermaid
flowchart TD
    step["Proposed step"] --> fast["Fast: a decision model"]
    fast -- "below escalate_at" --> keep["The fast verdict"]
    fast -- "at or above it" --> careful["Careful: a repeated chat judge"]
    careful --> replace["The careful verdict"]
```

Calibrate each stage on its own honest scores, collected as above with that
stage as the monitor. Collect the careful stage's scores on every honest step,
not only on the ones the fast stage would escalate.

```python
import json
from pathlib import Path

from langchain.chat_models import init_chat_model

from langchain_sync_monitors import (
    CalibratedMonitor,
    CascadeMonitor,
    DecisionModelMonitor,
    LLMMonitor,
    OpenRouterDecisionModel,
    RepeatedMonitor,
)

fast_scores = json.loads(Path("fast_honest_scores.json").read_text(encoding="utf-8"))
careful_scores = json.loads(Path("careful_honest_scores.json").read_text(encoding="utf-8"))
judge_model = init_chat_model("openrouter:xiaomi/mimo-v2.6-pro", temperature=1.0)

monitor = CascadeMonitor(
    # Fast: one decision model request per step.
    fast=CalibratedMonitor(
        monitor=DecisionModelMonitor(
            decision_model=OpenRouterDecisionModel(model="typesafe/jev-1.13"),
        ),
        honest_scores=fast_scores,
    ),
    # Careful: three judge calls, only for escalated steps.
    careful=CalibratedMonitor(
        monitor=RepeatedMonitor(monitor=LLMMonitor(model=judge_model), repeats=3),
        honest_scores=careful_scores,
    ),
    # Escalate about 5% of honest steps.
    escalate_at=0.95,
)
```

With both stages calibrated, `escalate_at` is a share too: 0.95 sends about
5% of honest steps to the careful stage, which bounds its cost. A threshold
`t` on the result then flags at most about a share `1 - t` of honest steps,
since only the escalated ones can reach it.

Keep `escalate_at` at or below the protocol's lowest threshold. A step the
fast stage keeps has a suspicion below `escalate_at`, so a threshold below
`escalate_at` could trigger the protocol on a step the careful stage never
read. A value of `escalate_at` outside 0 to 1, or NaN, raises
`ConfigurationError`. Unlike a protocol threshold, its type is not checked, so
pass a float.

## Related guides

- [Choose a protocol](choose-a-protocol.md) for the thresholds each protocol takes.
- [Use Defer to Resample](use-defer-to-resample.md) for the protocol whose audit budget calibration serves best.

## References
