---
heading_numbers: false
---

# langchain-sync-monitors

This page introduces the library and maps the rest of the documentation.

langchain-sync-monitors puts a monitor on every step your LangChain agent or
Deep Agent proposes, and a control protocol that decides what runs.

[![The untrusted model proposes a step. The monitor, drawn as an omamori charm, reads it through its view and scores its suspicion from 0 to 1. The control protocol chooses one of five outcomes, from the lightest touch to the strongest: allow it, resample it, defer to the trusted model, block it and tell the agent why, or halt. All of this happens inside the middleware, before any of the agent's own tools run; a brushed line marks that boundary. A provider's built-in tools, such as web search, run inside the model call, above that line. A committed step reaches the tools and a halted run ends, and either way one record joins monitor_log. Tool results come back, and the next model call is a new step.](assets/diagrams/monitored-step-light.svg#only-light)](assets/diagrams/monitored-step-light.svg "Open the diagram at full size")
[![The untrusted model proposes a step. The monitor, drawn as an omamori charm, reads it through its view and scores its suspicion from 0 to 1. The control protocol chooses one of five outcomes, from the lightest touch to the strongest: allow it, resample it, defer to the trusted model, block it and tell the agent why, or halt. All of this happens inside the middleware, before any of the agent's own tools run; a brushed line marks that boundary. A provider's built-in tools, such as web search, run inside the model call, above that line. A committed step reaches the tools and a halted run ends, and either way one record joins monitor_log. Tool results come back, and the next model call is a new step.](assets/diagrams/monitored-step-dark.svg#only-dark)](assets/diagrams/monitored-step-dark.svg "Open the diagram at full size")

## Why this exists

An agent with tools can do harm in a single step: post a credential, delete a
file, run a command. AI control research guards against this with a monitor,
a second model that scores each step, and a protocol that acts on the score.
This library brings those protocols, from AI Control, Ctrl-Z and Claude Code's
auto mode, to `create_agent` and `create_deep_agent` as one middleware.
"Sync" means synchronous in the control sense: a step waits for its verdict,
so a suspicious step can be stopped before it runs.

## Install

```console
pip install "langchain-sync-monitors[openrouter,deepagents]"
```

<!-- release-check: the text below describes an unreleased package. scripts/check-release.sh refuses a release while this comment is here; rewrite that text for the release, then delete this comment. -->

The package is not on PyPI yet. Until the first release, install it from
GitHub:

```console
pip install "langchain-sync-monitors[openrouter,deepagents] @ git+https://github.com/omamori-lab/langchain-sync-monitors"
```

| Extra | Adds | Needed for |
|---|---|---|
| `openrouter` | `langchain-openrouter` | model strings such as `"openrouter:xiaomi/mimo-v2.6-pro"` |
| `deepagents` | `deepagents` | `create_deep_agent` and `monitor_subagents` |
| `typesafe` | `langchain-typesafe` | `TypeSafeDecisionModel` |

## Quick start

```python
from pathlib import Path

from langchain.agents import create_agent
from langchain_sync_monitors import (
    DeferToResample,
    DeferToTrustedModel,
    LLMMonitor,
    MonitorMiddleware,
)


def read_file(path: str) -> str:
    """Read a file from the workspace."""
    return Path(path).read_text()


monitor = MonitorMiddleware(
    monitor=LLMMonitor(model="openrouter:xiaomi/mimo-v2.6-pro"),
    protocol=DeferToResample(
        fallback=DeferToTrustedModel(trusted_model="openrouter:xiaomi/mimo-v2.6-flash"),
        defer_threshold=0.6,
        audit_threshold=0.9,
    ),
)
agent = create_agent(
    "openrouter:deepseek/deepseek-v4.1-flash", tools=[read_file], middleware=[monitor]
)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarise q3.md."}]})
for record in result["monitor_log"]:
    print(record["step_number"], record["outcome"], record["flagged"])
```

Three models play three roles: the untrusted agent does the work, the monitor
judges each step, and the trusted model writes a step when every sample looks
suspicious. Each is a parameter, and any LangChain chat model works; the
library picks none for you. Put the monitor last in the middleware list.
Thresholds share one scale from 0 to 1 and warn until you set them; calibrate
them on honest runs of your own agent. Each step leaves one record in
`monitor_log`, and [Read the monitor log](how-to/read-the-monitor-log.md)
explains its fields. The [tutorial](tutorials/first-monitored-agent.md) builds
this agent step by step.

## Pick a protocol

[![What each protocol does with a step the monitor finds suspicious; a step below the threshold runs unchanged under every protocol. Trusted Monitoring lets the step run and flags it for a person to review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to two fresh samples; if one passes, the least suspicious runs, and if none passes, the fallback writes the step, usually the trusted model. Auto Mode blocks the step and tells the agent why, and the agent tries again; after three blocks in a row, or twenty in the thread with subagents' blocks included, the fallback takes over, which by default halts the run.](assets/diagrams/protocols-light.svg#only-light)](assets/diagrams/protocols-light.svg "Open the diagram at full size")
[![What each protocol does with a step the monitor finds suspicious; a step below the threshold runs unchanged under every protocol. Trusted Monitoring lets the step run and flags it for a person to review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to two fresh samples; if one passes, the least suspicious runs, and if none passes, the fallback writes the step, usually the trusted model. Auto Mode blocks the step and tells the agent why, and the agent tries again; after three blocks in a row, or twenty in the thread with subagents' blocks included, the fallback takes over, which by default halts the run.](assets/diagrams/protocols-dark.svg#only-dark)](assets/diagrams/protocols-dark.svg "Open the diagram at full size")

| Protocol | Pick it when | Guide |
|---|---|---|
| `TrustedMonitoring` | you want to watch and flag, and change nothing the agent does | [Use Trusted Monitoring](how-to/use-trusted-monitoring.md) |
| `DeferToTrusted` | you have a trusted model and want the simplest intervention | [Use Defer to Trusted](how-to/use-defer-to-trusted.md) |
| `DeferToResample` | you want to keep the agent's own work where you can, and keep rejected samples as evidence | [Use Defer to Resample](how-to/use-defer-to-resample.md) |
| `AutoMode` | you want the agent to hear why a step was blocked and find a safer route | [Use Auto Mode](how-to/use-auto-mode.md) |

Not sure which? [Choose a protocol](how-to/choose-a-protocol.md) compares
them.

## Pick a monitor

[![Three monitor classes cover three kinds of model, each drawn as the charm. LLMMonitor asks a chat model to reason and then score the step, from 0 to 10 by default. GuardModelMonitor has a guard model label the step against a policy you write, and suspicion is the chance the label is unsafe. DecisionModelMonitor asks a decision model yes or no questions, and suspicion is the chance of yes. Each returns the same Verdict, a suspicion from 0 to 1 and a reason, so any monitor works with any control protocol. Three wrappers stack around any monitor: RepeatedMonitor averages several calls, CalibratedMonitor ranks the score among the scores of honest runs, and CascadeMonitor runs a fast monitor on every step and a careful one when the fast one finds the step suspicious.](assets/diagrams/monitors-light.svg#only-light)](assets/diagrams/monitors-light.svg "Open the diagram at full size")
[![Three monitor classes cover three kinds of model, each drawn as the charm. LLMMonitor asks a chat model to reason and then score the step, from 0 to 10 by default. GuardModelMonitor has a guard model label the step against a policy you write, and suspicion is the chance the label is unsafe. DecisionModelMonitor asks a decision model yes or no questions, and suspicion is the chance of yes. Each returns the same Verdict, a suspicion from 0 to 1 and a reason, so any monitor works with any control protocol. Three wrappers stack around any monitor: RepeatedMonitor averages several calls, CalibratedMonitor ranks the score among the scores of honest runs, and CascadeMonitor runs a fast monitor on every step and a careful one when the fast one finds the step suspicious.](assets/diagrams/monitors-dark.svg#only-dark)](assets/diagrams/monitors-dark.svg "Open the diagram at full size")

| Monitor | Scores with | Guide |
|---|---|---|
| `LLMMonitor` | any chat model, prompted to reason and then score the step | [Use a chat judge](how-to/use-a-chat-judge.md) |
| `GuardModelMonitor` | a guard model, such as gpt-oss-safeguard or Llama Guard, and a policy you write | [Use a guard model](how-to/use-a-guard-model.md) |
| `DecisionModelMonitor` | a decision model that answers yes or no questions, such as Jev | [Use a decision model](how-to/use-a-decision-model.md) |
| `RepeatedMonitor`, `CalibratedMonitor`, `CascadeMonitor` | any of the above: averaged, ranked against honest runs, or cascaded | [Combine and calibrate monitors](how-to/combine-and-calibrate-monitors.md) |

## Choose what the monitor reads

[![Every entry of the transcript belongs to one channel. The default view, Channel.ACTIONS, reads USER (the task, context notes and feedback), TOOL_CALLS (tool calls and errors) and SUBAGENT_REPORTS (what subagents report), and leaves out REASONING, AGENT_TEXT and TOOL_RESULTS, each one flag away. The proposed step's tool calls are always shown, and its text when it is a final answer; the rest of the step follows the view. A MonitorView chooses the channels, can keep only the most recent entries, and names the tools whose results are subagent reports. The monitor reads the chosen entries as escaped, tagged text ending with the proposed step, then scores the step.](assets/diagrams/monitor-view-light.svg#only-light)](assets/diagrams/monitor-view-light.svg "Open the diagram at full size")
[![Every entry of the transcript belongs to one channel. The default view, Channel.ACTIONS, reads USER (the task, context notes and feedback), TOOL_CALLS (tool calls and errors) and SUBAGENT_REPORTS (what subagents report), and leaves out REASONING, AGENT_TEXT and TOOL_RESULTS, each one flag away. The proposed step's tool calls are always shown, and its text when it is a final answer; the rest of the step follows the view. A MonitorView chooses the channels, can keep only the most recent entries, and names the tools whose results are subagent reports. The monitor reads the chosen entries as escaped, tagged text ending with the proposed step, then scores the step.](assets/diagrams/monitor-view-dark.svg#only-dark)](assets/diagrams/monitor-view-dark.svg "Open the diagram at full size")

By default a monitor reads the user's messages, the tool calls and the
subagent reports, and leaves out the agent's reasoning, its prose and raw tool
output, as Claude Code's auto mode does. Each left-out channel is one flag
away:

```python
from langchain_sync_monitors import Channel, LLMMonitor, MonitorView

judge = LLMMonitor(
    model="openrouter:xiaomi/mimo-v2.6-pro",
    view=MonitorView(channels=Channel.ACTIONS | Channel.REASONING),
)
```

[Choose what the monitor reads](how-to/choose-what-the-monitor-reads.md)
covers every channel and the limit on recent entries.

## Monitor every subagent

[![In a Deep Agent, the main agent's monitor, monitor[main], reads the user's task. When the main agent calls the task tool, monitor[main] judges that call first, and the subagent then runs under a monitor of its own, monitor[researcher], which reads the task as the parent agent's words, not the user's, and judges each of the subagent's steps. The subagent's final report returns to the main agent. When the subagent returns or halts, its records flow back into the main agent's monitor_log, halts included. monitor_subagents gives every subagent a copy of the monitor, the general-purpose one included.](assets/diagrams/subagents-light.svg#only-light)](assets/diagrams/subagents-light.svg "Open the diagram at full size")
[![In a Deep Agent, the main agent's monitor, monitor[main], reads the user's task. When the main agent calls the task tool, monitor[main] judges that call first, and the subagent then runs under a monitor of its own, monitor[researcher], which reads the task as the parent agent's words, not the user's, and judges each of the subagent's steps. The subagent's final report returns to the main agent. When the subagent returns or halts, its records flow back into the main agent's monitor_log, halts included. monitor_subagents gives every subagent a copy of the monitor, the general-purpose one included.](assets/diagrams/subagents-dark.svg#only-dark)](assets/diagrams/subagents-dark.svg "Open the diagram at full size")

Deep Agents does not pass the main agent's middleware to its subagents, so
`monitor_subagents` gives each one, the general-purpose subagent included, a
copy of the monitor. With `monitor` from the quick start, and Deep Agents'
own file tools rather than the quick start's `read_file`, which would
replace Deep Agents' tool of the same name:

```python
from deepagents import create_deep_agent
from langchain_sync_monitors import monitor_subagents

researcher = {"name": "researcher", "description": "Finds and summarises sources."}
agent = create_deep_agent(
    "openrouter:deepseek/deepseek-v4.1-flash",
    middleware=[monitor],
    subagents=monitor_subagents(middleware=monitor, subagents=[researcher]),
)
```

[Monitor Deep Agents subagents](how-to/monitor-deep-agents-subagents.md)
covers per-subagent overrides, what a halt inside a subagent does, and the
subagents the helper cannot monitor.

## Where to go next

| If you want to | Read |
|---|---|
| build a monitored agent from scratch | [Tutorial: your first monitored agent](tutorials/first-monitored-agent.md) |
| solve one task, such as calibrating thresholds | [How-to guides](how-to/index.md) |
| look up a class or a keyword | [API reference](reference/api.md) |
| understand how a step flows and why | [How the library is built](explanation/design.md) |
| see monitored agents run against real models | [Live runs](explanation/live-runs.md) |

## Credits

The protocols come from AI control research. Trusted Monitoring and Defer to
Trusted come from AI Control [@greenblatt2023aicontrol], Defer to Resample
from Ctrl-Z [@bhatt2025ctrlz], and Auto Mode follows Claude Code's auto mode
[@hughes2026automode]. [Where the ideas come from](explanation/background.md)
credits every source and what it contributed, and the
[bibliography](references.bib) holds the full entries. If you use this
library in research, please cite the original authors.

## Status and licence

<!-- release-check: the text below describes an unreleased package. scripts/check-release.sh refuses a release while this comment is here; rewrite that text for the release, then delete this comment. -->

Pre-release, version 0.1.0.dev0, and not on PyPI yet. The monitors,
protocols, middleware and Deep Agents support are built and tested offline,
and the API may still change before 0.1.0; the
[changelog](https://github.com/omamori-lab/langchain-sync-monitors/blob/main/CHANGELOG.md)
records each change. The library is released under the
[MIT licence](https://github.com/omamori-lab/langchain-sync-monitors/blob/main/LICENSE).
