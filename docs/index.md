# langchain-sync-monitors

This page introduces the library and maps the rest of the documentation.

langchain-sync-monitors puts a monitor on every step your LangChain agent or
Deep Agent proposes, and a control protocol that decides what runs.

![One monitored step. The untrusted model proposes a step, the monitor reads it through its view and scores it from 0 to 1, and the control protocol chooses one of five outcomes: allow, resample, defer to trusted, block with feedback, or halt. Only the committed step reaches the tools, with one record in monitor_log.](assets/diagrams/monitored-step-light.svg#only-light)
![One monitored step. The untrusted model proposes a step, the monitor reads it through its view and scores it from 0 to 1, and the control protocol chooses one of five outcomes: allow, resample, defer to trusted, block with feedback, or halt. Only the committed step reaches the tools, with one record in monitor_log.](assets/diagrams/monitored-step-dark.svg#only-dark)

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

The package is not on PyPI yet. Until the first release, install it from
GitHub:

```console
pip install "langchain-sync-monitors[openrouter,deepagents] @ git+https://github.com/Antonio-Tresol/langchain-sync-monitors"
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

![The four control protocols. Trusted Monitoring lets a suspicious step run and flags it for review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to 2 fresh samples and runs the least suspicious one that passes, or has the trusted model write the step. Auto Mode blocks the step and tells the agent why; the agent tries again, and after 3 blocks in a row or 20 in the thread the run halts.](assets/diagrams/protocols-light.svg#only-light)
![The four control protocols. Trusted Monitoring lets a suspicious step run and flags it for review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to 2 fresh samples and runs the least suspicious one that passes, or has the trusted model write the step. Auto Mode blocks the step and tells the agent why; the agent tries again, and after 3 blocks in a row or 20 in the thread the run halts.](assets/diagrams/protocols-dark.svg#only-dark)

| Protocol | Pick it when | Guide |
|---|---|---|
| `TrustedMonitoring` | you want to watch and flag, and change nothing the agent does | [Use Trusted Monitoring](how-to/use-trusted-monitoring.md) |
| `DeferToTrusted` | you have a trusted model and want the simplest intervention | [Use Defer to Trusted](how-to/use-defer-to-trusted.md) |
| `DeferToResample` | you want to keep the agent's own work where you can, and keep rejected samples as evidence | [Use Defer to Resample](how-to/use-defer-to-resample.md) |
| `AutoMode` | you want the agent to hear why a step was blocked and find a safer route | [Use Auto Mode](how-to/use-auto-mode.md) |

Not sure which? [Choose a protocol](how-to/choose-a-protocol.md) compares
them.

## Pick a monitor

![Three kinds of monitor model, one scale. LLMMonitor, GuardModelMonitor and DecisionModelMonitor each return a Verdict with a suspicion from 0 to 1 and a reason, so any monitor fits any protocol. RepeatedMonitor, CalibratedMonitor and CascadeMonitor wrap any monitor and stack.](assets/diagrams/monitors-light.svg#only-light)
![Three kinds of monitor model, one scale. LLMMonitor, GuardModelMonitor and DecisionModelMonitor each return a Verdict with a suspicion from 0 to 1 and a reason, so any monitor fits any protocol. RepeatedMonitor, CalibratedMonitor and CascadeMonitor wrap any monitor and stack.](assets/diagrams/monitors-dark.svg#only-dark)

| Monitor | Scores with | Guide |
|---|---|---|
| `LLMMonitor` | any chat model, prompted to reason and then score the step | [Use a chat judge](how-to/use-a-chat-judge.md) |
| `GuardModelMonitor` | a guard model, such as gpt-oss-safeguard or Llama Guard, and a policy you write | [Use a guard model](how-to/use-a-guard-model.md) |
| `DecisionModelMonitor` | a decision model that answers yes or no questions, such as Jev | [Use a decision model](how-to/use-a-decision-model.md) |
| `RepeatedMonitor`, `CalibratedMonitor`, `CascadeMonitor` | any of the above: averaged, ranked against honest runs, or cascaded | [Combine and calibrate monitors](how-to/combine-and-calibrate-monitors.md) |

## Choose what the monitor reads

![What the monitor reads. The transcript has six channels: USER, REASONING, AGENT_TEXT, TOOL_CALLS, TOOL_RESULTS and SUBAGENT_REPORTS. The default view, Channel.ACTIONS, reads the user's messages, the tool calls and the subagent reports. A MonitorView picks the channels, and the monitor reads them as escaped, tagged text that ends with the proposed step.](assets/diagrams/monitor-view-light.svg#only-light)
![What the monitor reads. The transcript has six channels: USER, REASONING, AGENT_TEXT, TOOL_CALLS, TOOL_RESULTS and SUBAGENT_REPORTS. The default view, Channel.ACTIONS, reads the user's messages, the tool calls and the subagent reports. A MonitorView picks the channels, and the monitor reads them as escaped, tagged text that ends with the proposed step.](assets/diagrams/monitor-view-dark.svg#only-dark)

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

![A monitor in every subagent. The main agent's monitor reads the user's task. A subagent such as researcher runs under a monitor of its own, which reads its task as the parent agent's words. The subagent's report returns through the task tool, and its records, halts included, flow back into the main agent's monitor_log.](assets/diagrams/subagents-light.svg#only-light)
![A monitor in every subagent. The main agent's monitor reads the user's task. A subagent such as researcher runs under a monitor of its own, which reads its task as the parent agent's words. The subagent's report returns through the task tool, and its records, halts included, flow back into the main agent's monitor_log.](assets/diagrams/subagents-dark.svg#only-dark)

Deep Agents does not pass the main agent's middleware to its subagents, so
`monitor_subagents` gives each one, the general-purpose subagent included, a
copy of the monitor. With `monitor` and `read_file` from the quick start:

```python
from deepagents import create_deep_agent
from langchain_sync_monitors import monitor_subagents

researcher = {"name": "researcher", "description": "Finds and summarises sources."}
agent = create_deep_agent(
    "openrouter:deepseek/deepseek-v4.1-flash",
    tools=[read_file],
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

## Credits

The protocols come from AI control research. Trusted Monitoring and Defer to
Trusted come from AI Control [@greenblatt2023aicontrol], Defer to Resample
from Ctrl-Z [@bhatt2025ctrlz], and Auto Mode follows Claude Code's auto mode
[@hughes2026automode]. [Where the ideas come from](explanation/background.md)
credits every source and what it contributed, and the
[bibliography](references.bib) holds the full entries. If you use this
library in research, please cite the original authors.

## Status and licence

Pre-release, version 0.1.0.dev0, and not on PyPI yet. The monitors,
protocols, middleware and Deep Agents support are built and tested offline,
and the API may still change before 0.1.0; the
[changelog](https://github.com/Antonio-Tresol/langchain-sync-monitors/blob/main/CHANGELOG.md)
records each change. The library is released under the
[MIT licence](https://github.com/Antonio-Tresol/langchain-sync-monitors/blob/main/LICENSE).
