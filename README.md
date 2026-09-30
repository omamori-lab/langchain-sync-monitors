# langchain-sync-monitors

**A monitor for every step your LangChain agent or Deep Agent proposes, and a control protocol that decides what runs.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/diagrams/monitored-step-dark.svg">
  <img alt="The untrusted model proposes a step. The monitor, drawn as an omamori charm, reads it through its view and scores its suspicion from 0 to 1. The control protocol chooses one of five outcomes, from the lightest touch to the strongest: allow it, resample it, defer to the trusted model, block it and tell the agent why, or halt. All of this happens inside the middleware, before any of the agent's own tools run; a brushed line marks that boundary. A provider's built-in tools, such as web search, run inside the model call, above that line. A committed step reaches the tools and a halted run ends, and either way one record joins monitor_log. Tool results come back, and the next model call is a new step." src="docs/assets/diagrams/monitored-step-light.svg">
</picture>

## Why this exists

An agent with tools can do harm in a single step: post a credential, delete a
file, run a command. AI control research guards against this with a monitor,
a second model that scores each step, and a protocol that acts on the score.
This library brings those protocols, from AI Control, Ctrl-Z and Claude Code's
auto mode, to `create_agent` and `create_deep_agent` as one middleware.
"Sync" means synchronous in the control sense: a step waits for its verdict,
so a suspicious step can be stopped before any of the agent's own tools run
it.

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
    # The monitor: a chat model that judges every step the agent proposes.
    monitor=LLMMonitor(model="openrouter:xiaomi/mimo-v2.6-pro"),
    # The protocol: sample a suspicious step again, and if every sample stays
    # suspicious, have the trusted model write the step instead.
    protocol=DeferToResample(
        fallback=DeferToTrustedModel(trusted_model="openrouter:xiaomi/mimo-v2.6-flash"),
        # Resample a step that scores 0.6 or more, and flag it for review
        # when even its least suspicious sample scores 0.9 or more.
        defer_threshold=0.6,
        audit_threshold=0.9,
    ),
)
# The untrusted model does the work, with the monitor last in the list.
agent = create_agent(
    "openrouter:deepseek/deepseek-v4.1-flash", tools=[read_file], middleware=[monitor]
)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarise q3.md."}]})
# One record per step: what the protocol did, and whether to review it.
for record in result["monitor_log"]:
    print(record["step_number"], record["outcome"], record["flagged"])
```

Three models play three roles: the untrusted agent does the work, the monitor
judges each step, and the trusted model writes a step when every sample looks
suspicious. Each is a parameter, and any LangChain chat model works; the
library picks none for you. Thresholds share one scale from 0 to 1 and warn
until you set them; calibrate them on honest runs of your own agent. Put the
monitor last in the middleware list; call
`check_monitor_placement(middleware=[...])` with your list to be warned about
middleware that undermines it.

## Limits

- **Tools the provider runs.** A provider's built-in tools, such as
  Anthropic's web search or OpenAI's code interpreter, run inside the model
  call: they have run before the monitor sees the step, and run again for
  every sample drawn. `ProviderToolWarning` flags the ones it can see, for
  Anthropic, OpenAI and Gemini, but not all: not OpenRouter's web plugin, for
  one.
- **What the judge reads.** By default it reads the task, the tool calls, the
  subagent reports and a proposed final answer. It does not read the agent's
  reasoning, the text the agent writes beside a tool call, or raw tool
  output; each is one flag away.
- **Halts.** A halt stands until a later run brings a message the monitor
  records as new input from the user. A middleware listed before the monitor
  can still lift it, with an untagged human message written from its
  `before_agent` hook, or from its `after_agent` hook without sending the run
  back to the model.
- **Subagents.** Deep Agents subagents that run in parallel do not see each
  other's blocks, so together they can pass Auto Mode's limit on blocks in
  the thread; their parent counts every one of them at its next step. A
  subagent whose run raises returns no records unless the run is resumed.
- **Time and graph steps.** Each step waits for its verdict and streams whole
  once committed, not token by token. The monitor adds two graph steps per
  model call and two per run, which count towards `recursion_limit`.

[How the library is built](docs/explanation/design.md) explains each of these.

## Read the docs

| If you want to | Read |
|---|---|
| build a monitored agent step by step, with or without an API key | [Monitor your first agent](docs/tutorials/first-monitored-agent.md) |
| pick `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample` or `AutoMode` | [Choose a control protocol](docs/how-to/choose-a-protocol.md) |
| judge with a chat model, a guard model or a decision model | [Use a chat judge](docs/how-to/use-a-chat-judge.md), [Use a guard model](docs/how-to/use-a-guard-model.md), [Use a decision model](docs/how-to/use-a-decision-model.md) |
| average, calibrate or cascade monitors | [Combine and calibrate monitors](docs/how-to/combine-and-calibrate-monitors.md) |
| choose what the monitor reads | [Choose what the monitor reads](docs/how-to/choose-what-the-monitor-reads.md) |
| monitor every Deep Agents subagent | [Monitor Deep Agents subagents](docs/how-to/monitor-deep-agents-subagents.md) |
| read the records, the stream events and the logs | [Read the monitor log](docs/how-to/read-the-monitor-log.md) |
| find halted and flagged steps in a tracer | [See the monitor's decisions in LangSmith and Langfuse](docs/how-to/see-decisions-in-langsmith-and-langfuse.md) |
| look up a class or a keyword | [API reference](https://omamori-lab.github.io/langchain-sync-monitors/reference/api/) |
| understand how a step flows, and why | [How the library is built](docs/explanation/design.md) |
| see monitored agents run against real models | [Live runs of a monitored agent](docs/explanation/live-runs.md) |

## Credits

The protocols come from AI control research. Trusted Monitoring and Defer to
Trusted come from [AI Control](https://arxiv.org/abs/2312.06942) (Greenblatt
et al., 2023), Defer to Resample from
[Ctrl-Z](https://arxiv.org/abs/2504.10374) (Bhatt et al., 2025), and Auto
Mode follows
[How we built Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode)
(Hughes, Anthropic, 2026). [Where the ideas come from](docs/explanation/background.md)
credits every source and what it contributed, and
[`docs/references.bib`](docs/references.bib) holds the full entries. If you
use this library in research, please cite the original authors.

## Status and licence

<!-- release-check: the text below describes an unreleased package. scripts/check-release.sh refuses a release while this comment is here; rewrite that text for the release, then delete this comment. -->

Pre-release, version 0.1.0.dev0, and not on PyPI yet. The monitors,
protocols, middleware and Deep Agents support are built, tested offline and
tried in [live runs](docs/explanation/live-runs.md), and the API may still
change before 0.1.0; [CHANGELOG.md](CHANGELOG.md)
records each change. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to work
on the library, and [docs/plans](docs/plans/README.md) holds each feature's
design plan.

Released under the MIT licence; see [LICENSE](LICENSE).
