# Choose a protocol

This guide helps you pick one of the four control protocols, and states what
all four share: thresholds, halts, the tools a provider runs itself, and the
cost in graph steps.

[TOC]

The monitor gives every step a suspicion score from 0 to 1, and the protocol
decides what that score does. Every protocol works with every monitor, so you
can change one without touching the other.

## Decide in three questions

[![Three questions choose one of the four control protocols. First, stop suspicious steps, or only flag them for a person to review? To only flag them, use Trusted Monitoring. To stop them, ask second: tell a blocked agent why, so it can take a safer route? If yes, use Auto Mode. If the agent should not learn what the monitor catches, ask third: pay for up to two more samples of a suspicious step? If yes, use Defer to Resample; if no, use Defer to Trusted.](../assets/diagrams/protocol-choice-light.svg#only-light)](../assets/diagrams/protocol-choice-light.svg "Open the diagram at full size")
[![Three questions choose one of the four control protocols. First, stop suspicious steps, or only flag them for a person to review? To only flag them, use Trusted Monitoring. To stop them, ask second: tell a blocked agent why, so it can take a safer route? If yes, use Auto Mode. If the agent should not learn what the monitor catches, ask third: pay for up to two more samples of a suspicious step? If yes, use Defer to Resample; if no, use Defer to Trusted.](../assets/diagrams/protocol-choice-dark.svg#only-dark)](../assets/diagrams/protocol-choice-dark.svg "Open the diagram at full size")

1. **Stop or flag.** A flag marks a step's record for a person to review
   later. It never blocks: by the time anyone reads the log, the step has run.
   To watch an agent, or to collect the honest-run scores a threshold needs,
   use Trusted Monitoring.
2. **Tell the agent or not.** Auto Mode tells the agent why a step was
   blocked. That helps an honest agent that overreached, and it also tells a
   scheming agent what the monitor catches. The Defer protocols replace a
   suspicious step without telling the agent.
3. **Resample or defer at once.** Defer to Resample asks the agent's model
   again before giving up on the step, so an honest step that looked odd once
   often still runs. Defer to Trusted hands the step to the trusted model at
   once: fewer calls, but the stronger model's work on that step is lost.

## Compare them

[![What each protocol does with a step the monitor finds suspicious; a step below the threshold runs unchanged under every protocol. Trusted Monitoring lets the step run and flags it for a person to review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to two fresh samples; if one passes, the least suspicious runs, and if none passes, the fallback writes the step, usually the trusted model. Auto Mode blocks the step and tells the agent why, and the agent tries again; after three blocks in a row, or twenty in the thread with subagents' blocks included, the fallback takes over, which by default halts the run.](../assets/diagrams/protocols-light.svg#only-light)](../assets/diagrams/protocols-light.svg "Open the diagram at full size")
[![What each protocol does with a step the monitor finds suspicious; a step below the threshold runs unchanged under every protocol. Trusted Monitoring lets the step run and flags it for a person to review. Defer to Trusted has the trusted model write the step instead. Defer to Resample draws up to two fresh samples; if one passes, the least suspicious runs, and if none passes, the fallback writes the step, usually the trusted model. Auto Mode blocks the step and tells the agent why, and the agent tries again; after three blocks in a row, or twenty in the thread with subagents' blocks included, the fallback takes over, which by default halts the run.](../assets/diagrams/protocols-dark.svg#only-dark)](../assets/diagrams/protocols-dark.svg "Open the diagram at full size")

The call counts assume a monitor that makes one model call per judgement, such
as an `LLMMonitor` whose reply is readable. An unreadable reply is asked for
again, up to `max_parse_retries` more times, 2 by default, and a wrapper such
as `RepeatedMonitor` multiplies the monitor's calls.

| | Trusted Monitoring | Defer to Trusted | Defer to Resample | Auto Mode |
|---|---|---|---|---|
| What runs | The agent's step, always | The agent's step, or the trusted model's when it is suspicious | The least suspicious acceptable sample, or the fallback's step when none is acceptable | The first acceptable attempt, or the fallback's step once a block limit is reached |
| Model calls, ordinary step | 2: agent and monitor | 2 | 2 | 2 |
| Model calls, suspicious step, with the defaults | 2 | 3: adds the trusted model | Up to 6: three samples and three judgements; 7 when the trusted model writes the step | Up to 6: three attempts and three judgements; 7 when the trusted model writes the step |
| Latency, suspicious step | No extra | One trusted call | Up to two more rounds in sequence, or one in parallel, plus the fallback's call | One more round per attempt, plus the fallback's call |
| What the agent sees | Its own step | The trusted step, as if it were its own | The chosen sample, or the fallback's step; never a rejected sample | Each retry sees the step's blocked attempts and their feedback; afterwards, only the accepted step unless feedback stays in the transcript; at a limit, the fallback's step |
| Use it when | You want to observe, calibrate or audit, and blocking honest work costs more than catching a bad step late | A capable trusted model exists and you want the simplest intervention | You want to keep most of the stronger model's work and gather evidence from repeated samples | Most suspicious steps are an honest agent overreaching, and a person would otherwise approve each step |

The defaults are `max_resamples=2` for Defer to Resample and
`max_consecutive_blocks=3` for Auto Mode. A round is one call of the agent's
model followed by one call of the monitor. The fallback's call is one call of
the trusted model when the fallback is `DeferToTrustedModel`, and none for
`HaltRun`. Parallel resampling saves time only under `ainvoke()`; under
`invoke()` the samples are drawn one after another. Auto Mode keeps blocked
attempts and their feedback in the agent's messages only with
`feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT`.

Each protocol records its steps with these outcomes, and flags them for review
as follows:

| Protocol | Outcomes in the log | Flagged when |
|---|---|---|
| Trusted Monitoring | `allowed` | The suspicion is at or above `audit_threshold` |
| Defer to Trusted | `allowed`, `deferred_to_trusted` | The suspicion is at or above `audit_threshold` |
| Defer to Resample | `allowed`, `resampled`, `deferred_to_trusted`, `halted` | The lowest suspicion across the samples is at or above `audit_threshold`, or the step halted |
| Auto Mode | `allowed`, `steered`, `deferred_to_trusted`, `halted` | An attempt was blocked, or the fallback wrote the step |

Any protocol can also log `halted`. With
`when_subagent_halts=SubagentHalt.STOP_WHOLE_RUN`, a monitor halts its agent
once a subagent was halted, and records a flagged step with no samples; see
[Monitor Deep Agents subagents](monitor-deep-agents-subagents.md).

## Where each protocol comes from

- **Trusted Monitoring** and **Defer to Trusted** come from the AI control
  paper [@greenblatt2023aicontrol]. There an untrusted model writes code, a
  weaker trusted model rates how suspicious it looks, and the most suspicious
  2% is shown to a person before it is used; a backdoor the person finds shuts
  the untrusted model down. Defer to Trusted also submits code the trusted
  model writes when the untrusted model's code scores above a deferral
  threshold but outside the audited 2%. Here nothing waits for a person: a flag
  marks the step's record, and the step has already run.
- **Defer to Resample** comes from Ctrl-Z [@bhatt2025ctrlz]. When a step looks
  suspicious, it draws up to two new samples, runs the least suspicious one
  below the deferral threshold, and defers to the trusted model only when every
  sample is above it. At the same deferral threshold it costs much less
  usefulness than deferring at once, because a fresh sample of an innocent step
  may look innocent. In Ctrl-Z's BashBench setting, with a 1% audit budget and
  usefulness scaled from the trusted model's 0% to the untrusted model's 100%,
  it reached 83.2% safety at 95% usefulness, against 72.7% for Defer to
  Trusted. Those figures come from protocols in which a person audits the most
  suspicious actions, and a caught attack ends the episode; this library's flag
  stops nothing. With stronger models, and attacks the untrusted model designs
  itself, Terekhov et al. found that, on BashBench, Defer to Resample performs
  similarly to Defer to Trusted even without prompt injection, and that
  resampling gives text aimed at the monitor several tries, which turns an
  injection that works only sometimes into a best-of-n attack
  [@terekhov2025adaptive].
- **Auto Mode** follows Claude Code's auto mode [@hughes2026automode]. There a
  classifier reviews, before it runs, each action that neither an allowlist of
  safe tools, the user's own narrow allow rules nor the rule for file edits
  inside the project already allows. A blocked action goes back to the agent
  with the reason, and a session that reaches three denials in a row or twenty
  in all is stopped and handed to a person, or, in headless mode, ended. Here
  the monitor judges every step. The library keeps both limits, counts the
  total per conversation thread rather than per session, and hands the step to
  a fallback. The default fallback, `HaltRun`, ends the run, as headless mode
  does.

## What every protocol shares

### Thresholds

Every threshold is on the monitor's 0 to 1 scale, and a step at or above it
counts. The defaults, 0.6 to flag, defer or block and 0.9 to audit, are
placeholders: a protocol built with one emits an
`UncalibratedThresholdWarning`. Pass your own value, set from the scores your
monitor gives on honest runs, as
[Calibrate the threshold](use-trusted-monitoring.md#calibrate-the-threshold)
shows.

A threshold may be any real number from 0 to 1, a `Fraction` or `Decimal`
included, and its range is checked on the exact value before it becomes a
float; `-0.0` reads as 0. A `bool`, a string such as `"0.6"`, NaN or a value
outside the range raises `ConfigurationError` when the protocol is built. So
does any other option of the wrong type: a limit given as a float or a
`bool`, other than `math.inf` for Auto Mode's `max_total_blocks`, a plain
string where an enum member belongs, or the `DeferToTrusted` protocol where
the `DeferToTrustedModel` fallback belongs.

### A halt stands

A halt ends the run with a final message and no tool calls. It comes from the
`HaltRun` fallback, which is Auto Mode's default, or from
`SubagentHalt.STOP_WHOLE_RUN` after a subagent was halted. The middleware
routes the agent to its end, even an agent that would otherwise loop until it
has a structured response.

The halt then stands until a later run brings a message the monitor records as
new input from the user: the count of the thread's recorded run inputs must
grow. Until then, if a hook sends the run back to the model, as Deep Agents'
`RubricMiddleware` does when it grades the task unmet [@deepagents2026], each
further step halts again without a sample. The agent's model is not called, the
record is `halted`, flagged, with no samples, and the final message is
`STANDING_HALT_MESSAGE`, from `halts.py`, whatever message your `HaltRun` has:

> [Safety monitor] This run is stopped: the safety monitor halted an earlier
> step, and that halt stands until a later run brings new input the monitor
> can confirm. The rejected steps are recorded in monitor_log for review.

[![Before the protocol runs, the monitor checks two halts, and either one decides the step without a sample. A new step can follow a halt, because an after_agent hook can send a halted run back to the model. First, if this monitor's last step halted and the thread has recorded no run input since, the step halts again with STANDING_HALT_MESSAGE, from halts.py. If not, and the monitor has SubagentHalt.STOP_WHOLE_RUN, and a subagent halted since this agent's last step, the step halts, and that halt stands as this agent's own. Otherwise the protocol decides the step as usual. Either halt draws no sample: its record is halted and flagged, and holds no samples.](../assets/diagrams/halt-stands-light.svg#only-light)](../assets/diagrams/halt-stands-light.svg "Open the diagram at full size")
[![Before the protocol runs, the monitor checks two halts, and either one decides the step without a sample. A new step can follow a halt, because an after_agent hook can send a halted run back to the model. First, if this monitor's last step halted and the thread has recorded no run input since, the step halts again with STANDING_HALT_MESSAGE, from halts.py. If not, and the monitor has SubagentHalt.STOP_WHOLE_RUN, and a subagent halted since this agent's last step, the step halts, and that halt stands as this agent's own. Otherwise the protocol decides the step as usual. Either halt draws no sample: its record is halted and flagged, and holds no samples.](../assets/diagrams/halt-stands-dark.svg#only-dark)](../assets/diagrams/halt-stands-dark.svg "Open the diagram at full size")

Nothing written during a run lifts the halt: not a hook sending the run back
to the model, not a harness nudge, not a note from `unconfirmed_input`. A new
run that brings no new message stays halted too. Two paths remain open, both
through another middleware listed before the monitor:

- an untagged human message its `before_agent` hook writes counts as the
  run's input, and lifts the halt;
- an untagged human message its `after_agent` hook writes without sending the
  run back to the model counts as the next run's input, and lifts it.

`check_monitor_placement` does not warn about these hooks, which many
middleware have for sound reasons.

### Server tools

A monitor judges each step before any of the agent's own tools run. Some
providers also run built-in tools themselves, on their servers, inside the
model call. Those have already run by the time the monitor sees the step, and
they run again for every sample or attempt the protocol draws, so no protocol
can stop them. Give the agent tools of its own for any action that must be
judged before it runs.

The middleware emits a `ServerToolWarning`, once per middleware instance,
when a model call receives one of the server tools it knows:

- Anthropic's web search, web fetch, code execution and MCP connector
  [@anthropic2026tooluse];
- OpenAI's web search, file search, code interpreter, image generation and
  remote MCP [@openai2026tools];
- Gemini's Google Search, Google Maps, code execution and URL context
  [@google2026geminitools].

It reads the tools of the model request. It also reads the tools bound on the
model with `bind_tools` before the agent was built, but only for an agent with
no tools of its own and no `response_format`: otherwise `create_agent` binds
the model afresh, and those tools never reach it [@langchain2026]. It cannot
see, and so never warns about:

- a server-side feature switched on in the model's own settings, such as
  OpenRouter's web plugin or an `:online` model [@langchainopenrouter2026];
- a tool an integration has already turned into an object of its own SDK;
- tools bound inside a wrapper such as `with_fallbacks(...)`;
- tools queued with `bind_tools` on a configurable model from
  `init_chat_model(..., configurable_fields=...)`.

Those run the same way, without a warning. The tool search of Anthropic and
of OpenAI, and Anthropic's advisor, run at the provider too, but act on
nothing outside it, so the warning leaves them out. Client tools that a
provider defines and your application runs, such as Anthropic's bash, text
editor, memory and computer use tools and OpenAI's computer use and patch
tools, come back as tool calls, which the monitor judges before they run.

### Middleware options

`MonitorMiddleware` takes the `monitor` and the `protocol`, and these options:

| Option | Default | What it does |
|---|---|---|
| `label` | `"monitor"` | Names the monitor `label[agent_name]`, `monitor[main]` by default, and its records. Stacked monitors need distinct labels, and Auto Mode counts its thread total per label. |
| `agent_name` | `"main"` | The agent the records belong to. |
| `task_author` | `TaskAuthor.USER` | Who wrote the task the monitor reads; `TaskAuthor.PARENT_AGENT` inside a subagent. |
| `feedback_visibility` | `FeedbackVisibility.HIDDEN` | Whether Auto Mode's blocked attempts and their feedback stay in the agent's messages. |
| `when_subagent_halts` | `SubagentHalt.STOP_SUBAGENT_ONLY` | What a subagent's halt does to this agent's run. |

`label` and `agent_name` must be non-blank strings without `:` or `|`, for
the reason [Names and labels](../explanation/design.md#names-and-labels)
gives, and each enum option a member of its enum; anything else raises
`ConfigurationError`.

### Graph steps, streams and failures

| Topic | What happens |
|---|---|
| Graph steps | The middleware's hooks add two graph steps per agent step, however many model calls the step makes, and two per run; [Allow for the graph steps](read-the-monitor-log.md#allow-for-the-graph-steps) gives the `recursion_limit` a run needs. |
| Streams | `stream_mode="messages"` shows only committed steps, each whole once the protocol has decided. Rejected samples and the built-in monitors' calls never stream, though a custom monitor's call made with its own `tags` does; Auto Mode with `FeedbackVisibility.IN_TRANSCRIPT` commits its blocked attempts with the step, so those do. [Read the monitor log](read-the-monitor-log.md) covers the other stream modes. |
| A failed step | If a sample, a monitor call or the fallback raises, nothing is committed and the error propagates. The step is written to `stream_mode="custom"` as a `monitor_step_failed` event with the samples judged so far, and logged as a warning that gives their suspicions when there were any. A middleware outside the monitor that retries failed model calls, such as `ModelRetryMiddleware`, runs the whole step again with fresh samples. |
| Placement | The monitor goes last in the `create_agent` middleware list. `check_monitor_placement(middleware=[...])` warns about middleware placed where it undermines the monitor, a second monitor included: of the library's protocols, only Trusted Monitoring, and Defer to Resample or Auto Mode set to draw at most one sample per step and then halt, keep the records of a monitor inside them sound, as [Two monitors in one agent](../explanation/design.md#two-monitors-in-one-agent) explains. |

## Related guides

- [Use Trusted Monitoring](use-trusted-monitoring.md)
- [Use Defer to Trusted](use-defer-to-trusted.md)
- [Use Defer to Resample](use-defer-to-resample.md)
- [Use Auto Mode](use-auto-mode.md)
- [Use a chat judge](use-a-chat-judge.md) for the monitor most setups start with.
- [Combine and calibrate monitors](combine-and-calibrate-monitors.md) for thresholds that mean an audit budget.
- [Read the monitor log](read-the-monitor-log.md) for what each protocol records.

## References
