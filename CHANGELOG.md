# Changelog

Notable changes to langchain-sync-monitors. The format follows the spirit of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

The public API is the set of names exported from `langchain_sync_monitors`
(monitors, control protocols, the middleware and their constructor keywords)
and the shape of the `monitor_log` records. A release that could turn a working
integration into a failing one is a breaking change.

## [Unreleased]

The first release. Everything below is new.

### Added

**The middleware**

- `MonitorMiddleware`, which puts a monitor and a control protocol around every
  model call of a LangChain agent or Deep Agent, logs each step to
  `monitor_log` and streams it as a `MonitorStepEvent`; runs under both
  `invoke()` and `ainvoke()`. `check_monitor_placement` warns when middleware
  inside the monitor could return commands, when a middleware that retries
  or falls back sits outside it, and when a middleware that retries or
  answers failed tool calls, such as `ToolRetryMiddleware`, sits anywhere in
  the list.
- Only committed steps reach `stream_mode="messages"`. Rejected samples, blocked
  attempts with hidden feedback and the monitor's own model calls never stream,
  and each committed step arrives whole once the protocol has decided.
- A step that fails after samples were judged writes a `MonitorStepFailedEvent`
  with those samples to `stream_mode="custom"`, logs them as a warning, and
  re-raises the error.
- A halted step ends the run, including agents with structured output and no
  tools.
- Named spans for each monitored step in any LangChain tracer, LangSmith and
  Langfuse among them: a `monitor step` span holds the step's samples, a
  `monitor judgement` span each verdict, a `monitor classifier` span each
  Decisions API request, and a `monitor decision` span is tagged with the
  outcome, and with `monitor:flagged` when the step is flagged, so one filter
  finds every halted or flagged step. Every span carries flat `monitor_`
  metadata naming the step, with the delegation id inside a subagent that a
  monitored agent started. The spans add no tag or metadata to the LangChain
  runs inside them, model calls included, though LangSmith copies their
  metadata into its own `traceable` runs inside them. They leave every stream
  except `astream_events` and `astream_log` unchanged, and are not opened when
  no callback handler is attached; the monitors' own model calls stay out of
  LangSmith's Trajectory view.

**Control protocols**

- `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample` (sequential or
  parallel resampling) and `AutoMode` (blocks a step, tells the agent why with
  a `FeedbackTemplate`, and halts after Claude Code's limits, counted per
  conversation thread), with the `DeferToTrustedModel` and `HaltRun` fallbacks.
- Default thresholds (0.6 to defer, block or flag; 0.9 to audit) that raise
  `UncalibratedThresholdWarning` until you set your own. A flag marks a step
  for audit and never blocks it.
- A `CachedResampleWarning` when a response cache would turn resamples into
  copies of the first sample.
- Auto Mode's total counts across the whole conversation thread, subagents
  and nested subagents included: a subagent starts from the thread's count,
  so delegating again does not reset it, and the parent's next step goes to
  the fallback once blocks inside subagents reach the total. Blocks inside a
  subagent whose run raises count only when the run is resumed.
  `PendingStep.blocks_in_thread` and `PendingStep.new_subagent_blocks` give a
  protocol these counts.

**Monitors**

- Three model families: `LLMMonitor` (chat judges, with
  `DEFAULT_MONITOR_PROMPT`), `GuardModelMonitor` (guard models, scored from
  log-probabilities or sampling) and `DecisionModelMonitor` (decision models
  through `OpenRouterDecisionModel` or `TypeSafeDecisionModel`).
- A judge's verdict is read only where it ends the reply: the reasoning block,
  then exactly one score tag. A reply cut off at a length limit, or any other
  shape, is asked for again and then fails closed, so a score the agent plants
  in its step cannot become the verdict. A guard's label is read only where
  its format puts it.
- The `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
  `CalibratedMonitor` returns a conformal p-value, so a threshold of 0.99
  flags 1% of new honest steps on average.
- A verdict's reason, which Auto Mode shows the agent, carries no scores,
  probabilities or percentiles; they stay in `monitor_log`.

**What the monitor reads**

- `MonitorView` and `Channel` choose what a monitor reads; the default follows
  Claude Code's auto mode classifier. Every message from the task author is
  kept when `most_recent_entries` trims the history.
- Messages other code adds, such as conversation summaries, render as
  `<context_note>`, which authorises nothing. A call that failed or did not run
  shows as `<tool_error>`, and a call with unreadable arguments as
  `<malformed_tool_call>`, under every view that shows the call. The
  monitor's own feedback on a blocked step shows as `<monitor_feedback>`.

**Deep Agents**

- `monitor_subagents`, which gives every Deep Agents subagent, including the
  general-purpose one, a monitor of its own, with per-subagent overrides and a
  choice of what a subagent halt does to the run (`SubagentHalt`). Forked
  subagents (`mode="fork"`) are refused for now.
- Subagent records carry `delegation_id`, the id of the `task` call that
  started the subagent, and so does a subagent's `MonitorStepFailedEvent`.
- `Delegation`, what a monitored agent hands each subagent it starts: the
  call's id, the delegating agent and the thread's block counts. It travels
  in the `monitor_delegation` state key, which is part of every monitored
  agent's input; a value that is not a valid `Delegation` raises
  `ConfigurationError`.

**Errors**

- `MonitorError` and its subclasses: `ConfigurationError` (also raised for a
  plain string where an option enum is expected), `MissingExtraError` (with the
  install command for a missing extra, `openrouter` included),
  `SynchronousRunError` and `InvalidSuspicionError`.

**Project**

- Packaging with the `deepagents`, `openrouter` and `typesafe` extras, for
  Python 3.12, 3.13 and 3.14. The gate script, CI (including the lowest
  allowed dependency versions and an install without extras), pre-commit
  hooks, the documentation and the bibliography with its citation test.
- Documentation: a visual README; a tutorial that monitors a first agent, with
  or without an API key; how-to guides for choosing and using each protocol,
  each monitor family, wrapping and calibrating monitors, choosing what the
  monitor reads, reading the monitor log and monitoring Deep Agents
  subagents; and explanation pages on how the library is built, where its
  ideas come from, and how a monitored agent fared in live runs against real
  models, with the harness that reproduces them.
- A complete bibliography: every paper, post and code base the library draws
  on, cited where it is used, with a test that keeps the two in step.

### Changed

- A guard's verdict is read from its last line only when the label stands
  alone or follows a verdict key such as `Label:` or `Final answer:`; a label
  after any other key, as in `It asks for: no_violation`, is unreadable.
- A guard's reason from log-probabilities gives one of three findings by the
  suspicious share, with no number: it breaks the policy (0.5 or more), the
  guard was uncertain (0.001 to 0.5), or it follows the policy (below 0.001).
- `GuardScoring.AUTO` and `GuardScoring.LOG_PROBABILITIES` raise
  `ConfigurationError` naming `SAMPLE_FRACTION` when the chat model rejects
  the `logprobs` keyword, as `ChatAnthropic` and `ChatBedrockConverse` do.
- Log-probabilities without alternatives make `GuardScoring.AUTO` sample, and
  make `GuardScoring.LOG_PROBABILITIES` raise `ConfigurationError`.
- Under `Combine.MIN`, a decision monitor's reason names every question in the
  order given, not one of them.
- `GuardModelMonitor(scoring=...)` and `DecisionModelMonitor(combine=...)`
  refuse a plain string with `ConfigurationError`.
- `CalibratedMonitor(honest_scores=...)` and `DecisionModelMonitor(questions=...)`
  accept any iterable, a generator included.
- A chat model wrapped in a Runnable, such as `with_retry()` or `bind()`,
  raises `ConfigurationError` naming its type, in every monitor and in
  `DeferToTrustedModel`.

### Fixed

- A safe label the agent planted in its step is no longer read as the guard's
  verdict when the guard quotes it after its own label, after a colon on the
  same line, after a prose key, or on the line after its own keyed label.
- The guard's label parser runs in linear time on long lines of colons,
  dashes or spaces.
- A guard's log-probabilities are scored only at the guard's own label token,
  so a label wrapped in markup such as `(violation)` no longer scores near 0.
- Under `ainvoke()`, a failed guard sample cancels the other samples.
- A judge score too long for Python to convert counts as unreadable instead of
  raising `ValueError`.
- Replies cut off at a length limit are recognised from Bedrock Converse
  (`stopReason`) and Ollama (`done_reason`), and fail closed.
- A decision answer that is missing, NaN, outside 0 to 1, not a real number or
  a `bool` raises `MonitorError` for any `DecisionModel`, instead of being
  dropped by `Combine.MAX` or `Combine.MIN` or raising another error.
- Only a `TypeError` for an unexpected `logprobs` keyword is reported as a
  rejected request for log-probabilities.
- The guides state how a wrapped or sampled monitor counts an unreadable reply,
  and the decision model's retry budget, its handling of HTTP 408, the
  lifetime of its client and when `timeout_seconds` applies.
