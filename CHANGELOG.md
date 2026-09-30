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
- The middleware's `before_agent`, `before_model`, `after_model` and
  `after_agent` hooks, which show as graph nodes in a trace and add two graph
  steps per model call and two per run, all counted by an explicit
  `recursion_limit`. They keep three private state keys,
  `monitor_task_messages`, `monitor_seen_human_messages` and
  `monitor_run_open`, which never enter a subagent's input or a run's output
  but do appear in `stream_mode="updates"`. The hooks also write back, by id,
  the human messages they tag as notes, so `stream_mode="updates"` can carry
  such a message twice; merge messages by id.
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
  The default feedback tells the agent that none of its tools ran the blocked
  step, which stays true when a provider's built-in tool in that step already
  ran inside the model call.
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
- The library adds no score, probability or percentile to a verdict's reason,
  which Auto Mode shows the agent; they stay in `monitor_log`.
  `DEFAULT_MONITOR_PROMPT` asks a chat judge to keep its score out of the
  reasoning that becomes the reason, but the judge's own wording may still
  hint at one. The prompt also explains every tag the judge can read,
  provider tool calls and unrecognised blocks included.

**What the monitor reads**

- `MonitorView` and `Channel` choose what a monitor reads of the history; the
  default follows Claude Code's auto mode classifier. The step being judged
  always shows its tool calls and malformed calls, whatever the view. Every
  message from the task author is kept when `most_recent_entries` trims the
  history.
- Only the human messages a run receives as its input render as `<user>`, or
  `<delegator>` inside a subagent. Every other human message renders as
  `<context_note>`, which authorises nothing: one another component tagged,
  such as a conversation summary, and one written during a run without a tag,
  such as Deep Agents' video reads and harness nudges. The monitor writes the
  note's tag into the message in the agent's state too, so a saved or
  replayed history keeps it, and a message a tool writes never carries the
  monitor's own `lc_source`. A tool that writes back a message by id keeps
  its author only if the message is unchanged. After a run that stopped
  before its end, such as one whose model call raised or whose client
  cancelled it, the next run's new messages, the user's own included, are
  notes from `unconfirmed_input` for the rest of the thread: the judge takes
  no approval from them, but still applies any limit they set.
- A call that failed or did not run shows as `<tool_error>`, and a call with
  unreadable arguments as `<malformed_tool_call>`, under every view that shows
  the call. The monitor's own feedback on a blocked step shows as
  `<monitor_feedback>`.
- A provider's built-in tools, which run inside the model call, show as
  `<provider_tool_call>` and `<provider_tool_result>`: those LangChain maps
  for Anthropic, OpenAI's Responses API and Gemini's code execution, and
  Gemini's Google Search, read from `grounding_metadata`. A part of a reply
  LangChain cannot map shows whole as `<unrecognised_block>`; a refusal shows
  as the agent's prose. Each sample's
  `SampleRecord.proposal` keeps the provider tool calls and their full
  results.

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
