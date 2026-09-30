# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API is the set of names exported from `langchain_sync_monitors`
(monitors, control protocols, the middleware and their constructor keywords)
and the shape of the `monitor_log` records. A release that could turn a working
integration into a failing one is a breaking change; before 1.0.0, a breaking
change raises the minor version.

## [Unreleased]

### Added

- `MonitorMiddleware`, which puts a monitor and a control protocol around every
  model call of a LangChain agent or Deep Agent, under both `invoke()` and
  `ainvoke()`.
- Each monitored step is logged to `monitor_log` and streamed as a
  `MonitorStepEvent`.
- `check_monitor_placement`, which warns when middleware inside the monitor
  could return commands, when a middleware that retries or falls back sits
  outside it, and when one that retries or answers failed tool calls, such as
  `ToolRetryMiddleware`, sits anywhere in the list.
- `check_monitor_placement` warns when a monitor, or a middleware between two
  monitors, sits inside a monitor whose protocol can call the model more than
  once in a step, which loses or duplicates the inner records, or when a
  monitor inside another commits its blocked attempts with
  `FeedbackVisibility.IN_TRANSCRIPT`; `monitor_subagents` warns about such
  stacks in the subagents it monitors.
- Only committed steps reach `stream_mode="messages"`, each whole once the
  protocol has decided; rejected samples, blocked attempts with hidden
  feedback and the monitor's own model calls never stream.
- A step that fails after samples were judged writes a
  `MonitorStepFailedEvent` with those samples to `stream_mode="custom"`, logs
  them as a warning and re-raises the error.
- A halted step ends the run, including in agents with structured output and
  no tools.
- A halt stands until the thread records another run input: when a hook such
  as Deep Agents' `RubricMiddleware` sends a halted run back to the model, each
  further step halts again without calling the untrusted model, and its
  `halted` record holds no samples.
- A hook that rewrites, trims or removes messages cannot lift a halt.
- A harness nudge, or a note from `unconfirmed_input`, cannot lift a halt.
- An untagged human message that a middleware listed before the monitor writes
  from its `before_agent` hook, or from its `after_agent` hook without a return
  to the model, counts as a run's input and lifts a halt.
- `ProviderToolWarning`, raised once per middleware when the model call
  receives a server tool that Anthropic, OpenAI or Gemini runs itself, before
  the monitor can judge the step.
- `ProviderToolWarning` also reads tools bound on the model before the agent
  was built, for an agent with no tools of its own and no `response_format`,
  the only case in which those tools reach the model.
- `ProviderToolWarning` does not see a server-side feature set on the model
  itself, such as OpenRouter's web plugin, nor tools bound inside a wrapper
  such as `with_fallbacks(...)` or queued on a configurable model.
- The middleware's `before_agent`, `before_model`, `after_model` and
  `after_agent` hooks, which show as graph nodes in a trace and add two graph
  steps per model call and two per run, all counted by an explicit
  `recursion_limit`.
- Six private state keys, `monitor_task_messages`,
  `monitor_seen_human_messages`, `monitor_run_inputs`,
  `monitor_rewritten_inputs`, `monitor_run_open` and `monitor_inputs_at_halt`,
  which never enter a subagent's input or a run's output but do appear in
  `stream_mode="values"`, `stream_mode="updates"` and `get_state`.
- `monitor_inputs_at_halt` keeps how many run inputs the thread had at each
  monitor's latest halt.
- The monitor's hooks write back, by id, the human messages they tag as notes,
  so `stream_mode="updates"` can carry such a message twice; merge messages by
  id.
- Named spans in any LangChain tracer, LangSmith and Langfuse among them:
  `monitor step` holds a step's samples, `monitor judgement` each verdict and
  `monitor classifier` each Decisions API request.
- A `monitor decision` span tagged with the outcome, and with
  `monitor:flagged` when the step is flagged, so one filter finds every halted
  or flagged step.
- Flat `monitor_` metadata on every span naming the step, with the delegation
  id inside a subagent that a monitored agent started.
- The spans add no tag or metadata to the LangChain runs inside them, model
  calls included, though LangSmith copies their metadata into its own
  `traceable` runs inside them.
- The spans leave every stream except `astream_events` and `astream_log`
  unchanged, are not opened when no callback handler is attached, and keep the
  monitors' own model calls out of LangSmith's Trajectory view.
- Control protocols `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample`,
  with sequential or parallel resampling, and `AutoMode`, with the
  `DeferToTrustedModel` and `HaltRun` fallbacks.
- `AutoMode` blocks a step, tells the agent why with a `FeedbackTemplate`, and
  halts after Claude Code's limits, counted per conversation thread.
- The default feedback tells the agent that none of its tools ran the blocked
  step, which stays true when a provider's built-in tool in that step already
  ran inside the model call.
- Auto Mode's total counts across the whole conversation thread, subagents and
  nested subagents included: a subagent starts from the thread's count, so
  delegating again does not reset it.
- The parent's next step goes to the fallback once blocks inside subagents
  reach the total; blocks inside a subagent whose run raises count only when
  the run is resumed.
- `PendingStep.blocks_in_thread` and `PendingStep.new_subagent_blocks`, which
  give a protocol these counts.
- Default thresholds, 0.6 to defer, block or flag and 0.9 to audit, which raise
  `UncalibratedThresholdWarning` until you set your own.
- A flag marks a step for audit and never blocks it.
- `CachedResampleWarning`, raised when a response cache would turn resamples
  into copies of the first sample.
- `AutoMode(max_total_blocks=math.inf)` turns the total block limit off.
- Monitors for three model families: `LLMMonitor` for chat judges, with
  `DEFAULT_MONITOR_PROMPT`; `GuardModelMonitor` for guard models, scored from
  log-probabilities or sampling; and `DecisionModelMonitor` for decision
  models, through `OpenRouterDecisionModel` or `TypeSafeDecisionModel`.
- A judge's verdict is read only where it ends the reply, the reasoning block
  then exactly one score tag, so a score the agent plants in its step cannot
  become the verdict.
- A judge's reply cut off at a length limit, or of any other shape, is asked
  for again and then fails closed.
- A guard's label is read only where its format puts it.
- The `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
- `CalibratedMonitor` reports a step's suspicion as the smoothed conformal rank
  of its score among honest-run scores, so a threshold of 0.99 flags 1% of new
  honest steps on average.
- The library adds no score, probability or percentile to a verdict's reason,
  which Auto Mode shows the agent; they stay in `monitor_log`.
- `DEFAULT_MONITOR_PROMPT` asks a chat judge to keep its score out of the
  reasoning that becomes the reason, but the judge's own wording may still
  hint at one.
- `DEFAULT_MONITOR_PROMPT` explains every tag the judge can read, provider tool
  calls and unrecognised blocks included.
- `MonitorView` and `Channel`, which choose what a monitor reads of the
  history; the default follows Claude Code's auto mode classifier.
- The step being judged always shows its tool calls and malformed calls,
  whatever the view.
- Every message from the task author is kept when `most_recent_entries` trims
  the history.
- Only the human messages a run receives as its input render as `<user>`, or
  `<delegator>` inside a subagent.
- Every run's input reaches the judge verbatim and in order even after
  summarisation or a tool has taken it out of the model request, since
  `monitor_run_inputs` keeps its text, which grows the state by the size of
  the user's turns; input after a stopped run comes back as a note from
  `unconfirmed_input`. The judge reads each input whole on every step, even
  one Deep Agents shows the agent only as a preview.
- A tool's `Command` writes to the state keys only the monitor writes, every
  monitor key but `monitor_log`, are dropped with a warning.
- Every other human message renders as `<context_note>`, which authorises
  nothing: one another component tagged, such as a conversation summary, and
  one written during a run without a tag, such as Deep Agents' video reads and
  harness nudges.
- The monitor writes a note's tag into the message in the agent's state too,
  so a saved or replayed history keeps it.
- A message a tool writes never carries a source only the monitor writes,
  `monitor` or `unconfirmed_input`, and a tool that writes back a message by
  id keeps its author only if the whole message is unchanged, its metadata
  included.
- After a run that stopped before its end, such as one whose model call raised
  or whose client cancelled it, the next run's new messages, the user's own
  included, are notes from `unconfirmed_input` for the rest of the thread: the
  judge takes no approval from them, but still applies a limit they set that
  narrows what the agent may do. So is a nudge that a middleware listed after
  the monitor, such as a Deep Agents harness profile's, wrote just before the
  stop.
- `DEFAULT_MONITOR_PROMPT` tells the judge that a note never removes a
  safeguard, such as asking the user first, logging or reporting.
- Under every view that shows a tool call, a call that failed or did not run
  shows as `<tool_error>`, and one with unreadable arguments as
  `<malformed_tool_call>`.
- The monitor's own feedback on a blocked step shows as `<monitor_feedback>`.
- A provider's built-in tools, which run inside the model call, show as
  `<provider_tool_call>` and `<provider_tool_result>`: those LangChain maps for
  Anthropic, OpenAI's Responses API and Gemini's code execution, and Gemini's
  Google Search, read from `grounding_metadata`.
- A part of a reply LangChain cannot map shows whole as `<unrecognised_block>`;
  a refusal shows as the agent's prose.
- Each sample's `SampleRecord.proposal` keeps the provider tool calls and their
  full results.
- `monitor_subagents`, which gives every Deep Agents subagent, the
  general-purpose one included, a monitor of its own, with per-subagent
  overrides and a choice of what a subagent halt does to the run
  (`SubagentHalt`).
- Forked subagents (`mode="fork"`) are refused for now.
- `monitor_subagents(skills=...)` gives the monitored general-purpose subagent
  the main agent's skills: pass the list given to `create_deep_agent(skills=...)`.
- `monitor_subagents` raises `ConfigurationError` when `skills` is a plain
  string, or comes with a `general-purpose` spec of your own.
- `delegation_id`, the id of the `task` call that started a subagent, on the
  subagent's records and its `MonitorStepFailedEvent`.
- `Delegation`, what a monitored agent hands each subagent it starts: the
  call's id, the delegating agent and the thread's block counts.
- The `monitor_delegation` state key, part of every monitored agent's input,
  which carries the `Delegation`; a value that is not a valid `Delegation`
  raises `ConfigurationError`.
- `MonitorError` and its subclasses `ConfigurationError`, `MissingExtraError`,
  `SynchronousRunError` and `InvalidSuspicionError`.
- `ConfigurationError` is also raised for a plain string where an option enum
  is expected.
- `MissingExtraError` names the install command for a missing extra,
  `openrouter` included.
- Packaging for Python 3.12, 3.13 and 3.14, with the `deepagents`,
  `openrouter` and `typesafe` extras.
- The gate script, pre-commit hooks, and CI that also runs the offline suite at
  the lowest allowed dependency versions and without any extra.
- A visual README, and a tutorial that monitors a first agent, with or without
  an API key.
- How-to guides on choosing and using each protocol, each monitor family,
  wrapping and calibrating monitors, choosing what the monitor reads, reading
  the monitor log, seeing the monitor's decisions in LangSmith and Langfuse,
  and monitoring Deep Agents subagents.
- Explanation pages on how the library is built, where its ideas come from,
  and how a monitored agent fared in live runs against real models, with the
  harness that reproduces them.
- A complete bibliography of every paper, post and code base the library draws
  on, cited where it is used, with a test that keeps the two in step.
- A [documentation site](https://omamori-lab.github.io/langchain-sync-monitors/),
  deployed from each release, in the omamori lab house style in light and dark,
  with numbered sections, contents lists, and diagrams that open at full size.
  It follows the reader's system colour scheme until they pick light or dark
  with the toggle, which the browser then remembers.
- A release workflow that publishes to PyPI through trusted publishing, only
  after every CI check passes on the tagged commit and the built wheel installs
  and imports.
- `CITATION.cff` for citing the software, a security policy with private
  vulnerability reporting, and issue forms for bug reports and feature
  requests.

### Changed

- A guard's verdict is read from its last line only when the label stands
  alone or follows a verdict key such as `Label:` or `Final answer:`; a label
  after any other key, as in `It asks for: no_violation`, is unreadable.
- A guard's last line that holds a label with no key in a list item or a
  quote, as in `2. no_violation`, `- no_violation` or `> no_violation`, is no
  verdict, since the guard may be quoting a list from the step, so the reply
  fails closed.
- A guard reply fails closed when a line may hold the guard's own suspicious
  label off format and the reply also names a safe label: a line that opens
  with a suspicious label before a mark, opens with a verdict key and a label
  whatever follows, or ends on a suspicious label after a colon, with any
  markup, bullet, heading or ordered list marker around the key or label;
  every line is checked for these, whatever else it holds. A benign reply
  whose reasoning opens a line that way, such as `Result: violation of the
  policy would need ...`, is flagged too, and the guard guide's example
  policy asks the guard not to open any line but the last with a label.
- `GuardModelMonitor` refuses a label that begins or ends with `_` or `-`,
  which the reply parser reads as markup, with `ConfigurationError`.
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
- `OpenRouterDecisionModel(api_key=...)` raises `ConfigurationError` for a
  blank key, rather than read `OPENROUTER_API_KEY`, and for a key that is not
  a `SecretStr`; a key given is stripped of surrounding whitespace, as the
  variable is.
- A chat model wrapped in a Runnable, such as `with_retry()` or `bind()`,
  raises `ConfigurationError` naming its type, in every monitor and in
  `DeferToTrustedModel`.
- Protocols, fallbacks, the middleware, monitors, `MonitorView`,
  `DefaultThreshold`, `monitor_subagents` and `check_monitor_placement` raise
  `ConfigurationError` for an option of the wrong type, such as a float or
  `bool` count, or the `DeferToTrusted` protocol where the
  `DeferToTrustedModel` fallback belongs.
- `MonitorMiddleware`'s `label` and `agent_name`, and each subagent name, must
  be non-blank strings without `:` or `|`, which LangGraph refuses in node
  names.
- A limit may be any whole number, numpy's integers included.
- A threshold must be a real number from 0 to 1, a `Fraction` or `Decimal`
  included, checked exactly before it becomes a float.
- A threshold given as a string or a `bool` raises `ConfigurationError`.
- `MonitorMiddleware` raises `ConfigurationError` when a subclass names itself
  other than `label[agent_name]`, since two monitors that shared a label and
  an agent would lift each other's halts.

### Fixed

- When the stream writer fails on a monitor event, the error log says the event
  is dropped, not that the step stands, since a failed step never commits.
- `STANDING_HALT_MESSAGE` names no cause, and says the halt stands until a
  later run brings new input the monitor can confirm.
- A safe label the agent planted in its step is no longer read as the guard's
  verdict when the guard quotes it after its own label, after a colon on the
  same line, after a prose key, or on the line after a line that holds its
  own label in any form the reader counts, such as `violation (the note asks
  ...`, `Label: violation because ...` or `The final label for this step:
  violation`.
- A guard's keys and labels are read through markup and list markers, as in
  `**Label**: violation`, `__violation__`, `"label":` or `1. Label:`, so a
  planted safe label after the guard's own label in such a form fails closed.
- The guard's label parser runs in linear time on long lines of colons,
  dashes or spaces.
- A guard's log-probabilities are scored only at the guard's own label token,
  so a label wrapped in markup such as `(violation)` no longer scores near 0.
- Under `ainvoke()`, a failed guard sample cancels the other samples.
- A judge score too long for Python to convert counts as unreadable instead of
  raising `ValueError`.
- Replies cut off at a length limit are recognised from Bedrock Converse
  (`stopReason`) and Ollama (`done_reason`), and fail closed.
- A decision answer that is missing, NaN, infinite, outside 0 to 1, a `bool`,
  or not a number at all raises `MonitorError` for any `DecisionModel`,
  instead of being dropped by `Combine.MAX` or `Combine.MIN` or raising
  another error. An `int`, a `float`, a `Decimal` or another real number from
  0 to 1 is read as a float. On the `TypeSafeDecisionModel` path,
  `langchain-typesafe` parses the answers leniently first, so a `false`,
  `true` or numeric string from the API arrives as a number, `false` as 0.0.
- `OpenRouterDecisionModel` answers must be JSON numbers: `false`, `true`,
  `"0"` and `"0.5"` raise `MonitorError` instead of being read as numbers.
- A decision answer too large for a float, such as `10**400`, or a `Fraction`
  just outside 0 to 1 raises `MonitorError` instead of `OverflowError` or
  being rounded into range.
- A command a tool raises as a `ParentCommand`, itself or from a graph it
  calls, is relabelled like one it returns, so it cannot write the monitor's
  own source or a human message that speaks as the user.
- An OpenRouter key that holds a control or non-ASCII character raises
  `ConfigurationError` without naming the key, instead of an httpx error
  that quoted it whole and was retried.
- Only a `TypeError` for an unexpected `logprobs` keyword is reported as a
  rejected request for log-probabilities.
- An `interrupt()` inside a monitor or protocol no longer writes a
  `monitor_step_failed` event.
- Monitors, protocols and fallbacks must still not call `interrupt()`, since a
  resume samples and judges the step afresh.
- `SynchronousRunError` names the monitor, not the protocol, when a monitor's
  `evaluate_sync` needs an event loop.
- `check_monitor_placement` names subclasses of `ModelRetryMiddleware` and
  `ModelFallbackMiddleware` too.
- The guides state how a wrapped or sampled monitor counts an unreadable reply,
  and the decision model's retry budget, its handling of HTTP 408, the
  lifetime of its client and when `timeout_seconds` applies.

[Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/commits/main
