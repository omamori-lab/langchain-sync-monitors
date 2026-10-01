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

## [0.1.0] - 2026-10-01

### Added

- `MonitorMiddleware`, which puts a monitor and a control protocol around every
  model call of a LangChain agent or Deep Agent, under both `invoke()` and
  `ainvoke()`.
- Each committed step adds one `StepRecord`, with every sample judged, to
  `monitor_log`, and is written to `stream_mode="custom"` as a
  `MonitorStepEvent`. Records hold plain values, numpy scores included, so a
  checkpointer stores them.
- A step that raises before it is committed writes a `MonitorStepFailedEvent`
  to `stream_mode="custom"`, with the samples judged so far, if any; the
  middleware logs those samples as a warning and raises the error again.
  LangGraph's own control flow, such as `interrupt()`, passes through without
  an event.
- Only committed steps reach `stream_mode="messages"`, each whole once the
  protocol has decided; rejected samples, blocked attempts with hidden
  feedback and the built-in monitors' own model calls never stream.
- `check_monitor_placement`, which warns with a `MonitorPlacementWarning` when
  middleware inside a monitor could return commands, when middleware that
  retries or falls back sits outside it, subclasses of `ModelRetryMiddleware`
  and `ModelFallbackMiddleware` included, and when one that retries or answers
  failed tool calls, such as `ToolRetryMiddleware`, sits anywhere in the list.
- `check_monitor_placement` also warns about stacked monitors that would lose
  or duplicate records: a monitor, or a middleware between two monitors,
  inside one whose protocol can call the model more than once in a step, or a
  monitor inside another that commits its blocked attempts with
  `FeedbackVisibility.IN_TRANSCRIPT`. `monitor_subagents` warns about such
  stacks in the subagents it monitors.
- The middleware's `before_agent`, `before_model`, `after_model` and
  `after_agent` hooks, which show as graph nodes in a trace and add graph
  steps that an explicit `recursion_limit` counts: two per agent step, however
  many samples it draws, and two per run.
- Seven private state keys, `monitor_task_messages`,
  `monitor_seen_human_messages`, `monitor_run_inputs`,
  `monitor_rewritten_inputs`, `monitor_run_open`, `monitor_inputs_at_halt` and
  `monitor_subagent_returns`, which never enter a run's output or an isolated
  subagent's input but do appear in `stream_mode="values"`,
  `stream_mode="updates"` and `get_state`.
- A halted step ends the run, in agents with structured output and no tools
  too.
- A halt stands until the thread records a new run input: each later step,
  as when Deep Agents' `RubricMiddleware` sends the run back to the model,
  halts again without calling the untrusted model, and its record holds no
  samples. No hook that rewrites, trims or removes messages, no harness nudge
  and no note lifts it.
- `ProviderToolWarning`, raised once per middleware when a model call receives
  a server tool that Anthropic, OpenAI or Gemini runs itself, before the
  monitor can judge the step. It also reads tools bound on the model of an
  agent with no tools of its own and no `response_format`, the only case in
  which those tools reach the model.
- Named spans in any LangChain tracer, LangSmith and Langfuse among them:
  `monitor step` holds a step's samples, `monitor judgement` each verdict,
  `monitor classifier` each Decisions API request, and `monitor decision` the
  outcome as a tag, with `monitor:flagged` on a flagged step. Each carries
  flat `monitor_` metadata naming the step, with the delegation id inside a
  subagent.
- The built-in monitors' own model calls are named `monitor call` in traces,
  with their model still shown, and carry `ls_message_view_exclude`, which
  keeps them out of LangSmith's Trajectory view.
- The spans add no tag or metadata to the LangChain runs inside them, though
  LangSmith copies their metadata into its own `traceable` runs inside them.
  They leave every stream except `astream_events` and `astream_log`
  unchanged, and are not opened when no callback handler is attached.
- Control protocols `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample`,
  with sequential or parallel resampling, and `AutoMode`, with the
  `DeferToTrustedModel` and `HaltRun` fallbacks.
- `AutoMode` blocks a suspicious step, tells the agent why with a
  `FeedbackTemplate`, and lets it try again. After three blocks in one step,
  or twenty in the conversation thread, Claude Code's limits, its
  `when_limit_reached` fallback writes the step, which by default halts the
  run.
- The default feedback tells the agent that none of its tools ran the blocked
  step, which stays true when a provider's built-in tool in that step already
  ran inside the model call.
- Auto Mode's total counts the whole conversation thread, subagents and nested
  subagents included: a subagent starts from the thread's count, so
  delegating again does not reset it, and once blocks inside subagents reach
  the total, the parent's next step goes to the fallback.
  `AutoMode(max_total_blocks=math.inf)` turns the total off.
- `PendingStep.blocks_in_thread` and `PendingStep.new_subagent_blocks`, which
  give a protocol these counts.
- Default thresholds, 0.6 to defer, block or flag and 0.9 to audit, which raise
  `UncalibratedThresholdWarning` until you set your own.
- A flag marks a step for audit and never blocks it.
- `CachedResampleWarning`, raised when a response cache would turn resamples
  into copies of the first sample.
- Monitors for three model families: `LLMMonitor` for chat judges, with
  `DEFAULT_MONITOR_PROMPT`; `GuardModelMonitor` for guard models, scored from
  log-probabilities or sampling; and `DecisionModelMonitor` for decision
  models, through `OpenRouterDecisionModel` or `TypeSafeDecisionModel`.
- The `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
  `CalibratedMonitor` reports a step's suspicion as the smoothed conformal
  rank of its score among honest-run scores, so a threshold of 0.99 flags 1%
  of new honest steps on average.
- A judge's verdict is read only where it ends the reply, the reasoning block
  then exactly one score tag, so a score the agent plants in its step cannot
  become the verdict.
- A judge's reply cut off at a length limit, as OpenAI, Anthropic, Gemini,
  OpenRouter, Bedrock Converse and Ollama report it, or of any other shape, is
  asked for again and then fails closed.
- A guard's label is read only where its format puts it: on its last line,
  alone or after a verdict key such as `Label:` or `Final answer:`, through
  markup and list markers. `GuardModelMonitor` raises `ConfigurationError` for
  a label that begins or ends with `_` or `-`, which the reader takes for
  markup.
- A guard's reply fails closed when its last line holds a label with no key in
  a list item or a quote, or when any line may hold its suspicious label off
  format and the reply also names a safe label, so a safe label the agent
  planted cannot become the verdict. A benign reply whose reasoning opens a
  line with a suspicious label fails closed too.
- `GuardScoring.AUTO` samples when the model gives log-probabilities without
  alternatives, where `GuardScoring.LOG_PROBABILITIES` raises
  `ConfigurationError`. Both raise it, naming `SAMPLE_FRACTION`, when the chat
  model rejects the `logprobs` keyword, as `ChatAnthropic` and
  `ChatBedrockConverse` do.
- The library adds no score, probability or percentile to a verdict's reason,
  which Auto Mode shows the agent; they stay in `monitor_log`. A guard's
  reason from log-probabilities names one of three findings instead: it breaks
  the policy, the guard was uncertain, or it follows the policy.
- A decision answer that is missing, NaN, infinite, a `bool`, or not a real
  number from 0 to 1 raises `MonitorError` for any `DecisionModel`, and
  `OpenRouterDecisionModel` takes only JSON numbers.
- `OpenRouterDecisionModel` refuses an `api_key` that is blank, not a
  `SecretStr`, or holds a control or non-ASCII character, with a
  `ConfigurationError` that does not quote the key. A key is stripped of
  surrounding whitespace, as `OPENROUTER_API_KEY` is.
- `MonitorView` and `Channel`, which choose what a monitor reads of the
  history; the default follows Claude Code's auto mode classifier. Under any
  view, the step being judged shows its tool calls and malformed calls, and
  `most_recent_entries` keeps every message from the task author.
- `DEFAULT_MONITOR_PROMPT` explains every tag the judge can read, provider
  tool calls and unrecognised blocks included, and tells the judge that a note
  never removes a safeguard, such as asking the user first, logging or
  reporting.
- Only the human messages a run receives as its input render as `<user>`, or
  `<delegator>` inside a subagent. Every other human message, such as a
  conversation summary or a harness nudge, renders as `<context_note>`, which
  authorises nothing.
- Every run's input reaches the judge whole and in order on every step, even
  after summarisation or a tool has taken it out of the model request, and
  even when Deep Agents shows it to the agent only as a preview, since
  `monitor_run_inputs` keeps its text, which grows the state by the size of
  the user's turns.
- After a run that stopped before its end, such as one whose model call raised
  or whose client cancelled it, the next run's new messages, the user's own
  included, are notes from `unconfirmed_input` for the rest of the thread: the
  judge takes no approval from them, but still applies a limit they set. So is
  a nudge that a middleware listed after the monitor wrote just before the
  stop.
- The monitor writes a note's tag into the message in the agent's state, by
  id, so a saved or replayed history keeps it; `stream_mode="updates"` can
  then carry the message twice, so merge messages by id.
- Under every view that shows a tool call, the judge reads a call that failed
  or did not run as `<tool_error>`, one with unreadable arguments as
  `<malformed_tool_call>`, and the monitor's own feedback as
  `<monitor_feedback>`. A part of a reply LangChain cannot map shows whole as
  `<unrecognised_block>`; a refusal shows as the agent's prose.
- A provider's built-in tools show as `<provider_tool_call>` and
  `<provider_tool_result>`: those LangChain maps for Anthropic, OpenAI's
  Responses API and Gemini's code execution, and Gemini's Google Search, read
  from `grounding_metadata`. Each sample's `SampleRecord.proposal` keeps their
  calls and full results.
- A tool's `Command` writes to the state keys only the monitor writes, every
  monitor key but `monitor_log`, are dropped with a warning.
- A tool's writes to `monitor_log` are checked record by record: an
  `Overwrite` only adds its records, and a write that starts with the whole
  current log adds only the records after it. A record that claims to be a
  step of the calling agent, or is not a whole `StepRecord` with counts of
  zero or more, is kept out with a warning, and counts as a halted subagent
  when it is a halt or malformed.
- A message a tool writes, through a `Command` it returns or a `ParentCommand`
  it raises, cannot speak as the user or carry a source only the monitor
  writes, `monitor` or `unconfirmed_input`. One it writes back by id keeps its
  author only if the whole message, metadata included, is unchanged.
- A tool call that reuses the id of the call that started its agent, and
  returns records under that agent's name, raises `ConfigurationError`.
- A malformed record read from `monitor_log` raises `MonitorError` naming it,
  its position and the fields at fault.
- `monitor_subagents`, which gives every Deep Agents subagent, the
  general-purpose one included, a monitor of its own, with per-subagent
  overrides and a choice of what a subagent halt does to the run
  (`SubagentHalt`). It refuses forked subagents (`mode="fork"`) for now.
- `monitor_subagents(skills=...)` gives the monitored general-purpose subagent
  the main agent's skills: pass the list given to
  `create_deep_agent(skills=...)`. A plain string, or a `general-purpose` spec
  of your own beside it, raises `ConfigurationError`.
- `Delegation`, which a monitored agent hands each subagent it starts in the
  `monitor_delegation` state key: the `task` call's id, the delegating agent
  and the thread's block counts. A value that is not a valid `Delegation`
  raises `ConfigurationError`.
- `delegation_id`, the id of the `task` call that started a subagent, on the
  subagent's records and its `MonitorStepFailedEvent`.
- A record is an agent's own only when it names both the agent and its
  delegation, so a subagent that shares its parent's name never passes for
  the parent: `SubagentHalt` applies to it, the parent numbers its own steps
  apart from it, and the halt message names it by the `subagent_type` of the
  call that started it.
- `MonitorError` and its subclasses `ConfigurationError`, `MissingExtraError`,
  `SynchronousRunError` and `InvalidSuspicionError`. `MissingExtraError` names
  the install command for a missing extra.
- Protocols, fallbacks, the middleware, monitors, `MonitorView`,
  `DefaultThreshold`, `monitor_subagents` and `check_monitor_placement` raise
  `ConfigurationError` for an option of the wrong type, such as a plain string
  where an enum is expected, a float or `bool` count, or the `DeferToTrusted`
  protocol where the `DeferToTrustedModel` fallback belongs.
- A limit may be any whole number, numpy's integers included. A threshold must
  be a real number from 0 to 1, a `Fraction` or `Decimal` included, checked
  exactly before it becomes a float; a string or a `bool` raises
  `ConfigurationError`.
- A monitor's or `DeferToTrustedModel`'s chat model must be the model itself:
  one wrapped in a Runnable, such as `with_retry()` or `bind()`, raises
  `ConfigurationError` naming its type.
- `CalibratedMonitor(honest_scores=...)` and `DecisionModelMonitor(questions=...)`
  accept any iterable, a generator included.
- `MonitorMiddleware`'s `label` and `agent_name`, and each subagent name, must
  be non-blank strings without `:` or `|`, which LangGraph refuses in node
  names. A subclass that names itself other than `label[agent_name]` raises
  `ConfigurationError`, since two monitors that shared a label and an agent
  would lift each other's halts.
- Packaging for Python 3.12, 3.13 and 3.14, with the `deepagents`,
  `openrouter` and `typesafe` extras, published to PyPI through trusted
  publishing.
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
  deployed from each release, in light and dark.
- `CITATION.cff` for citing the software, a security policy with private
  vulnerability reporting, and issue forms for bug reports and feature
  requests.
- For contributors: one gate script, pre-commit hooks, and CI that also runs
  the offline suite at the lowest allowed dependency versions and without any
  extra.

### Known limits

- A provider's built-in tools, such as web search or code execution, run
  inside the model call, before the monitor judges the step, and again for
  every sample drawn. `ProviderToolWarning` does not see a server-side feature
  set on the model itself, such as OpenRouter's web plugin, tools an
  integration has already turned into its SDK's objects, or tools bound inside
  a wrapper such as `with_fallbacks(...)` or queued on a configurable model.
- An untagged human message that a middleware listed before the monitor writes
  from its `before_agent` hook, or from its `after_agent` hook without a
  return to the model, counts as a run's input and lifts a halt.
- A subagent whose run raises returns no records: its steps, and its blocks
  towards Auto Mode's total, reach the parent only if the failed run is
  resumed from its checkpoint. Subagents that run in parallel can together
  pass the total.
- `DEFAULT_MONITOR_PROMPT` asks a chat judge to keep its score out of the
  reasoning that becomes the reason, but the judge's own wording may still
  hint at one.
- `langchain-typesafe` parses decision answers leniently before
  `TypeSafeDecisionModel` sees them, so a `false`, `true` or numeric string
  from the API arrives as a number, `false` as 0.0.
- Monitors, protocols and fallbacks must not call `interrupt()`, since a resume
  samples and judges the step afresh.
- The rest are listed in
  [Known limits and open paths](https://omamori-lab.github.io/langchain-sync-monitors/explanation/design/#known-limits-and-open-paths).

[Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/omamori-lab/langchain-sync-monitors/releases/tag/v0.1.0
