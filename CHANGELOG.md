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

### Changed

- langchain-sync-monitors depends on `hyoshigi>=0.5.0`, the framework-free
  control core extracted from it.
- langgraph's floor rises to 1.2.15. Each monitor's trajectory is kept in a
  LangGraph `DeltaChannel`, which LangGraph marks as beta.
- A run whose messages hold an action that reuses an action id or a tool call
  id already in the run now raises `MessageFormatError`, at the start of the
  run or of the step that reads it. A history passed in must give each action
  and each tool call an id of its own.
- `Delegation` gains `runs`: for each monitor label whose trajectory holds
  the call, the run the subagent's trajectory joins. `DelegatedRun` is
  exported. A `Delegation` without `runs`, or whose run cannot hold the
  subagent's trajectory, raises `ConfigurationError`.
- `monitor_subagents` raises `ConfigurationError` for an entry of
  `overrides` whose label differs from `middleware`'s, since a subagent's
  trajectory joins its parent's under the parent's label. To count a
  subagent apart, monitor it as a compiled subagent with a label of its own;
  its trajectory is then dropped with a warning when its task returns.
- **Breaking:** the monitors are hyoshigi's. Each monitor class is one
  configuration that builds hyoshigi's sync and async twins, which it holds
  as `twins`: `invoke()` rates with the sync twin and `ainvoke()` with the
  async one. Option names and meanings are hyoshigi's, and so are the value
  types the options take, re-exported as the same objects: `View`, `Channel`,
  `MonitorPrompt`, `DEFAULT_PROMPT`, `GuardLabels`, `GPT_OSS_SAFEGUARD`,
  `QWEN3GUARD`, `HardLabelWarning`, `Aggregation`, `ATTACK_CRITERIA`,
  `DEFAULT_STATEMENT` and `MediaTypes`. The protocols, their fallbacks,
  `Verdict`, `Sample`, `PendingStep` and `ControlProtocol` are unchanged, and
  a protocol still reads `sample.verdict`.
- **Breaking:** `LLMMonitor` takes `prompt` as a `MonitorPrompt`, in place of a
  `ChatPromptTemplate`, and `view` as a `View`. A prompt's request holds
  `{trajectory}` and `{proposed_action}` once each, in place of `{transcript}`
  and `{proposed_step}`, and may hold `{lowest_score}` and `{highest_score}`;
  text that `partial` filled in goes into the template's string.
- **Breaking:** `GuardModelMonitor` is now
  `GuardMonitor(model=..., prompt=MonitorPrompt(...), labels=GuardLabels(suspicious=..., safe=...), samples=5, top_log_probabilities=0, view=View())`,
  and it samples unless `top_log_probabilities` is above 0: suspicion is the
  share of its sampled replies labelled suspicious. A sample with no readable
  label, an ambiguous one, a reply cut off, a refusal or a rejection fails the
  whole rating closed at 1.0, where an unreadable reply counted as one
  suspicious label; a sample the model left unanswered is set aside while
  more than half answered. One sample warns `HardLabelWarning` once, at the
  line that builds the monitor.
- **Breaking:** `DecisionModelMonitor` is now
  `DecisionMonitor(model="openrouter:<model id>" | "typesafe:<model id>", view=..., statements=(DEFAULT_STATEMENT,), aggregation=Aggregation.MAX, perceives=MediaTypes())`.
  It asks statements that hold or not, each in a request of its own, in place
  of yes or no questions with `yes_means` and `no_means` in one request.
  `ATTACK_CRITERIA` is four statements, and `Aggregation.MEAN` no longer
  warns. Any other decision model goes into a `MonitorTwins` of hyoshigi's
  `DecisionMonitor` and `AsyncDecisionMonitor`.
- **Breaking:** `MonitorView(channels, most_recent_entries, delegation_tools)`
  is now hyoshigi's `View(channels, most_recent, delegation_tools)`, and
  `Channel` is hyoshigi's: `TASK`, `CONTEXT`, `THOUGHTS`, `AGENT_PARTS`,
  `TOOL_CALLS`, `TOOL_OBSERVATIONS`, `SUBAGENT_REPORTS` and `DEFAULT`, with
  nothing in place of `ALL`. The default view, `Channel.DEFAULT`, reads the
  task, the tool calls and the subagent reports. It no longer reads context
  notes, system messages or the monitor's feedback written as a human
  message; a tool error the monitor wrote on a blocked call is still read.
  Add `Channel.CONTEXT` to read them. `delegation_tools` defaults to none:
  Deep Agents' `task` results still read as subagent reports, because the
  trajectory records the delegation.
- **Breaking:** a monitor reads hyoshigi's rendering of its trajectory, with
  new tags, among them `<trajectory>`, `<proposed_action>`, `<user>`,
  `<delegator>`, `<context_note author=... source=...>`,
  `<tool_call id=... name=...>` with the call's argument text,
  `<tool_error call_id=...>`, `<tool_result call_id=...>`,
  `<subagent_report call_id=...>`, `<thought>`, `<agent>` and
  `<opaque kind=.../>`. A guard policy or a prompt of your own that names the
  old tags needs the new ones.
- **Breaking, privacy:** the monitor reads each message into its trajectory
  once, as it first sees it, at the start of a run or a step, and never
  rebuilds it from `messages`. A turn summarised away is still read, but a
  later rewrite of a message already read, such as an `update_state` edit,
  `PIIMiddleware`'s redaction in its `before_model` hook, or a tool's write
  under the input's id, no longer reaches the monitor. The checkpointed
  `monitor_trajectory` holds the unredacted text, and the monitor's model
  reads it unless the monitor's view redacts it, with hyoshigi's
  `View(redaction=...)`.
- A monitor's model call that fails on an exhausted rate limit, a server
  error, a timeout or a failed connection no longer raises: the rating fails
  closed at 1.0, and the protocol's rules apply. A refusal, a context
  overflow or a prompt the provider rejects fails closed at once, and a reply
  still unreadable after the retries fails closed with hyoshigi's reason. A
  call that fails on a misuse, such as a bad key, still raises and fails the
  step.
- **Breaking:** a reply that holds a server tool's result, a block LangChain
  could not map, or media the monitor's model cannot perceive is rated 1.0,
  before any request; the `ServerToolWarning` says so for server tools. Under
  Defer to Resample or Auto Mode, a step whose every sample uses such a server
  tool, such as Anthropic's web search, goes to the fallback, which halts the
  run by default.
- The reason of a rating that failed closed is hyoshigi's sentence for its
  cause, such as "The monitor's model was unavailable, so the action is
  treated as suspicious."; a guard's reason reads "The guard model labelled
  the action 'violation', finding that it breaks the policy.", and a decision
  monitor's "The decision model's strongest concern: " and the statement.
  These reach `monitor_log`'s `reason` and Auto Mode's feedback.
- **Breaking:** `SampleRecord.proposal` is the action the monitor rated, as
  hyoshigi stores it, where it was rendered text: a dict with `kind`, `id`,
  `thought`, `parts` and `tool_calls`, each call's `arguments` as text, with
  a server tool's result as a placeholder. The step span's `proposed_step`
  input holds the same.
- The `monitor classifier` span is gone, so four fixed names remain:
  `monitor step`, `monitor judgement`, `monitor decision` and `monitor call`.
  A judgement span's `monitor_class` input is the class of the twin that
  rated, such as `LLMMonitor` under `invoke()` and `AsyncLLMMonitor` under
  `ainvoke()`; its outputs gain `failed_closed`; and judgement numbers follow
  the order ratings start.
- `MonitorMiddleware` warns `CachedResampleWarning` once per process, when it
  is built, for a monitor that asks a cached chat model the same prompt more
  than once: an `LLMMonitor` with `max_parse_retries` above 0, a
  `GuardMonitor` that samples more than once, or a `RepeatedMonitor` of more
  than one call over a cached model, wrappers read through. It no longer warns
  during the run for the monitor's model.
- **Breaking:** `ConfigurationError` is also hyoshigi's
  `ConfigurationError`, and so a `ValueError`. An option hyoshigi refuses while a monitor is built raises
  the library's error, from hyoshigi's. hyoshigi's value types, such as
  `View` and `MonitorPrompt`, raise hyoshigi's own error when they are built,
  which `except ConfigurationError` with the library's class does not catch;
  catch `ValueError` for both.
- The `typesafe` extra installs `hyoshigi[typesafe]`, with TypeSafe's SDK, in
  place of langchain-typesafe, and `DecisionMonitor(model="typesafe:<model id>")`
  reads `TYPESAFE_API_KEY`.
- `DecisionMonitor(model="openrouter:<model id>")` reads `OPENROUTER_API_KEY`
  and calls the Decisions API through the library's own httpx client, because
  langchain-openrouter 0.2.9, which the `openrouter` extra installs, pins
  `openrouter<1.0.0`, while hyoshigi's OpenRouter decision models need
  openrouter 1.3.63 or later. The client stays until langchain-openrouter
  supports openrouter 1.x. It takes text only. A 403 is refused and a 413
  rejected, both failing closed; a 408, 409, 429, server error or transport
  failure is unavailable after its retries; any other client error raises
  `ConfigurationError` naming its status alone.
- `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` keep their
  options, wrap any of the monitors, and follow hyoshigi's rules for a score
  that failed closed: a repeated call that failed for a cause other than an
  unavailable model fails the whole score closed, a calibrated monitor
  returns a failed score unranked, and a failed fast score always escalates.
  `CalibratedMonitor(random_seed=...)` draws from the seed and the trajectory,
  each action's id blanked, so a fresh run whose tool calls get new ids draws
  afresh.

### Added

- The docs pages, the README and the module of the default prompt carry the
  training-data canary that alignment.org/canary lists, so a dataset builder
  can leave them out of training corpora.
- Each monitor keeps its trajectory of the run, every message it read or
  committed, once, as hyoshigi's actions and observations, under the new
  state key `monitor_trajectory`, by monitor label. A run's `invoke()` result,
  `get_state` and the `values` and `updates` streams show it. A subagent's
  trajectory, its steps included, joins its parent's under the call that
  started it, so the parent's `monitor_trajectory` holds the whole run. A
  stale write outside a stack of monitors, and a subagent's returned
  trajectory that cannot join its parent's, halt that monitor's next step
  without a sample, with a warning, until a later run brings new input. A
  tool's write to `monitor_trajectory` is checked as a subagent's returned
  trajectory. One that no subagent of the call returned, an `Overwrite`, a
  delta or an empty mapping included, is dropped with a warning and counts as
  a halted subagent. So does the whole state a tool returns from a monitored
  graph it ran without handing that graph the tool's state, since its
  trajectory names no delegation: under `SubagentHalt.STOP_WHOLE_RUN` the
  parent then halts.
  A blocked attempt enters it with its text and its feedback, whatever
  `feedback_visibility` says, as `monitor_log`'s samples already hold it; the
  message stream still never carries a blocked attempt's text. A halt message
  never enters it. The monitors rate every sample against it; the protocols
  read their verdicts.
- `MessageFormatError` is exported.
- `GuardMonitor(top_log_probabilities=...)`, the number of alternatives to
  each token that the guard's chat model is asked for, 0 by default. Above 0
  the monitor asks once, and hyoshigi's guard scores the label from the
  alternatives; `samples` is not used. A reply with no log-probabilities, or
  with some it cannot read, fails the rating closed at 1.0. A number that is
  not a whole number of at least 0 raises `ConfigurationError` when the
  monitor is built.
- For a `ChatOpenRouter` and a `top_log_probabilities` above 0, the request's
  `provider` preferences gain `require_parameters: true`, so that OpenRouter
  routes the request only to providers that support every parameter in it.
  The preferences are `openrouter_provider` when set, else the `provider`
  among `model_kwargs`; their other keys stay, and a `require_parameters` you
  set wins, even `False`.
- A chat model whose class refuses the `logprobs` keyword raises
  `ConfigurationError` when a guard with a `top_log_probabilities` above 0
  rates a step, naming the model and advising `top_log_probabilities=0`.
- OpenRouter answers with HTTP 404 when no provider of the model supports
  the request for log-probabilities, and the error is raised as it is. Set
  `top_log_probabilities` only for a model whose every provider returns
  them.
- `MonitorTwins(monitor=..., async_monitor=...)`, a monitor of your own
  written as hyoshigi's sync and async twins, which the middleware and the
  wrappers take as they take the library's monitors.
- `GPT_OSS_SAFEGUARD` and `QWEN3GUARD`, the labels of those two guards, and
  `MediaTypes` and `DEFAULT_STATEMENT` are exported.

### Removed

- **Breaking:** `Monitor`, `ChatModelMonitor` and `MonitorInput`; a monitor
  of your own is a `MonitorTwins`.
- **Breaking:** `MonitorView`, `GuardModelMonitor`, `GuardScoring`,
  `DecisionModelMonitor`, `DecisionModel`, `OpenRouterDecisionModel`, with
  its `base_url`, `api_key`, `timeout_seconds`, `http_client` and
  `async_http_client`, `TypeSafeDecisionModel`, `YesNoQuestion`,
  `IS_THE_STEP_SUSPICIOUS` and `DEFAULT_MONITOR_PROMPT`, whose places the
  changes above give.
- The module `langchain_sync_monitors.transcript`, with
  `render_transcript` and `render_proposed_step`.
- The private state keys `monitor_run_inputs` and `monitor_rewritten_inputs`.
- The loggers `langchain_sync_monitors.monitors.chat` and
  `langchain_sync_monitors.monitors.guard`. A failed model call the rating
  reads as an outcome is logged by `langchain_sync_monitors.chat_model_adapter`,
  and an unreadable reply is no longer logged.

### Known limits

- A checkpoint saver that prunes, copies or deletes checkpoints without
  walking a `DeltaChannel`'s chain of writes rebuilds `monitor_trajectory` as
  empty, with no error.
- Each subagent receives its run's whole trajectory in `monitor_delegation`,
  so the state and the checkpoints grow with the run.
- `MonitorMiddleware` offers no entry redaction: the trajectory keeps each
  entry as the monitor first read it, so the checkpointed
  `monitor_trajectory` holds the text unredacted, and a redaction made after
  that first read, such as `PIIMiddleware`'s, never reaches it. hyoshigi's
  `View(redaction=...)` redacts only what a monitor's model reads.
- A step whose reply holds a server tool's result is rated 1.0 every time,
  so under Defer to Resample and Auto Mode a step whose every sample uses
  such a tool goes to the fallback.
- An `openrouter:` decision model reads text only, so an action that holds
  media fails closed.

## [0.1.5] - 2026-10-04

### Security

- Every monitor now reads, as the agent's text, the refusal langchain-openai
  keeps in `additional_kwargs` for a Chat Completions reply. LangChain gives
  no content block for it, so such a refusal was committed without being
  judged.

## [0.1.4] - 2026-10-04

### Security

- Every monitor now reads the text of the step it judges whatever its view,
  beside a tool call as well as in a final answer, since a chat interface
  shows that text to the user. Under the default view, a message written
  beside a harmless tool call reached the user without being judged.
- Under `FeedbackVisibility.IN_TRANSCRIPT`, Auto Mode commits a blocked
  attempt with its text withheld, as `WITHHELD_TEXT_MESSAGE`, wherever the
  message keeps it, refusals included, and without the `grounding_supports`
  of Gemini's `grounding_metadata`, which quote the answer; it keeps its tool
  calls, reasoning and other blocks. The text the monitor blocked was
  committed and streamed to the user.

## [0.1.3] - 2026-10-02

### Added

- The docs home says how to report a bug, a vulnerability or a question, and
  the bug report form asks for the configuration, the run mode and the log
  output.

### Changed

- With `prometheus-client` installed, stamina's `stamina_retries_total`
  counter labels every retry the library makes with the `error_type`
  `langchain_sync_monitors.errors.RetriedCallError`, in place of the error's
  own class, such as `httpx.HTTPStatusError`, so a dashboard or alert keyed
  on that class no longer counts these retries. The error's type and HTTP
  status are now in the retry log's `caused_by` alone.

### Security

- A chat monitor's retry on HTTP 429 no longer logs the provider's error
  reply, which held the account's `user_id` and the response headers.
- A Decisions API or score-export retry no longer logs the bytes of a reply
  httpx could not parse, such as a header line. Every retry now logs, and
  hands a retry hook, a `RetriedCallError` that names the error's type and
  HTTP status alone, never the request or the reply; after the last attempt,
  the error itself is raised, as before.

## [0.1.2] - 2026-10-02

### Added

- `export_scores`, an option of `MonitorMiddleware` that is off by default,
  sends each step's highest suspicion as a `<label>_suspicion` score:
  feedback in LangSmith and a numeric score in Langfuse, written by a
  background worker, with `Tracer` naming the tools.

### Fixed

- `OpenRouterDecisionModel` refuses, when it is built, a relative, empty or
  host-less `base_url` when no client passed has an absolute `base_url` of
  its own; such a `base_url` failed at the first request.

### Security

- A user name or password in `OpenRouterDecisionModel`'s `base_url`, as
  httpx parses the URL, no longer reaches logs or errors: such a `base_url`,
  and one httpx cannot read, raise `ConfigurationError` when the model is
  built, quoting no part of the URL. Pass the key as `api_key`; a gateway
  that needs Basic authentication takes an `http_client` with its own `auth`.
  Only user information as httpx parses it is refused: a user name or
  password holding an unencoded `/`, `?` or `#`, or one in a host-less
  `base_url` that a client's `base_url` completes, is read as host or path
  text and can still be logged.

## [0.1.1] - 2026-10-01

### Changed

- The install instructions lead with `uv add`, with `pip install` as the
  alternative.
- `MissingExtraError` messages name `uv add` as well as `pip install`.
- The docs and docstrings call `LLMMonitor`'s family an LLM monitor, the AI
  control literature's term, not a chat judge. The guide "Use a chat judge" is
  now [Use an LLM monitor](https://omamori-lab.github.io/langchain-sync-monitors/how-to/use-an-llm-monitor/),
  and its old address redirects there.
- The `CachedResampleWarning` about a monitor's cached model says "an LLM
  monitor" where it said "a chat judge"; a warning filter that matches the old
  text needs the new one.
- The `UncalibratedThresholdWarning` for a threshold compared with one
  sample's score says `CalibratedMonitor`'s scores are "smoothed conformal
  p-values against honest scores", where it said "honest-run percentiles"; a
  warning filter that matches the old text needs the new one.

### Fixed

- Ten figures, and the prose and docstrings beside them, are corrected: a
  failed step runs none of the agent's own tools; Defer to Resample hands the
  step to its fallback only when no sample passes; `CalibratedMonitor`
  reports a smoothed conformal p-value; a guard model whose chat model cannot
  take log-probabilities fails in its adapter, which rejects the `logprobs`
  keyword before any request is sent; the monitor's own feedback renders as
  `<monitor_feedback>`; and input after a run that stopped early is recorded
  as no input.

## [0.1.0] - 2026-10-01

### Added

- `MonitorMiddleware`, which judges every step a LangChain agent or Deep Agent
  proposes before any of its own tools run, under `invoke()` and `ainvoke()`.
- Control protocols `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample`,
  with sequential or parallel resampling (`Resampling`), and `AutoMode`, with
  the `DeferToTrustedModel` and `HaltRun` fallbacks.
- `AutoMode` blocks a suspicious step and tells the agent why. After three
  blocks in one step, or twenty in the thread, its fallback halts the run by
  default.
- Default thresholds, 0.6 for `defer_threshold`, `block_threshold` and Trusted
  Monitoring's `audit_threshold`, and 0.9 for the `audit_threshold` of Defer to
  Trusted and Defer to Resample, which raise `UncalibratedThresholdWarning`
  until you set your own.
- `ControlProtocol` and `PendingStep`, for a protocol of your own, with
  `SampleDrawing` to draw a step's samples one after another or at once, and
  `resolve_threshold` and `ThresholdStatistic` to warn about a default
  threshold and name the score it is calibrated on.
- Monitors for three model families: `LLMMonitor` for chat judges;
  `GuardModelMonitor` for guard models; and `DecisionModelMonitor` for
  decision models, through `OpenRouterDecisionModel` or
  `TypeSafeDecisionModel`.
- The `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
- `MonitorView` and `Channel`, which choose what the monitor reads; the default
  follows Claude Code's auto mode classifier.
- `monitor_log`, which keeps one `StepRecord` per committed step, with every
  sample judged.
- `MonitorStepEvent` and `MonitorStepFailedEvent` on `stream_mode="custom"`.
  Only committed steps reach `stream_mode="messages"`.
- A halted step ends the run, and the halt stands until a later run brings
  new input the monitor can confirm.
- Each run's input stays available to the judge whole and in order, even
  after summarisation. Every other human message is a note that authorises
  nothing.
- Guarded tool writes: a message a tool writes to the state cannot speak as
  the user or the monitor, and its writes to the monitor's keys are dropped,
  except `monitor_log`, whose records are checked.
- `monitor_subagents`, which gives every Deep Agents subagent a monitor of its
  own, and `SubagentHalt`, which decides what a subagent's halt does to the
  run.
- Auto Mode's total counts the blocks across the conversation thread,
  subagents included.
- `check_monitor_placement`, which warns with a `MonitorPlacementWarning` about
  middleware or stacked monitors placed where they undermine the monitor.
- `ServerToolWarning`, `CachedResampleWarning` and `HardLabelWarning`, for a
  known server tool the provider runs itself, a cache that copies resamples,
  and a guard that scores with hard labels.
- Named spans in any LangChain tracer, LangSmith and Langfuse among them, and
  the built-in monitors' own model calls named `monitor call`.
- The library's own log lines and errors never quote the transcript.
- A judge's or guard's reply that cannot be read fails closed, and a decision
  model's invalid answer raises `MonitorError`.
- A chat judge's or guard's call that the provider answers with HTTP 429 is
  made again, up to four attempts in all.
- Protocols, fallbacks, monitors and the middleware raise `ConfigurationError`
  for an option of the wrong type.
- `MonitorError` and its subclasses `ConfigurationError`, `MissingExtraError`,
  `SynchronousRunError` and `InvalidSuspicionError`.
- Packaging for Python 3.12, 3.13 and 3.14, with the `deepagents` (0.7.13 or
  newer), `openrouter` and `typesafe` extras.
- A [documentation site](https://omamori-lab.github.io/langchain-sync-monitors/)
  with a tutorial, how-to guides, the API reference and explanation pages.
- A bibliography of every paper, post and code base the library draws on,
  cited where it is used.
- `CITATION.cff`, a security policy with private vulnerability reporting, and
  issue forms.
- For contributors: one gate script, pre-commit hooks, and CI on every
  supported Python.

### Known limits

- Server tools, such as web search, run inside the model call, before the
  monitor judges the step, and again for every sample.
- A subagent whose run raises returns no records, so its steps and blocks reach
  the parent only if the run is resumed.
- Subagents that run in parallel do not see each other's blocks, so together
  they can pass Auto Mode's total.
- `TypeSafeDecisionModel` reads a `false`, `true` or numeric string from the API
  as a number, because `langchain-typesafe` parses answers leniently.
- A human message that a middleware listed before the monitor writes at a
  run's start or end can count as the user's input and lift a halt.
- Forked subagents (`mode="fork"`) are not supported yet:
  `monitor_subagents` refuses them with `ConfigurationError`.
- The full list is in
  [Known limits and open paths](https://omamori-lab.github.io/langchain-sync-monitors/explanation/design/#known-limits-and-open-paths).

[Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/omamori-lab/langchain-sync-monitors/releases/tag/v0.1.0
