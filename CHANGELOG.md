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

- `OpenRouterDecisionModel` raises `ConfigurationError` when it is built with
  a `base_url` that is not an `http` or `https` URL with a host, such as an
  empty or relative one, with no part of the URL in the message; such a
  `base_url` failed only at the first request. A relative `base_url` that
  reached a server through an `http_client` with its own `base_url` now
  needs the whole URL.

### Security

- A password in `OpenRouterDecisionModel`'s `base_url` no longer reaches logs
  or errors. httpx quoted it in its request log line and in the error stamina
  logs on a retry, and sent it as Basic authentication in place of the key. A
  `base_url` httpx could not read failed at the first request with
  `httpx.InvalidURL`, which could quote the start of the password as the
  port. Both now raise `ConfigurationError` when the model is built, with no
  part of the URL in the message; pass the key as `api_key`. A `base_url`
  with a user name or password used to send requests, so a gateway that took
  their Basic authentication in place of the key now needs an `http_client`
  you pass with its own `auth`.

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

[Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/omamori-lab/langchain-sync-monitors/releases/tag/v0.1.0
