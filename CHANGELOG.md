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

- `MonitorMiddleware`, which judges every step a LangChain agent or Deep Agent
  proposes before any of its own tools run, under `invoke()` and `ainvoke()`.
- Control protocols `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample`
  and `AutoMode`, with the `DeferToTrustedModel` and `HaltRun` fallbacks.
- `AutoMode` blocks a suspicious step and tells the agent why. After three
  blocks in one step, or twenty in the thread, its fallback halts the run by
  default.
- Default thresholds, 0.6 to defer, block or flag and 0.9 to audit, which raise
  `UncalibratedThresholdWarning` until you set your own.
- Monitors for three model families: `LLMMonitor` for chat judges,
  `GuardModelMonitor` for guard models and `DecisionModelMonitor` for decision
  models.
- The `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
- `MonitorView` and `Channel`, which choose what the monitor reads; the default
  follows Claude Code's auto mode classifier.
- `monitor_log`, which keeps one `StepRecord` per committed step, with every
  sample judged.
- `MonitorStepEvent` and `MonitorStepFailedEvent` on `stream_mode="custom"`.
  Only committed steps reach `stream_mode="messages"`.
- A halted step ends the run, and the halt stands until a new run brings new
  input.
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
- `ProviderToolWarning`, `CachedResampleWarning` and `HardLabelWarning`, for a
  known tool the provider runs itself, a cache that copies resamples, and a
  guard that scores with hard labels.
- Named spans in any LangChain tracer, LangSmith and Langfuse among them, and
  the built-in monitors' own model calls named `monitor call`.
- A judge's or guard's reply that cannot be read fails closed, and a decision
  model's invalid answer raises `MonitorError`.
- Protocols, fallbacks, monitors and the middleware raise `ConfigurationError`
  for an option of the wrong type.
- `MonitorError` and its subclasses `ConfigurationError`, `MissingExtraError`,
  `SynchronousRunError` and `InvalidSuspicionError`.
- Packaging for Python 3.12, 3.13 and 3.14, with the `deepagents`,
  `openrouter` and `typesafe` extras.
- A [documentation site](https://omamori-lab.github.io/langchain-sync-monitors/)
  with a tutorial, how-to guides, the API reference and explanation pages.
- A bibliography of every paper, post and code base the library draws on,
  cited where it is used.
- `CITATION.cff`, a security policy with private vulnerability reporting, and
  issue forms.
- For contributors: one gate script, pre-commit hooks, and CI on every
  supported Python.

### Known limits

- A provider's built-in tools, such as web search, run inside the model call,
  before the monitor judges the step, and again for every sample.
- A subagent whose run raises returns no records, so its steps and blocks reach
  the parent only if the run is resumed.
- Subagents that run in parallel do not see each other's blocks, so together
  they can pass Auto Mode's total.
- `TypeSafeDecisionModel` reads a `false`, `true` or numeric string from the API
  as a number, because `langchain-typesafe` parses answers leniently.
- A human message that a middleware listed before the monitor writes at a
  run's start or end can count as the user's input and lift a halt.
- The full list is in
  [Known limits and open paths](https://omamori-lab.github.io/langchain-sync-monitors/explanation/design/#known-limits-and-open-paths).

[Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/omamori-lab/langchain-sync-monitors/releases/tag/v0.1.0
