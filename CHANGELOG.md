# Changelog

Notable changes to langchain-sync-monitors. The format follows the spirit of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

The public API is the set of names exported from `langchain_sync_monitors`
(monitors, control protocols, the middleware and their constructor keywords)
and the shape of the `monitor_log` records. A release that could turn a working
integration into a failing one is a breaking change.

## [Unreleased]

### Added

- `MonitorMiddleware`, which puts a monitor and a control protocol around every
  model call of a LangChain agent or Deep Agent, logs each step to
  `monitor_log` and streams it as a `MonitorStepEvent`; runs under both
  `invoke()` and `ainvoke()`. `check_monitor_placement` warns when middleware
  inside the monitor could return commands.
- `monitor_subagents`, which gives every Deep Agents subagent, including the
  general-purpose one, a monitor of its own, with per-subagent overrides and a
  choice of what a subagent halt does to the run (`SubagentHalt`).
- Control protocols: `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample`
  (sequential or parallel resampling) and `AutoMode` (blocks a step, tells the
  agent why with a `FeedbackTemplate`, and halts after Claude Code's limits),
  with the `DeferToTrustedModel` and `HaltRun` fallbacks.
- Monitors for three model families: `LLMMonitor` (chat judges, with
  `DEFAULT_MONITOR_PROMPT`), `GuardModelMonitor` (guard models, scored from
  log-probabilities or sampling) and `DecisionModelMonitor` (decision models
  through `OpenRouterDecisionModel` or `TypeSafeDecisionModel`), plus the
  `RepeatedMonitor`, `CalibratedMonitor` and `CascadeMonitor` wrappers.
- Default thresholds (0.6 to defer, block or flag; 0.9 to audit) that raise
  `UncalibratedThresholdWarning` until you set your own.
- Project scaffold: packaging, the gate script, CI, pre-commit hooks, the
  documentation skeleton and the bibliography with its citation test.
- An explanation page on how the library is built, with diagrams of a
  monitored step, the protocols, auto mode and subagents.
