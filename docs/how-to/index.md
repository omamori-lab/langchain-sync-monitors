# How-to guides

This page lists the how-to guides, each of which solves one concrete task.

## Protocols

- [Choose a protocol](choose-a-protocol.md): which protocol fits, compared on
  what runs, cost, latency and what the agent sees.
- [Use Trusted Monitoring](use-trusted-monitoring.md): flag suspicious steps
  for audit and let them run.
- [Use Defer to Trusted](use-defer-to-trusted.md): hand suspicious steps to a
  trusted model.
- [Use Defer to Resample](use-defer-to-resample.md): draw more samples and run
  the least suspicious one, as in Ctrl-Z.
- [Use Auto Mode](use-auto-mode.md): block a step, tell the agent why, and halt
  after repeated blocks.
- [Monitor Deep Agents subagents](monitor-deep-agents-subagents.md): give every
  subagent a monitor and choose what a subagent halt does.

## Monitors

- [Use a chat judge](use-a-chat-judge.md): any chat model as the monitor, with
  a prompt you control.
- [Use a guard model](use-a-guard-model.md): a safety classifier that follows a
  policy you write.
- [Use a decision model](use-a-decision-model.md): yes or no questions answered
  as probabilities.
- [Combine and calibrate monitors](combine-and-calibrate-monitors.md): average,
  cascade and calibrate monitors against honest runs.
- [Choose what the monitor reads](choose-what-the-monitor-reads.md): the views,
  channels and tags a monitor sees.
- [Read the monitor log](read-the-monitor-log.md): the records, the stream
  events and the logs a monitored run leaves.
