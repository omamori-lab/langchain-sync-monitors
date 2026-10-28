# How-to guides

This page lists the how-to guides, each of which solves one concrete task.

## Protocols

- [Choose a protocol](choose-a-protocol.md): which protocol fits, compared on
  what runs, cost, latency and what the agent sees.
- [Use Trusted Monitoring](use-trusted-monitoring.md): let every step run,
  and flag the suspicious ones for a person to review.
- [Use Defer to Trusted](use-defer-to-trusted.md): hand suspicious steps to a
  trusted model.
- [Use Defer to Resample](use-defer-to-resample.md): draw more samples and run
  the least suspicious one, as in Ctrl-Z.
- [Use Block and Feed Back](use-block-and-feed-back.md): block a step, tell
  the agent why, and audit its next step after too many blocks.
- [Answer an audit](answer-an-audit.md): approve, edit, reject or stop a step
  the protocol holds for a person.
- [Monitor Deep Agents subagents](monitor-deep-agents-subagents.md): give every
  subagent a monitor and choose what a subagent halt does.

## Monitors

- [Use an LLM monitor](use-an-llm-monitor.md): any chat model as the monitor, with
  a prompt you control.
- [Use a guard model](use-a-guard-model.md): a safety classifier that follows a
  policy you write.
- [Use a decision model](use-a-decision-model.md): statements about each step,
  each answered with the probability that it holds.
- [Combine and calibrate monitors](combine-and-calibrate-monitors.md): average,
  cascade and calibrate monitors against honest runs.
- [Choose what the monitor reads](choose-what-the-monitor-reads.md): the
  trajectory a monitor keeps, and the views, channels and tags it reads.
- [Read the monitor log](read-the-monitor-log.md): the records, the stream
  events and the logs a monitored run leaves.
- [See the monitor's decisions in LangSmith and Langfuse](see-decisions-in-langsmith-and-langfuse.md):
  the spans each step opens in a tracer, the filters that find halted and
  flagged steps, and each step's suspicion sent as a score.
