# Explanation

This page lists the explanations, which cover why the library is built the way
it is.

- [How the library is built](design.md) walks through one monitored step and
  each part of the design: where the middleware sits, what the monitor reads,
  the protocols, the monitor models, subagents, the sync and async paths,
  what streams and traces show, and the log records.
- [Where the ideas come from](background.md) credits every paper and codebase
  the library draws on.
- [Live runs of a monitored agent](live-runs.md) reports a small evaluation
  with real models: an agent under each monitor and three of the protocols,
  once honestly and once with a hidden side task, step by step.
