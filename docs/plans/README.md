# Plans

One directory per feature. Each holds the plan written before the feature was
built and the research behind its decisions:

```
docs/plans/<feature>/
  plan.html        the plan: diagrams, readable pseudocode, checks, decisions
  research/        the evidence the decisions rest on
```

A plan records what was decided and why at the time. The code, its docstrings
and the documentation under `docs/` describe what the library does now.
Start a new directory for each new feature rather than editing an old plan.

| Feature | Status |
|---|---|
| [initial-implementation](initial-implementation/) | Built: monitors, protocols, middleware, Deep Agents support |
| [monitor-scores](monitor-scores/) | Built for 0.1.2: each step's suspicion as LangSmith feedback and a Langfuse score (`export_scores`) |
