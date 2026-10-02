# Monitor scores

Each monitored step's highest suspicion, written as a first-class number in
the tracing tools: feedback on the step's run in LangSmith, and a numeric
score on the step's observation in Langfuse. It is the `export_scores`
option of `MonitorMiddleware`, off by default. It was planned for 0.2 and
ships in 0.1.2.

- `plan.html`: open it in a browser. The plan the owner approved on
  1 October 2026: what each tool shows, where the writer runs, one step over
  time, what the switch does, and the pseudocode.
- `research/canonical-integration.md`: how LangSmith and Langfuse expect a
  guardrail's or an evaluator's result to be recorded, from their docs, SDK
  source and live services, with the options the owner decided between.
- `research/direct-endpoints.md`: the addendum. The owner rejected the
  Langfuse code-evaluator route, and this records the public endpoints the
  library uses instead, each fact checked against the live services, with
  corrections to the plan.
