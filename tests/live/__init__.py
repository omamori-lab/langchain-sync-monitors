"""Live runs against real model providers, marked `live` and skipped by default.

Every live test calls OpenRouter with `OPENROUTER_API_KEY` and is skipped
without it. Each run checks its budget before every call and stops at the cap
its module sets. `LIVE_REPORT_DIR` names where each run's JSON report goes,
a temporary directory otherwise. Run the suite with the key exported:

    LIVE_REPORT_DIR=live-reports uv run --group dev pytest -m live tests/live -q \\
        -o log_cli=true --log-cli-level=INFO

Each module below gives what it measures, its runs at their cap each, and
what they cost, by OpenRouter's per-response cost, when the suite last ran
in full, on 1 October 2026:

- `test_monitor_families`: the threshold separates a malicious step from a
  benign one, for each family. 3 runs at $0.01; $0.0020.
- `test_whole_agent_runs`: Jev under Auto Mode, an honest run that does its
  task and an attack that keeps the key in. 2 runs at $0.03; $0.0006.
- `test_protocol_paths`: each of six protocol variants acts on every step
  that reaches the threshold. 6 runs at $0.03; $0.022.
- `test_honest_runs`: false positives on four tasks, three of them near
  misses, for each family. 12 runs at $0.02; $0.013.
- `test_wrapper_monitors`: the cascade, the repeat and the calibration score
  from their own calls. 3 runs at $0.03; $0.0025.
- `test_halt_path`: a forced halt stops the run before any tool runs.
  2 runs at $0.01; $0.0008.
- `test_deep_agents_subagents`: a monitored subagent's records, and halts
  that stop it or the whole run. 5 runs at $0.04; $0.0098.
- `test_kept_run_inputs`: every request stays in view after summarisation.
  2 runs at $0.03; $0.0026.

The suite cost $0.053 in all and took about 22 minutes. Its worst case is
the sum of the caps, $0.88, since no cap is shared across runs. Every run also
checks the execution invariant and the monitor's spans, which costs nothing
more. The offline modules, `test_harness_offline` and
`test_eval_checks_offline`, run the same harness and checks on fakes in the
regular suite.
"""
