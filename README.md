# langchain-sync-monitors

Control monitors for LangChain agents (`create_agent`) and Deep Agents (`create_deep_agent`), built as middleware.

A monitor judges every step an agent proposes before anything runs. A control protocol decides what to do with that judgement: let the step run, draw another sample, hand the step to a trusted model, tell the agent to change course, or stop. "Sync" means synchronous in the control sense: a step waits for its verdict. The implementation is async-native and also runs under `invoke()`.

**Status:** design approved in draft 3; implementation not started. The work is tracked in this repository's issues.

## Plan

Open [`docs/plan.html`](docs/plan.html) in a browser. It has the architecture diagrams, readable Python pseudocode for every component, the checks to run before building, the LangChain findings, and the decisions taken so far. Supporting research is in [`docs/research/`](docs/research/).

## What gets built first

| Piece | Choices |
|---|---|
| What the monitor reads | `Channel` flags: user messages, reasoning, agent text, tool calls, tool results, subagent reports |
| How a step is scored | `LLMMonitor` (chat judges), `GuardModelMonitor` (guard models such as gpt-oss-safeguard), `DecisionModelMonitor` (decision models such as Jev); wrappers `RepeatedMonitor`, `CalibratedMonitor`, `CascadeMonitor` |
| What happens with the score | `TrustedMonitoring`, `DeferToTrusted`, `DeferToResample` (Ctrl-Z), `AutoMode` (after Claude Code auto mode) |
| Where it plugs in | `MonitorMiddleware` for any LangChain agent; `monitor_subagents()` for Deep Agents subagents |

Every model is a parameter you pass in. The library sets no default model.

## Credits

This library implements protocols and practices from the following work. Please cite the original authors.

- Bhatt, Rushing, Kaufman, Tracy, Georgiev, Matolcsi, Khan, Shlegeris. *Ctrl-Z: Controlling AI Agents via Resampling.* arXiv:2504.10374, 2025. Defer to Resample.
- Greenblatt, Shlegeris, Sachan, Roger. *AI Control: Improving Safety Despite Intentional Subversion.* arXiv:2312.06942, 2023 (ICML 2024). Trusted Monitoring and Defer to Trusted.
- Hughes. *How we built Claude Code auto mode.* Anthropic Engineering, 2026. The blocking and steering monitor with escalation limits.
- Apollo Research. *What makes a good monitoring prompt?* 2026. Monitor prompt design.
- Kaufman et al., *BashArena* (arXiv:2512.15688); Tracy et al., *LinuxArena* (arXiv:2604.15384); Koran et al., *Ensemble Monitoring for AI Control* (arXiv:2605.15377). Monitor averaging, percentile thresholds and tie-breaking.
- Liu et al. *On Calibration of LLM-based Guard Models for Reliable Content Moderation.* ICLR 2025, arXiv:2410.10414. Guard model scoring.

The full reference list is in the plan.

```bibtex
@misc{bhatt2025ctrlz,
  title         = {Ctrl-Z: Controlling {AI} Agents via Resampling},
  author        = {Bhatt, Aryan and Rushing, Cody and Kaufman, Adam and Tracy, Tyler and
                   Georgiev, Vasil and Matolcsi, David and Khan, Akbir and Shlegeris, Buck},
  year          = {2025},
  eprint        = {2504.10374},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2504.10374}
}

@misc{greenblatt2023aicontrol,
  title         = {{AI} Control: Improving Safety Despite Intentional Subversion},
  author        = {Greenblatt, Ryan and Shlegeris, Buck and Sachan, Kshitij and Roger, Fabien},
  year          = {2023},
  eprint        = {2312.06942},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2312.06942}
}

@misc{hughes2026automode,
  author       = {Hughes, John},
  title        = {How we built {Claude Code} auto mode: a safer way to skip permissions},
  howpublished = {Anthropic Engineering Blog},
  year         = {2026},
  url          = {https://www.anthropic.com/engineering/claude-code-auto-mode}
}
```
