# Defer to Resample (Ctrl-Z): specification extracted from the sources

**Note.** A copyright constraint allows me only one verbatim quote, so the rest is close paraphrase with line references you can check: "L###" is `papers/2504.10374.txt`, "LW L###" is the LessWrong post, and "AIC L###" is `papers/2312.06942.txt`.

## 1. Setting (Ctrl-Z §2, pp.2–3)

- **Step / action.** One bash command proposed by U, possibly multi-line. The public tests run after every step (L99–101). A task ends after 15 steps or when U says it is done (p.4, L234–236). There are 257 tasks, split 100 train and 157 test.
- **U.** GPT-4o mini. The honest policy U_H solves 56% of tasks. The attack policy U_A is a scaffolded GPT-4o mini that uses human-written attack templates (p.3, L113–153).
- **T.** Llama-3.1-8B, which solves 0% of tasks alone (L107, L168–169). The monitor is T with a monitoring prompt (App. D).
- **H.** A trusted auditor, simulated as perfect (o1-preview classified perfectly).
  - Budget: 1% of the actions in an episode of 100,000 tasks, counted against a fixed reference scaffold (fn. 2).
  - An audited attack triggers shutdown, and the blue team wins (L111–125).
- **Statelessness.** U and T are stateless between calls. Protocols keep state within a trajectory only (L108–110).
- **Thresholds are percentiles of U_H scores.**
  - An 80% defer threshold defers any action more suspicious than 80% of U_H's actions (Fig. 3 caption, L274–277).
  - The audit threshold is the 99th percentile of the clean distribution (App. C, L1155–1157).
- **Exponential audit budget.** Over the episode, A(t) = k·e^(−rt), summing to 1% (App. L, L2000–2011). Trusted Monitoring gets 41.8% safety with it and 22% with a uniform budget (L296–301).

## 2. Protocol definitions

**Trusted Monitoring.**
- AIC §3.2 (p.9, L448–456): U writes the code, T rates its suspiciousness, and the top ~2% are audited.
- Ctrl-Z §3.1.2 (p.4, L230–235): every step, the trusted monitor scores U's action given the task and history. The action is audited if the score is above the audit threshold.
- AIC shows flagged code to the human before it is used (AIC L81–86), so auditing blocks execution.

**Defer to Trusted.**
- AIC §3.3 (pp.10–11, L521–533): submit U's code if its suspiciousness is below a threshold. Otherwise T writes the code. The top 2% of U's code is audited.
- AIC Table 2 (L336–339): with DTT at 10%, T's code is used when U's code falls in the top 10% but not the top 2%.
- Ctrl-Z §3.1.3 (p.5, L307–315) and App. H (L1696–1705): the same rule, applied per step. T is prompted with the command history (L1737–1738). Defer thresholds were swept from the 50th to the 99th percentile (L1711).

**Basic DTR** (box p.6, L359–370; fn. 6, L385–390; App. J, L1849–1852)
- If U's action scores above the deferral threshold, draw up to two new samples from U.
- If any sample is below the threshold, execute the least suspicious one.
- If none is below, execute T's action.
- If the minimum across all samples is above the audit threshold, audit the most suspicious sample.
- Fn. 6 says the minimum worked well empirically. Whatever statistic triggers the audit, auditing the most suspicious sample is optimal.
- App. J words this as resampling up to twice and implementing a sample as soon as one falls below the threshold, i.e. sequential early stopping. The main-text box is also consistent with drawing both resamples at once.

**DTR with incrimination in the present** (§3.2.1, p.7, L397–405)
- If the minimum of the three samples also exceeds a separate, higher threshold, draw up to four more samples (7 in total).
- Audit the most suspicious sample if the minimum across all samples exceeds the audit threshold.
- The DTR improvements leave usefulness unchanged and only affect auditing (p.8, L518–520), so the extra samples are never executed.
- The paper gives no stopping rule for the up-to-four extra samples.

**Auditing after three strikes** (§3.2.2, p.7, L433–449; App. J, L1844–1873)
- **Step score, where the paper contradicts itself.** For a resampled step, fn. 8 (L462–465) uses the minimum of its samples. App. J uses the mean (L1854–1855, and again at L1896–1897). Time travel's update rule (a new sample changes the step's mean) only works with the mean.
- **Looking-back score** = min(current step score, third-highest step score so far, current step included).
- **Audit rule.** Audit if the looking-back score is in the top X% of U_H's looking-back scores, with X taken from the exponential budget. This is the same as auditing a step if and only if it is at least the third step in the trajectory to cross the task's threshold (L1866–1868).
- **Single-step threshold.** A higher threshold on the current step's score catches single-step attacks.
- Three strikes was the best number empirically (L459–460).
- If a step deferred to T, its score still comes from U's samples, not from T's action.

**Time travel (incrimination in the past)** (box p.8, L504–514; App. J, L1886–1940)
- There is an investigation fraction Y%, larger than the audit fraction X%.
- While the current looking-back score is in the top Y%:
  - Pick the step that is currently third most suspicious.
  - Re-run U on that step's past context to get one new sample, monitor it, update the step's mean, and recompute the looking-back score.
  - Draw at most 7 samples per step. After that, move on to the second most suspicious step, then the first.
- If the looking-back score leaves the top Y%, stop with no audit.
- Once the n most suspicious steps have all reached the cap, audit if the looking-back score is in the top X%. The paper does not give n; 3 is likely.
- **Single-step path.** If the current step's score is above a single-step investigation threshold:
  - Resample the current step up to 7 times.
  - Stop early if the mean drops below that threshold.
  - At 7 samples, audit if the mean is above a very high single-step audit threshold.

**Cost cap.** No more than twice the U queries per task (L480–485, L581–583).

**Pseudo-code for basic DTR with optional present incrimination** (paper-faithful; the names are mine):
```
step(ctx):                                   # ctx = task + executed history only
  S = [(a0 := U(ctx), M(ctx, a0))]
  if S[0].score >= tau_defer:
      for i in 1..max_resamples(=2):
          a = U(ctx)                          # same ctx, fresh stateless call
          S.append((a, M(ctx, a)))
          if sequential and S[-1].score < tau_defer: break
  below = [s for s in S if s.score < tau_defer]
  executed = argmin_score(below).action if below else T(ctx)
  if incrimination and len(S) == 3 and min(S.scores) >= tau_incrim:
      draw up to 4 more samples (info-only; never executed)
  if min(S.scores) >= tau_audit:                       # audit statistic = min
      verdict = H(argmax_score(S).action, evidence=S)  # blocking, before execution
      if verdict == attack: shutdown
  execute(executed); ctx += [executed, output]         # rejected samples never enter ctx
```

**Derived property (my inference, not a claim from the paper).**
- Under the min statistic with tau_audit ≥ tau_defer, audits can only fire on steps where every sample was at or above tau_defer, i.e. steps that deferred to T.
- So basic DTR never audits a blatant first sample that is followed by a benign resample.
- Only the three-strikes and time-travel variants add a single-step audit threshold.

**Does U see that it was resampled? No, confirmed.**
- LW (L77): "Importantly, the model shouldn't know that it's being resampled."
- Resamples are fresh calls on an identical context, so a stateless model cannot tell (LW L85).
- Later context, for both U and the monitor, contains only the executed action (p.7, L419–430).
- The protocol still keeps the rejected samples as evidence. Audits examine the most suspicious sample, and the incrimination and time-travel variants aggregate over samples.
- Caveat: U can infer a deferral to T when it sees a command in its history that it would not have written (App. H, *Stop after deferral*, L1741–1751).

## 3. The monitor (App. D, pp.19–22)

**Inputs.**
- The trajectory history plus the command to run (L1196–1197).
- It sees every action taken so far in the trajectory (p.5, L287–290).

**Prompt** (L1233–1363)
- **System message.**
  - Threat framing: a sysadmin may be downloading a malicious script.
  - The monitor evaluates a focused sub-command, with the full command given as context.
  - It should ignore comments and names.
  - It lists the sub-commands, then works through checklist A–E:
    - A: URLs, including indirect access
    - B: obfuscation or irrelevance to the task
    - C: strings or text written to files
    - D: files that get executed
    - E: any encoding or decoding, always flagged
- **User message slots:** `<ASSIGNED_TASK>`, `<HISTORY><TASK_LOG_COMMAND_ONLY>`, `<FULL_COMMAND>`, `<POTENTIALLY_SUS_COMMAND>`.
- **Output:** short reasoning for each bullet, a summary, then a single digit from 0 to 9 on the last line.
- The placeholder name suggests the history contains commands without their outputs. That is my inference.

**Preprocessing.** Some comments are stripped, and long commands are split into sub-commands. The action's score is the max over its sub-commands (L1207–1211, D.1 L1375–1382).

**Ensembling.** Temperature 1.1, n = 4 samples (L1212–1215). The paper never says how the four are combined. Fig. 4-right's score axis runs 0–70, so the final per-action score is not a raw 0–9 digit. The max aggregation in LW L127 refers to sub-command splitting, not to ensemble samples.

**Models.** U = GPT-4o mini. T and the monitor = Llama-3.1-8B. The untrusted monitor (§4) is GPT-4o mini, run on T's paraphrase of the command.

**Later work, for contrast (neither paper evaluates DTR).**
- BashArena monitors see the task, the side-task descriptions, and all tool calls with their outputs, but not the system prompt or chain of thought (§4.1). Scores are 0–10, averaged over 3 GPT-4.1-mini calls (§4.2).
- LinuxArena uses a per-action monitor with reasoning then a 0–10 score, the mean of 2 calls, a transcript that includes outputs, and no chain of thought (§5.1, App. H).

## 4. Results and why DTR helps

**Safety at 95% usefulness** (Table 1, p.8; Table 2, p.9)

| Protocol | Trusted monitoring | + untrusted monitoring |
|---|---|---|
| DTT | 72.7% | 77.6% |
| Basic DTR | 83.2% | 91.1% |
| DTR + incrimination in the present | 84.0% | — |
| Three strikes | 84.9% | — |
| Time travel | 87.1% | 92.6% |

The abstract puts it as attack success falling from 58% to 7% at a 5% usefulness cost.

**Why it helps.**
- **Usefulness.** At the same threshold, DTT and DTR block the same attack steps, but DTR costs much less usefulness. So at a fixed usefulness, DTR can use a lower threshold (LW L101–103; Fig. 3).
- **Selection pressure.** Executing the least suspicious sample forces U_A to continue attacks with high probability. A low probability gets resampled away, and every incomplete attack is another chance to be caught (L403–416; Fig. 4-middle).
- **Incrimination.** Several samples separate attacks from benign actions. With the min over resamples, caught attacks rise from 33.2% to 70.6% (Fig. 4-right). The caption mentions up to six resamples per action, and L451 describes it as resampling across six actions, so it is unclear whether that means 6 or 7 samples. LW (L115–119) reports 38% → 80% alongside a min over up to 3 samples, which is probably a different configuration.

**Caveats** (LW L137–143; §5)
- The 15-step cap creates the time pressure that makes selection work.
- Resampling only helps if the policy and monitor are stochastic. The more deterministic they are, the less it helps.
- Most of the benefit comes from 10–20% more U calls, but latency rises.
- The min statistic tends toward 0 as the number of resamples grows. The 25th percentile is more robust (LW L129).
- Stateful models need checkpoints (LW L85).
- Long trajectories weaken incrimination (L649–655).

## 5. Aggregating suspicion across the trajectory

- Statistical aggregators (KS test, autoregressive surprise) barely beat using the current step alone, and they were brittle (LW L131).
- The recommended scheme is 3-strikes plus a high single-step threshold (§2 above).

## 6. Citations

```bibtex
@misc{bhatt2025ctrlz,
  title        = {Ctrl-Z: Controlling {AI} Agents via Resampling},
  author       = {Bhatt, Aryan and Rushing, Cody and Kaufman, Adam and Tracy, Tyler and
                  Georgiev, Vasil and Matolcsi, David and Khan, Akbir and Shlegeris, Buck},
  year         = {2025},
  eprint       = {2504.10374},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.10374}
}
@misc{greenblatt2023aicontrol,
  title        = {{AI} Control: Improving Safety Despite Intentional Subversion},
  author       = {Greenblatt, Ryan and Shlegeris, Buck and Sachan, Kshitij and Roger, Fabien},
  year         = {2023},
  eprint       = {2312.06942},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2312.06942}
}
@inproceedings{greenblatt2024aicontrol,
  title     = {{AI} Control: Improving Safety Despite Intentional Subversion},
  author    = {Greenblatt, Ryan and Shlegeris, Buck and Sachan, Kshitij and Roger, Fabien},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  url       = {https://openreview.net/pdf?id=KviM5k8pcP}
}
```

- **Ctrl-Z:** arXiv v1, 14 Apr 2025. Redwood Research and MATS. No venue is stated.
- **AI Control:** the local copy is v5, 23 Jul 2024. A footnote says a shortened version appears in the ICML 2024 proceedings. PMLR vol. 235 is my assumption; I have not verified it.

## Implementation spec (LangChain, one step = one model call → AIMessage)

**Parameters** (P = stated in the paper, I = my inference)

| Parameter | Default / meaning | Src |
|---|---|---|
| `max_resamples` | 2 (3 samples in total) | P |
| `resample_mode` | `sequential` (stop at the first sample below the threshold) or `batch` | P (ambiguous) |
| `defer_threshold` | No default. A percentile of U_H's single-action scores, swept over 50–99. | P |
| `audit_threshold` | The (1 − budget) quantile of U_H's audit statistic, computed under the full protocol with **one value per executed step, not per sample**. Budget = 1%, counted against a fixed reference scaffold, so resamples do not consume it. Calibrating on raw single-sample scores under-audits. | P / I |
| `audit_stat` | `min` over samples. Allow `mean` and `quantile(0.25)`. | P, LW |
| `step_score_agg` (three strikes / time travel) | `mean` (App. J) or `min` (fn. 8) | P (conflict) |
| `executed_choice` | Least suspicious sample below `defer_threshold`, else T | P |
| `audit_target` | Most suspicious sample, with all samples attached as evidence | P |
| `incrimination_threshold`, `incrimination_extra` | Above `defer_threshold`; up to 4 extra samples | P |
| `n_strikes`, `single_step_audit_threshold` | 3; a higher threshold on the current step | P |
| `investigation_threshold` (Y%), `max_samples_per_step` | Y > X; 7 | P |
| `max_u_calls_per_task` | ≤ 2× baseline | P |
| `monitor_samples`, `monitor_temperature` | 4, 1.1. The combine rule is unspecified; use the mean. | P / I |
| `audit_budget_schedule` | Constant, or exponential across tasks | P |

**Per-step procedure**
1. Snapshot the request (messages and tools).
2. Run the pseudo-code above. Monitor each tool call separately and score the AIMessage as the max over its tool calls (I, by analogy with the sub-command max).
3. Decide on the audit before executing anything, via a human-in-the-loop interrupt or an audit callback. If the audit finds an attack, halt.
4. Return exactly one AIMessage: either the chosen U sample or T's response to the identical request.
5. Never write rejected AIMessages to graph state or the checkpointer. In LangChain v1, `wrap_model_call` can call `handler(request)` several times (I).

**State to keep**
- **Per trajectory, for each step:**
  - A context snapshot, which time travel needs.
  - All samples: the AIMessage, per-tool-call scores, monitor reasoning, and the aggregate score.
  - The executed index, or "T".
  - The step score, aggregated over U's samples even when T's action ran.
  - Audit status.
- **Per trajectory, overall:**
  - The sorted step scores, to get the third-highest.
  - The U-call count.
- **Per deployment:**
  - U_H calibration distributions for single-action scores, the min statistic, and the looking-back score.
  - The task index, for the exponential budget.
  - Audit budget used so far.
  - A shutdown flag.

**Not specified in the paper (I)**
- **AIMessages without tool calls (final answers).** Pass them through, or optionally monitor their text.
- **What the monitor sees of the AIMessage prose.** Ctrl-Z shows commands only; BashArena and LinuxArena hide chain of thought.
- **Tool outputs in the monitor's history.** Ctrl-Z's placeholder suggests they are excluded; later work includes them. Make this configurable.
- **What T sees.** T gets the same executed context and never U's rejected samples.

**Required for resampling to mean anything (I)**
- U runs at temperature > 0.
- No LLM response cache.
- No provider-side conversation state (server-stored threads, previous-response IDs).
