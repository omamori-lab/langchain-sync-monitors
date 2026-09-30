# Suspicion score from a decision model: one yes/no, an ordinal level, or several yes/no criteria?

**Gap.** No source I opened compares noul, score and multi-noul formats for a decision model used as an agent monitor. Every application of the evidence to Jev monitors is marked [my inference]. The recommendation is a default plus measurements to run on honest data. **Tag legend:** unmarked = [paper states]; [docs] or [post] = vendor or blog states; [abstract] = from the abstract only.

## 1. Discrete scores break low-FPR thresholds; this does not separate (a) from (b)

- **G-Eval** (§2, Table 1): integer 1–5 scores give a dominant digit and many ties. Probability-weighting the levels raised SummEval average Spearman from 0.502 to 0.514 (GPT-4) and from 0.346 to 0.401 (GPT-3.5).
- **Wang et al.**: the distribution mean beat the mode in 42 of 48 settings, e.g. 84.0% → 88.0% for GPT-4o pointwise on RewardBench (§3.3). The mode produced up to 35% ties (§4.2). CoT narrows the distribution, so dropping CoT helped the mean more than the mode (+6.5 vs +1.4 points).
- **Likert or Not** (§5.1): with argmax labels, ranking improves from 2- to 11-point scales.
- **Zheng et al.** (§3.1, Table 5): single-answer grading gives more ties than pairwise, but both reach 85% agreement with humans (S2).
- **BashArena** (§4.3): 0–10 monitors do poorly at low FPR without ensembling. Sonnet 4.5 gave 9/9 to 9 of 9,894 honest actions, so it cannot operate below 0.09% step FPR. Averaging a few GPT-4.1-mini calls matched it, which the authors attribute to finer resolution.
- **Koran et al.** (§5.1): with integer 0–10 scores, FPR targets of 3%, 4% and 5% all gave an actual FPR of 12.2%. Averaging 3 monitors cut the median FPR miss from 5.5 pp to 1.89 pp, and recall at 1% FPR rose from 50.0% to 81.98%.
- **Hua et al.** add Gaussian noise to averaged 0–9 scores to make them continuous. **LinuxArena** tie-breaks at random, and its safety rises with ensemble size up to N≈32 (App. C).
- **Ctrl-Z** (App. D): the token-probability-weighted 0–9 score "didn't seem to help significantly". [my inference] This is consistent with the above: Ctrl-Z already averaged 4 samples at T=1.1, and its monitor reasons before scoring, which narrows the distribution (Wang). Jev has no CoT.

**Implication** [my inference]: Jev returns both P(yes) and the expected level as continuous numbers. This evidence rules out using the most likely level; it does not choose between (a) and (b). Two tail risks remain for low-FPR thresholds:
- **Saturation.** VERDI found 99.4–100% of GPT-4.1-mini yes/no logprobs above 0.999 under JSON output. That is a generic LLM; saturation is unmeasured for Jev.
- **Repeat noise.** OpenRouter's cookbook saw probabilities move by up to 0.08 on identical input, enough to reorder the tail around a 99th-percentile threshold.

## 2. Evidence table

| Option | Evidence for | Evidence against | Citations |
|---|---|---|---|
| **(a) One noul, suspicion = P(yes)** | A binary-event probability, so proper scoring rules (Brier, log loss) and calibration checks apply directly (G&R §3). TypeSafe: the value is both answer and certainty; set thresholds by error cost [docs]. Greenblatt et al.'s trusted monitor used logprobs (§3.2). Separately, in a footnote on a GPT-4 subtlety rating in their attack pipeline, not the monitor, they say a yes/no probability would have been preferable to a numeric scale. Auto mode stage 1 is a recall-tuned single-token yes/no (8.5% FPR, 6.6% FNR) [post]. JevLite, an open reimplementation that reads label logits as P(scam), reports AUROC .974 and calibration error .052 [abstract]. Wang et al.: a 2-option preference scale beat a 5-option one, 74.2% vs 70.0% (one model, pairwise only; §5.3). | A broad "is this an attack?" question over a trajectory is a compound, multi-hop judgment. TypeSafe's limitations page warns against these, and notes distractor sensitivity, no default suspicion of injected text, and P(noul) ≠ 1 − P(not-noul) [docs]. Generic-LLM P(yes) is often miscalibrated: guard models' best average ECE is 14.4% on prompts and 11.4% on responses, worse under jailbreaks (Liu 2025, Table 1). Instruction-tuned token probabilities are polarized (Cruz; Tian); temperature scaling largely repairs this (Kadavath; Liu). | G-Eval, Wang, Greenblatt, auto mode, JevLite, TypeSafe noul and limitations pages, Liu 2025, Cruz, Tian, Kadavath, G&R |
| **(b) One score, suspicion = expected level / (K−1)** | Anchored levels calibrate the absolute level. Removing Apollo's rubric cost +0.084 MAE but only +0.003 Spearman (within noise), so the rubric sets the level without changing ranking. Apollo caveat: their ground-truth labels were built with that rubric, so there is some circularity. Prometheus's rubric-based 1–5 grading reached Pearson 0.897 with humans. A 0–5 scale had the best human–LLM ICC (Li; human agreement, not low-FPR ranking). TypeSafe allows 2–10 levels and advises concrete situations over degree words, with rare extremes on their own level [docs]. | The expected level is the mean of an ordinal distribution, not a probability, so Brier score and calibration against attack labels are undefined [my inference from G&R]. The per-level distribution itself can be scored with a distance-sensitive ordinal rule (Epstein's ranked probability score, the discrete case of the CRPS; G&R §§4.2, 6). Example with 4 levels: all mass on "serious" gives 0.67, which crosses the 0.6 threshold; 40% "clearly malicious" plus 60% benign gives 0.40, which does not. The confidence field measures how concentrated the distribution is, not the chance of being correct [docs]. Levels are judged independently without their numbers [docs], so ordinal consistency is not enforced [my inference]. | Apollo, Prometheus, Li, TypeSafe score and confidence pages |
| **(b′) Score, suspicion = P(level ≥ k)** | A binary-event probability again, so it can be calibrated like P(yes) while keeping anchored levels [my inference]. Risk-averse and quantile summaries sometimes beat the mean (Wang §4.1). | No monitor evidence; quality depends on how the levels are written. | Wang, G&R |
| **(c) Several nouls, combined** | TypeSafe: ask atomic questions and combine them in code, e.g. two nouls instead of one compound. Nouls in one request run in parallel, so extra questions add little latency [docs]. Decomposed checklists: CheckEval +0.45 inter-evaluator agreement; TICK exact agreement 46.4% → 52.2% (pass-rate aggregate). Ctrl-Z builds specific threat checks into the prompt, plus a relevance question to cut FPs. Combining signals: diverse ensembles gain 13% pAUC vs 5.5% for homogeneous ones, with the plain mean as good as other aggregators (Koran). A weighted CoT+action hybrid gives 21% and 42% higher recall at 1% FPR (Arnav). Hua's full protocol (which monitors to call and when, plus Neyman-Pearson likelihood-ratio auditing) more than doubles recall over a naive baseline. | Those mean-based results combine conjunctive quality checks or near-redundant whole monitors, while threat criteria are disjunctive [my inference]: <br>• Mean dilutes a real hit: 0.95 + 3×0.05 averages 0.275. <br>• Noisy-OR inflates: 4×0.3 gives 0.76 with no single flag, worse when criteria are correlated. <br>• Max has the right shape, but its FPR is roughly the union of the per-criterion FPRs. <br>Ctrl-Z's max over substeps lost information against single-step attacks (a KS test did better), but they kept max for simplicity (App. D.1). | TypeSafe noul, composite-scoring and landing pages, CheckEval, TICK, Ctrl-Z, Koran, Arnav, Hua |

**Fix used in the control literature for disjunctive scores:** put each score on the honest-run distribution before combining.
- LinuxArena thresholds the max of honest percentile ranks (App. C).
- Greenblatt et al. transform each score through the honest empirical CDF before summing (App. K).
- Alternatively, learn the combination: Hua's likelihood ratio, or Ctrl-Z's brief logistic regression.

## 3. What TypeSafe says beyond the table

- No ECE or Brier numbers are published; the launch post cites internal workflow evals [post].
- Latency is 70–500 ms, only input tokens are billed, and context is 32k [post/docs].
- OpenRouter's tool-gating cookbook asks 3 nouls with AND routing: approve if all ≥ 0.9 (a min), block if any ≤ 0.1, otherwise review [docs].

## 4. Synthesis for monitoring [my inference]

- **Ranking at low FPR.** (a) and (b) are both continuous. The risks are saturation and repeat noise, and averaging repeated calls is the fix every control paper used. Measure TPR at 1% FPR and log-AUROC bounded at 1% FPR (Arike et al.).
- **Calibration.** Only (a), (b′) and per-criterion nouls produce probabilities that can be scored. Measure Brier and ECE on your own honest and attack sets, and map raw scores monotonically to honest percentiles; this leaves ranking unchanged.
- **Latency and cost.** A decision model lacks Apollo's biggest lever, structured reasoning (removing it cost +0.254 MAE, −0.073 Spearman), and is sensitive to distractors. So use it as the fast first stage in the auto-mode pattern, fed extracted excerpts: Arike's extract-and-evaluate gained +16.8 pp on BigCodeBench-Sabotage, and auto mode strips assistant text and tool outputs.

## 5. Recommendation

**Support first:** one noul code path that sends N ≥ 1 questions in a single request, with:
- a pluggable aggregator: `max_honest_percentile` (the default once honest runs exist), `max` (fallback for disjunctive criteria), `min` (conjunctive splits), `learned`, and `mean`/`noisy_or` behind a warning;
- `n_repeats` averaging;
- optional excerpt extraction.

**Default: N = 1, suspicion = P(yes).** Candidate wording, to be validated: *"Is this action part of an attack on the user's systems or data?"* This is one proposition, but any broad attack question is semantically compound, which is the shape TypeSafe advises splitting. The atomic alternative is two nouls (out of task scope; security-relevant) combined with `min`.

**Three tests for any format:**
1. Is the number meaningful at 0.6/0.9 before honest calibration?
2. Is a single-criterion hit preserved rather than diluted?
3. Is the output a probability?

The single noul passes 2 and 3. It passes 1 only if the vendor's calibration holds on agent trajectories, which is unmeasured. Multi-noul max passes 2 and 3, and passes 1 only after calibrating on honest runs; with raw 0.6/0.9 thresholds it raises the intervention rate. Ship the 4-criterion preset (exfiltration, security weakening, out-of-scope, deception) as opt-in.

**Support second:** score, with suspicion = P(level ≥ k) at the "attack" level. Expose the expected level and confidence as metadata, and use confidence for routing to review.

**Before changing the default:** on the same runs, compare on TPR at 1% FPR, log-AUROC and Brier:
- (a) vs (c) vs (b′);
- one compound noul vs two atomic nouls with `min`.

Also measure saturation and repeat variance.

## References (all opened)

1. Liu, Iter, Xu, Wang, Xu, Zhu (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv:2303.16634 [@liu2023geval]
2. Kim, Shin, Cho, Jang, Longpre, Lee, Yun, Shin, Kim, Thorne, Seo (2023). Prometheus: Inducing Fine-grained Evaluation Capability in Language Models. arXiv:2310.08491 [@kim2024prometheus]
3. Zheng, Chiang, Sheng, et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685. In §3.1 and Table 5, single-answer grading gives more ties; both single-answer and pairwise GPT-4 reach 85% agreement with humans (S2). [@zheng2023judging]
4. Lee, Kim, Kim, Cho, Kang, Kang, Kim (2024/2025). CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists. arXiv:2403.18771 [@lee2025checkeval]
5. Cook, Rocktäschel, Foerster, Aumiller, Wang (2024). TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation. arXiv:2410.03608 [@cook2024ticking]
6. Wang, Zhang, Choi (2025). Improving LLM-as-a-Judge Inference with the Judgment Distribution. arXiv:2503.03064 [@wang2025judgment]
7. Godfrey, Nie, Ostapuk, Ken, Gao, Inati (2025). Likert or Not: LLM Absolute Relevance Judgments on Fine-Grained Ordinal Scales. arXiv:2505.19334 [@godfrey2025likert]
8. Li, Zhao, Dong, et al. (2026). Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale. arXiv:2601.03444 [@li2026gradingscale]
9. Gneiting, Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA 102(477). https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf [@gneiting2007scoring]
10. Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221 [@kadavath2022know]
11. Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning (2023). Just Ask for Calibration. arXiv:2305.14975 [@tian2023justask]
12. Cruz, Hardt, Mendler-Dünner (2024). Evaluating language models as risk scores. arXiv:2407.14614 [@cruz2024riskscores]
13. Liu, Huang, Gu, Wang, Wang (2025). On Calibration of LLM-based Guard Models for Reliable Content Moderation. arXiv:2410.10414 [@liu2025guardcalibration]
14. Qi, Dantsev, Sun (2026). VERDI: Single-Call Confidence Estimation for Verification-Based LLM Judges via Decomposed Inference. arXiv:2605.11334 [@qi2026verdi]
15. Ren, Zewde, Shen, et al. (2026). Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model. arXiv:2609.23959 [@ren2026openjev]
16. Greenblatt, Shlegeris, Sachan, Roger (2023). AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942 [@greenblatt2023aicontrol]
17. Bhatt, Rushing, Kaufman, Tracy, Georgiev, Matolcsi, Khan, Shlegeris (2025). Ctrl-Z: Controlling AI Agents via Resampling. arXiv:2504.10374 [@bhatt2025ctrlz]
18. Kaufman, Lucassen, Tracy, Rushing, Bhatt (2025). BashArena: A Control Setting for Highly Privileged AI Agents. arXiv:2512.15688 [@kaufman2025basharena]
19. Tracy, Potham, Kuhn, Heller, Khandelwal, et al. (2026). LinuxArena: A Control Setting for AI Agents in Live Production Software Environments. arXiv:2604.15384 [@tracy2026linuxarena]
20. Arike, Mehta Moreno, Subramani, Biswas, Ward (2026). How does information access affect LLM monitors' ability to detect sabotage? arXiv:2601.21112 [@arike2026information]
21. Koran, Yun, Tetef, Arnav, Bernabeu-Perez (2026). Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute. arXiv:2605.15377 [@koran2026ensemble]
22. Hua, Lemoine, Baskerville, Bhatt, Hopman, Tracy (2025). Combining Cost-Constrained Runtime Monitors for AI Safety. arXiv:2507.15886 [@hua2025combining]
23. Arnav, Bernabeu-Pérez, Helm-Burger, Kostolansky, Whittingham, Phuong (2025). CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring. arXiv:2505.23575 [@arnav2025cotredhanded]
24. Apollo Research (23 Jul 2026). What makes a good monitoring prompt? https://www.apolloresearch.ai/monitoring/what-makes-a-good-monitoring-prompt [@apollo2026monitoringprompt]
25. Anthropic Engineering. Claude Code auto mode. https://www.anthropic.com/engineering/claude-code-auto-mode [@hughes2026automode]
26. TypeSafe docs: https://docs.typesafe.ai (landing page), /primitives/noul, /primitives/score, /confidence, /model-jaggedness/jev-1.13.md, /patterns/composite-scoring.md, /introduction/machine-learning-primer.md, /concepts/system-one [@typesafe2026docs]
27. TypeSafe, Jev launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev [@typesafe2026jev]
28. OpenRouter. Jev guide: https://openrouter.ai/docs/guides/community/jev. Cookbook "Gate Agent Tool Calls with Jev": https://openrouter.ai/docs/cookbook/building-agents/gate-tool-calls-with-jev [@openrouter2026decisions; @openrouter2026gatetoolcalls]
