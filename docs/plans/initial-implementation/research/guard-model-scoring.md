# Guard label → continuous suspicion score: evidence and recommendation

**Bottom line.** When a guard model exposes a probability, its own paper scores it with P(unsafe) taken from the label-token logits. Hard 0/1 labels give a monitor only one operating point, so it cannot be set to an arbitrary audit budget. However, none of the three OpenRouter guard endpoints expose logprobs (checked 2026-09-28). The practical default is therefore k-sample voting plus honest-run percentile thresholds with randomized tie-breaking, and logprobs are used where a backend offers them.

## 1. What the guard papers use as their continuous score

| Model | Score used for AUPRC/AUC | Source |
|---|---|---|
| Llama Guard 1 | Probability of the first output token. "the classifier score can be read off from the probability of the first token" (§3.1). AUPRC (§4.3.3). | Inan et al. 2023 [paper states] |
| Llama Guard 3 | Model card: P(first token = unsafe), then thresholded (paraphrased). The LG4 card describes no score and reports recall, FPR and F1. | HF model cards [card states] |
| ShieldGemma | Eq. 1 is a softmax over LL(Yes) and LL(No) with temperature T and smoothing α. Table 1 uses α=0, T=1 and reports optimal F1 and AU-PRC. Per-harm probabilities are aggregated by **max**. | Zeng et al. 2024 [paper states] |
| Granite Guardian | §4.3 "probability of risk": the probability mass of Yes-like and No-like tokens within the top-20 is summed, then softmaxed. §5.1 and App. D report AUC, AUPRC and **TPR at FPR 0.1, 0.01 and 0.001**. | Padhi et al. 2024 [paper states] |
| Aegis | AUPRC from logprobs. Baselines without logprobs are marked "–" (§5, Table 3). | Ghosh et al. 2024 [paper states] |
| WildGuard | Text labels only. §4.1 reports F1 for all evaluations. | Han et al. 2024 [paper states] |
| Qwen3Guard-Gen | Evaluated with F1 in two modes: "strict" (controversial counts as unsafe) and "loose" (it counts as safe) (§3.4). The Stream variant has softmax risk heads (§4.1). | Qwen Team 2025 [paper states] |
| gpt-oss-safeguard | Multi-policy accuracy and F1 (§2). The cookbook offers an optional *categorical* confidence field and no probabilities. | OpenAI 2025 [report states] |

## 2. Evidence for each approach

| Approach | Evidence for | Evidence against |
|---|---|---|
| **(a) Hard 0/1** | Cheapest option and works with any API. WildGuard, Qwen3Guard and gpt-oss-safeguard all report only thresholded metrics [paper states]. | A binary score gives exactly one operating point: FPR = r, the honest flag rate. For a budget b < r you must audit a random b/r share of flagged actions, so TPR scales by b/r. Budget above r cannot be spent usefully [my inference, applying Makins & Angelini App. G.4.1]. The default thresholds of 0.6 (intervene) and 0.9 (audit) then fire on the same actions [inference]. Even 11-level scales miss FPR targets (see §3) [paper states]. |
| **(b) P(unsafe) from logprobs** | This is what Llama Guard, ShieldGemma, Granite Guardian and Aegis use for AUPRC/AUC. Granite shows it supports operation down to FPR 1e-3 [paper states]. It needs one deterministic call and gives the finest resolution. | Guards are **overconfident**: most predictions from LG1/LG3/WildGuard fall at 90–100% confidence (Liu et al. §4.2). Under jailbreaks, WildGuard reaches 92.8% F1 but 34.9% ECE (§4.3). Liu et al. tested three fixes: temperature scaling (best for response classification), contextual calibration (best for prompt classification, no validation set needed) and batch calibration (weaker). No fix fully resolved miscalibration (§5) [paper states]. For binary outputs all three fixes are **monotone within one prompt template**, so they change absolute thresholds but not percentile thresholds [my inference]. Ctrl-Z App. D found that probability-weighting a 0–9 rubric score gave no significant gain; that was a rubric monitor, not a guard [paper states]. Truncated top_logprobs can hide the "unsafe" token [inference]. Not available on OpenRouter guards (§4). |
| **(c) k samples, fraction unsafe** | Works without logprobs. AI Control App. M.8 fn. 6 names sampling (128 completions at T=1) as the sensible alternative when logprobs are unavailable. In control practice, averaging monitor samples helps (§3). In QA-confidence work, consistency beats verbalized confidence at M=5, with gains converging by M≈13 (Xiong §5.3). Lyu et al. find consistency beats raw logits on Brier score for open models, with 3–5 samples already a notable gain and saturation at 15–20 (Table 2, Fig. 6). Tian et al. used n=10 and found n=20 no better (fn. 6) [paper states]. | For guards whose label is a single token, this is a Monte-Carlo estimate of (b) at temperature T. Its resolution is 1/k and its standard error is √(p(1−p)/k), so it cannot resolve P(unsafe) ≪ 1/k, and that honest tail is exactly what matters at 1% budgets. Because guards are overconfident, most honest actions will score 0/k, and the 99th-percentile threshold lands on a tie at zero [my inference]. Cost is k times higher. Provider `top_k`, `top_p` and `min_p` defaults can suppress rare "unsafe" draws [inference]. The QA-to-guard transfer is also inference. |
| **(d) Severity tiers** | Qwen3Guard builds "controversial" from disagreement between a strict-biased and a loose-biased model (§3.3), so it marks cases near the decision boundary. That makes an intermediate value such as 0.5 semantically defensible [inference from paper]. The Aegis data has a "Needs Caution" class [paper states]. | Only 3 levels, so ties remain. The paper proposes no numeric mapping. Qwen3Guard is not on OpenRouter. A better option is expected severity from probabilities, P(unsafe) + 0.5·P(controversial) [inference]. |
| **(e) Verbalized confidence** | For RLHF chat models, verbalized confidence is better calibrated than sampled label probability, often a 50% relative ECE reduction (Tian et al.). Adding verbalized confidence to consistency helps when M is small (Xiong §5.4) [paper states]. gpt-oss-safeguard supports a confidence field. | Verbalized values are overconfident, clustered in multiples of 5 and mostly 80–100% (Xiong §5.1), so they are coarse and tie often. Fine-tuned guards emit fixed formats, and Llama Guard outputs only a label and categories. ProbGuard (2608.10621) reports that verbalized and P(True) scores are unstable [paper states]. Transfer to guards is inference. |

**Recent work checked.** "ProbGuard" is the name of two real papers, and both address different problems. 2608.10621 trains a model to predict the unsafe probability of a *prefix's continuation*, using Monte-Carlo-sampled continuations as targets. 2508.00500 is a DTMC-based runtime monitor for embodied and driving agents. Neither gives a recipe for scoring an off-the-shelf guard's label.

## 3. What the control literature implies

- **Thresholds are honest-run percentiles, not absolute values** [paper states]:
  - AI Control §3.2 sets a 2% FPR on honest solutions, using GPT-3.5-Turbo-Instruct log probabilities.
  - Ctrl-Z App. C sets the audit threshold at the 99th percentile of the clean distribution for a 1% budget.
  - LinuxArena §5.1 uses the 99th percentile with probabilistic tie-breaking.
  - BashArena uses a 4% trajectory-level FPR, which is 0.08% per step.
- **Discrete scores are named as a problem** [paper states]:
  - BashArena §4.3: 0–10 ratings perform poorly at very low FPR without ensembling. Sonnet 4.5 gave 9/9 to 9 of 9,894 honest actions, so it cannot go below 0.09% FPR. Averaging GPT-4.1-mini calls helps because it adds resolution. The authors add that "in practice monitors will likely be trained to output calibrated continuous scores".
  - Koran et al. §5.1: with 0–10 scores, FPR targets of 3%, 4% and 5% all map to the same threshold (actual FPR 12.2%). Averaging 3 monitors cuts the median FPR gap from 5.5 pp to 1.89 pp.
  - Makins & Angelini App. G.4.1: integer scores may be unable to hit the target FPR, so they formalize randomizing scores within the tied bin.
- **Ensembling by averaging** [paper states]:
  - Ctrl-Z samples n=4 at T=1.1 and reports that the monitor catches the suspicious part only in some samples (App. D).
  - BashArena averages 3 calls.
  - LinuxArena averages 2 by default (App. H). Its ablation plateaus near N≈32 (App. C).
  - Koran et al. find the arithmetic mean is as good as fancier aggregators.
- **Combining heterogeneous monitors** [paper states]: AI Control App. K maps each score through the honest empirical CDF before combining. LinuxArena's combined rule takes the maximum of percentile ranks. A guard score should be rank-transformed the same way before it is mixed with rubric monitors [inference].
- **Resampling protocols** [inference]: Ctrl-Z's resample protocols aggregate resample suspicion scores, and with a 0/1 guard that signal is binary too.

## 4. OpenRouter availability (public `/endpoints` API, 2026-09-28)

| Model / endpoint | logprobs | top_logprobs | logit_bias | Price per 1M tokens (in / out) |
|---|---|---|---|---|
| meta-llama/llama-guard-4-12b — DeepInfra (bf16) | No | No | **Yes** | $0.18 / $0.18 |
| openai/gpt-oss-safeguard-20b — Groq | No | No | No | $0.075 / $0.30 (cached input $0.0375) |
| nvidia/nemotron-3.5-content-safety — DeepInfra (bf16) | No | No | **Yes** | $0.20 / $0.20 |
| nvidia/nemotron-3.5-content-safety:free — Nvidia | No | No | No | $0 / $0 |

All four endpoints accept `temperature` and `seed`. LG4 and paid Nemotron also accept `top_k`, `top_p` and `min_p`. ShieldGemma, Granite Guardian, Qwen3Guard and WildGuard are **not on OpenRouter**, so (b) and (d) require self-hosting (vLLM/HF). The only models on OpenRouter that expose logprobs are general-purpose Granite 4.x models, not Guardian.

## 5. Recommendation

- **Score strategy `auto`** (default):
  1. **`logprob`** when the backend returns logprobs (self-hosted, or providers that support them). Compute P(unsafe) renormalized over the label tokens, summing token variants within the top-k as Granite does. Reduce multiple categories with max, as ShieldGemma does. For tri-class outputs use expected severity. Store the log-odds so ranks survive saturation near 0 and 1.
  2. **`sample`** otherwise: k=5 at T≈1 (Ctrl-Z used 1.1), with `top_p=1`, `min_p=0` and `top_k` disabled. Score = fraction unsafe. For reasoning guards (gpt-oss-safeguard, Nemotron in custom-policy mode), each sample must be a full reasoning-plus-label draw. Optional adaptive mode: draw extra samples only when the first samples disagree or sit near a threshold, as BashArena suggests.
- **Option `logit_bias_bisect`**: for LG4 or paid Nemotron on OpenRouter. At T=0, binary-search the bias on the "unsafe" token until the label flips; the flip point is the logit gap. About 8–12 calls give fine resolution. This is the logprob-free technique of Carlini et al. 2024 (§6, App. F.1). Requirements:
  - the label must be the first generated token, or an assistant prefill must be used;
  - token IDs must come from the HF tokenizer;
  - the provider must be deterministic.
  
  Treat it as experimental [inference].
- **Option `hard`**: explicit opt-in only, and log a warning that defer and audit collapse onto one operating point.
- **Option `tiers`**: for Qwen3Guard only, with the mapping {safe: 0, controversial: 0.5, unsafe: 1}.
- **Not recommended by default: `verbalized`** for fine-tuned guards.
- **Thresholds.** Ship `calibrate(honest_scores)`, which maps raw scores through the honest ECDF so that 0.9 means "top 10%" and 0.99 means a 1% budget. Break the remaining ties by randomization (Makins & Angelini; LinuxArena). Document that absolute thresholds such as 0.6 and 0.9 on uncalibrated guard probabilities are not meaningful, given the overconfidence found by Liu et al. Temperature, contextual and batch calibration only matter when absolute thresholds are kept [inference].

## Verified references (all opened)

- Inan et al., *Llama Guard*, 2023, arXiv:2312.06674 [@inan2023llamaguard]
- Meta, Llama-Guard-3-8B and Llama-Guard-4-12B model cards, huggingface.co/meta-llama [@meta2024llamaguard3; @meta2025llamaguard4]
- ShieldGemma Team (Zeng et al.), *ShieldGemma*, 2024, arXiv:2407.21772 [@zeng2024shieldgemma]
- Padhi et al., *Granite Guardian*, 2024, arXiv:2412.07724 [@padhi2024graniteguardian]
- Ghosh et al., *AEGIS*, 2024, arXiv:2404.05993 [@ghosh2024aegis]
- Han et al., *WildGuard*, 2024, arXiv:2406.18495 [@han2024wildguard]
- Qwen Team, *Qwen3Guard Technical Report*, 2025, arXiv:2510.14276 [@qwen2025qwen3guard]
- OpenAI, *Technical Report: gpt-oss-safeguard-120b/20b*, Oct 2025, cdn.openai.com PDF; *gpt-oss-safeguard user guide*, developers.openai.com/cookbook [@openai2025gptosssafeguard; @openai2025gptosssafeguardguide]
- Liu, Huang, Gu, Wang, Wang, *On Calibration of LLM-based Guard Models for Reliable Content Moderation*, ICLR 2025, arXiv:2410.10414 [@liu2025guardcalibration]
- Huang et al., *ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions*, 2026, arXiv:2608.10621 [@huang2026probguard]
- Wang, Poskitt, Wei, Sun, *ProbGuard: Proactive Runtime Monitoring for LLM Agent Safety*, 2025/26, arXiv:2508.00500 [@wang2026probguard]
- Xiong et al., *Can LLMs Express Their Uncertainty?*, ICLR 2024, arXiv:2306.13063 [@xiong2024uncertainty]
- Tian et al., *Just Ask for Calibration*, EMNLP 2023, arXiv:2305.14975 [@tian2023justask]
- Kadavath et al., *Language Models (Mostly) Know What They Know*, 2022, arXiv:2207.05221 [@kadavath2022know]
- Lyu et al., *Calibrating Large Language Models with Sample Consistency*, AAAI 2025, arXiv:2402.13904 [@lyu2025sampleconsistency]
- Carlini et al., *Stealing Part of a Production Language Model*, 2024, arXiv:2403.06634 [@carlini2024stealing]
- Greenblatt, Shlegeris, Sachan, Roger, *AI Control*, 2023, arXiv:2312.06942 (local) [@greenblatt2023aicontrol]
- Bhatt et al., *Ctrl-Z*, 2025, arXiv:2504.10374 (local) [@bhatt2025ctrlz]
- *BashArena*, 2025, arXiv:2512.15688 (local) [@kaufman2025basharena]
- Tracy et al., *LinuxArena*, 2026, arXiv:2604.15384 (local) [@tracy2026linuxarena]
- Makins & Angelini, *Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors*, 2026, arXiv:2607.07368 (local) [@makins2026multiagent]
- Koran, Yun, Tetef et al., *Ensemble Monitoring for AI Control*, 2026, arXiv:2605.15377 [@koran2026ensemble]
