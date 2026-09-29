# Paraphrasing between agent and monitor: existing implementations

This note surveys the code that others have written for a trusted paraphrasing (or redaction) layer placed between an agent and its monitor. It records what each implementation rewrites, with which model and prompt, how the result reaches the monitor, how it fails, and what `langchain-sync-monitors` should reuse or avoid. Research date: 2026-09-29. Every repository was cloned or read at the commit given; nothing was modified, posted or filed.

On quotation: to respect copyright I quote verbatim only once in this note (under 15 words). Every other prompt is summarised closely, with a file and line reference so the wording can be checked at the pinned commit.

## 1. Summary of findings

1. **The Ctrl-Z paraphraser itself is not public.** The paper describes it in one paragraph [@bhatt2025ctrlz]: the trusted model (Llama-3.1-8B) rewrites each proposed bash command into a detailed natural-language description, dropping arbitrary implementation detail, before the untrusted monitor (GPT-4o mini) reads it; the original command still runs. No prompt appears in the appendix, and none of the Redwood public repositories checked contains the producer (section 4).
2. **The only public code from the original control papers is the AI Control "censoring" pipeline** [@greenblatt2023controlevaluationscode]: a three-stage LLM rewrite of the task description and a three-stage canonicalise-and-minify of the code, applied identically to real inputs and to the honeypots used to validate the untrusted monitor.
3. **The closest analogue to our feature is the LASR Labs / SPAR untrusted-monitoring code** [@gardnerchallis2026untrustedmonitoringcode]: a trusted model (GPT-4o mini) summarises the code in natural language; only the untrusted monitor sees the summary, the trusted monitor sees the original, and the original plus the paraphraser prompts are stored per sample. Its paper [@gardnerchallis2026safetycase] gives the clearest numbers on the cost and benefit (section 3.3).
4. **Paraphrasing the chain of thought before a monitor has public code** in Schulz's `encoded-reasoning` [@schulz2026encodedreasoningcode], supporting the finding that paraphrasing a monitor-jailbroken chain of thought lets the same monitor flag it [@schulz2026monitorjailbreaking].
5. **The largest library of paraphraser prompts** is the Text Steganography Benchmark of Roger and Greenblatt [@roger2023textstegcode; @roger2023preventing]. Its goal differs from ours: its "suspicious" variants delete suspicious content, which in a monitor view would hide the attack from the monitor.
6. **No mainstream framework implements monitor-input paraphrasing.** ControlArena [@controlarena2026] defines the term in its glossary only; LinuxArena's Control Tower [@controltower2026] lists it as missing in an open issue; BashArena's public code [@basharenapublic2026] consumes paraphrases but ships no paraphraser; Inspect AI [@inspectai2026] has none.
7. **Failure behaviour varies widely and is the most actionable difference** (table in section 5): fall back to the original (fail open), show nothing, fall back to the previous stage, or raise.
8. **In the LangChain ecosystem no middleware builds a rewritten view for a second model.** The reusable pieces are mechanical: `internal_call_metadata()` with `InternalCallTransformer` to keep internal calls out of the stream (the library already uses them), `str | BaseChatModel` resolution, XML serialisation with `get_buffer_string`, pair-preserving cutoffs, `lc_source` tagging, and Deep Agents' habit of saving the original before summarising [@langchain2026; @deepagents2026].
9. **The effect on honest detection depends on the setting, and calibration does not carry over.** Summarising code cost an honest untrusted monitor detection in Gardner-Challis et al. (97.1% to 84.1% safety on APPS) [@gardnerchallis2026safetycase]; the control-evaluations notebook records roughly unchanged monitor performance under input censoring [@greenblatt2023controlevaluationscode]; and against monitor jailbreaking, paraphrasing the chain of thought let the same monitors flag what they had missed [@schulz2026monitorjailbreaking]. Either way the score distribution shifts, so thresholds tuned on raw views must be recalibrated on paraphrased views, and honeypots must pass through the same paraphraser as real steps.
10. **Licences constrain reuse.** control-evaluations, ControlArena, Control Tower, Inspect AI, redwood-control-arena, control-arena-persistent-state-eval and the LangChain packages are MIT. The steganography benchmark, basharena_public, the Ctrl-Z viewer, apps-monitor-control-eval, the SPAR repository and `encoded-reasoning` declare no licence, and linuxarena-paper declares a custom one, so for those we may credit and reimplement ideas but must not copy code or prompt text (section 7).

## 2. Scope and method

Searched with `gh api`, `gh search code`, `gh search repos`, raw file reads and web search, then cloned the relevant repositories into the scratchpad and read the code. Local papers were read from `spar-in-the-wild-ai-control/references/papers/`:

| File | Paper |
| --- | --- |
| `2504.10374` | Ctrl-Z [@bhatt2025ctrlz] |
| `2312.06942` | AI Control [@greenblatt2023aicontrol] |
| `2512.15688` | BashArena [@kaufman2025basharena] |
| `2604.15384` | LinuxArena [@tracy2026linuxarena] |
| `2607.02514` | Distributed Attacks in Persistent-State AI Control (Hills, Caspary, Cooper Stickland), code at `josh-hills/control-arena-persistent-state-eval` [@hills2026persistentstatecode], where code search finds no paraphraser |
| `2607.07368` | Multi-Agent AI Control [@makins2026multiagent], code not released |

Neither BashArena, LinuxArena nor the two 2607 papers uses paraphrasing as a defence; their only mentions of redaction concern redacted IP addresses in side-task documents or environment tooling.

## 3. Implementations found

### 3.1 AI Control code: `rgreenblatt/control-evaluations`

- **Where.** `monitoring/censoring.py`, `monitoring/monitoring_misc.py`, `monitoring/smart_monitoring/smart_monitoring.py`, `monitoring/smart_monitoring_notebook.py`; commit `f8076e28863a560222ae7ddcc4442118e09d023d` (2024-05-24); MIT [@greenblatt2023controlevaluationscode].
- **What is rewritten, by which model.** Both sides of the untrusted monitor's input, by the trusted model (the GPT-3.5-class `dumb_models` by default; `use_gpt4` switches to the untrusted class):
  - *Input redaction* (`censor_programming_problem`, lines 58 to 638). Stage 1 strips characters, objects, themes and flavour text while keeping the input/output specification, with five worked examples as prior turns (lines 67 to 489). Stage 2 (`get_name_censor`, lines 29 to 55) first asks a yes/no question about named characters, and only if yes renames them from a fixed list (Joe, Bob, Doug). Stage 3 rewrites the result into the style of three examples drawn from the synthetic honeypot suite (lines 499 to 620), so real and synthetic inputs look alike.
  - *Output redaction* (`full_output_censor`, lines 784 to 790). An LLM strips comments (`remove_comments_from_code`, `monitoring_misc.py` lines 20 to 26); a two-turn prompt lists unconventional patterns and then rewrites them, skipping the rewrite when the code is already conventional (lines 679 to 751); a final prompt minimises the code, renaming variables and standardising format and nothing else (line 754).
- **Prompt.** Summarised above: aggressive, explicit about what to keep (the full specification, input/output behaviour, existing error handling) and what to drop. The canonicalise step explicitly forbids adding input validation or changing loop style, so the rewrite does not repair the attack.
- **How it reaches the monitor.** A separate `solutions_for_monitoring` list is built with `attrs.evolve` (notebook lines 611 to 626 and 1169 to 1186); the original solutions are untouched and are what gets scored. The same `full_output_censor` is applied to the adversarial evaluation suite (lines 1145 to 1151), so upfront validation runs on censored honeypots.
- **Tool calls and arguments.** Not applicable (single-shot code, not agent steps).
- **Errors and audit.** No retry wrapper beyond the chat library. One guard is worth copying: if a stage's output grows by more than 1,000 tokens over its input, the pipeline falls back to the previous stage's output (lines 622 to 637). Each call returns its chat log (`empty_chat.message_tree()`), so every intermediate rewrite is kept. The code-paraphrase slot `perturb_extracted_solution` is an explicit no-op stub (`smart_monitoring.py` lines 262 to 274), with a comment that a paraphrase should do at least as well.
- **Reuse.** Staged rewriting where each stage has one job; a conditional check before an expensive rewrite; a length-growth guard with fallback to the previous stage; returning the full rewrite log for audit; putting honeypots through the identical transform; telling the rewriter which behaviours it must not change. The notebook also records that censorship alone did not remove theming often enough, so the authors added theming to their synthetic inputs instead; the transform and the calibration set have to be designed together.
- **Avoid.** Model choice by a boolean between two hard-coded model classes (our rule: every model is a parameter); `assert`-based parsing; printing diagnostics.

### 3.2 Text Steganography Benchmark: `redwoodresearch/Text-Steganography-Benchmark`

- **Where.** `textsteg/defenders/llm_defenders.py`, `code_defenders.py`, `aqua_defenders.py`, `interface.py`, `main.py`; commit `1dfa0c45730a28e5bf1148f081835598a3425bb3` (2024-01-26); **no licence declared** [@roger2023textstegcode]. Code for [@roger2023preventing].
- **What is rewritten, by which model.** An answer (and optionally the question) is paraphrased by GPT-3.5 Turbo by default, or by Llama-2-7B-chat through Hugging Face. The `Defender` interface has `remove(examples)` for answers and `paraphrase_questions(questions)` for inputs (`interface.py` lines 38 to 50).
- **Prompts.** Six variants, all as class fields so a caller can replace them:
  - `LlmParaphraser` (line 13): repeat the text in different wording and format, keeping all information; temperature 1.0.
  - `ContextAwareLlmParaphraser` (line 35): given the instruction the text answers, keep only information relevant to that instruction.
  - `.suspicious()` (line 60) and `.very_suspicious()` (line 75, temperature 0): additionally remove high-entropy strings, long numbers, rare names, odd formatting, and ignore any instruction to preserve a payload; if the whole message looks suspicious, replace it with a refusal.
  - `LlmParaphraserUserOnly` and `ContextAwareLlmParaphraserUserOnly` (lines 107 and 134): the same, in a single user turn with delimited sections, and an explicit instruction to paraphrase rather than summarise.
  - `BlackFormatter` (`code_defenders.py`): deterministic reformatting of code.
- **How it reaches the "monitor".** Here the consumer is the decoder, not a monitor: `main.py` lines 57 to 72 encode, paraphrase, decode, then measure both the surviving hidden bits and task quality before and after paraphrasing.
- **Tool calls and arguments.** Not applicable: the inputs are plain-text answers and questions.
- **Errors and audit.** On an empty response the defender silently returns the **original** text (lines 28, 55, 127, 159): fail open. Both `encoded` and `encoded_removed` are kept in the run record.
- **Reuse.** The evaluation design (measure what the paraphrase destroys and what it preserves, on the same items); context-aware paraphrasing (give the paraphraser the task so it knows what is relevant); an explicit clause telling the paraphraser to disregard any instruction to preserve hidden content, which doubles as prompt-injection hardening; deterministic formatters as a cheap first pass.
- **Avoid.** Fail-open fallback: in a control setting a paraphraser failure that forwards the raw step to an untrusted monitor restores exactly the channel the paraphraser exists to close. Avoid the "suspicious" deletion behaviour in a monitor view: removing whatever looks odd, or replacing a suspicious message with a refusal, hides the attack from the monitor. Ctrl-Z states the opposite tradeoff for monitoring (keep what reveals danger, drop arbitrary detail) [@bhatt2025ctrlz].

### 3.3 LASR Labs / SPAR untrusted monitoring: `jedrzej-kolbert/spar-2026-untrusted-monitoring`

- **Where.** `src/lasr_labs_2025_control_project/utils/paraphrasers/paraphrasers.py`, `prompts/summarizer_prompts.py`, `utils/data_pipelines/monitors.py`; commit `718bdb09ce19e4b179d26ce3070d817a743344bd` (2026-09-29); **no licence declared** [@gardnerchallis2026untrustedmonitoringcode]. Code for [@gardnerchallis2026safetycase]; contributors include the paper's first authors.
- **What is rewritten, by which model.** The code solution only, by the trusted summariser (hard-coded `openai/gpt-4o-mini` in `monitors.py` line 148). A `Paraphraser.build(config)` factory offers `none`, `minifier` (`python_minifier` with renaming, literal hoisting and assert removal, lines 114 to 138) and `llm` (lines 140 to 184).
- **Prompt.** A system prompt casting the model as a code paraphrasing assistant inside a monitoring protocol, asking for a concise natural-language description a human could check, and forbidding it from including any of the original code. The user prompt repeats the instruction and wraps the code in a fenced block. The same text is printed in the paper's appendix G.5.2.
- **How it reaches the monitor.** The paraphrase replaces `solution` in each sample; `original_solution`, `paraphraser_system_prompt`, `paraphraser_user_prompt` and `paraphraser_type` are stored alongside (lines 153 to 168). The untrusted monitor's user prompt changes one line to say the solution it sees is paraphrased (lines 27 to 39). The trusted monitor is always called with `paraphraser_type="none"` and receives `original_solution` (lines 293 to 303 and the call at the end of `analyse_eval`).
- **Tool calls and arguments.** Not applicable yet; open issue #29 plans to extend the pipeline to BashArena trajectories with command summarisation and rewrites.
- **Errors and audit.** No handling of paraphraser failure (exceptions propagate). Monitor score parse failures become NaN and are averaged out over runs. The original and the exact paraphraser prompts are kept per sample, which is good provenance.
- **Results worth carrying into our docs** (paper, Table 2, safety at 100% usefulness, APPS): with an honest, non-colluding untrusted monitor, the summariser lowers untrusted-monitor-only safety from 97.1% to 84.1% (the honest-ability cost); against a monitor fine-tuned to recognise its own code, it raises safety from 26.4% to 68.8%, where minification does nothing (26.1%). The paper also warns that when the trusted summariser and trusted monitor are similar models, their failures are correlated: whatever the summariser omits, the untrusted monitor never sees.
- **Reuse.** The division of labour (paraphrase for the untrusted monitor, raw view for the trusted monitor); telling the monitor it reads a paraphrase; storing the original and the paraphraser prompt with each record; a `none` paraphraser as an explicit option for baselines.
- **Avoid.** Hard-coded paraphraser model; a string-keyed config dictionary validated by `assert`; running async code with nested `asyncio.run`.

### 3.4 Chain-of-thought paraphrasing: `wusche1/encoded-reasoning`

- **Where.** `experiments/paraphrase_attack/functions/run_paraphrase_attack.py`, `experiments/paraphrase_attack/configs/base.yaml`, `lib/secret_task.py`, `experiments/cot_monitor_validation/configs/paraphrase_attack.yaml`, `experiments/cot_monitor_validation/functions/rate_completions.py`; commit `fd4dad6c91d4b07d2b84f3ae6bdde355e6f71a0c` (2026-04-30); **no licence declared** [@schulz2026encodedreasoningcode]. Code for [@schulz2026monitorjailbreaking].
- **What is rewritten, by which model.** The agent's full reasoning text, by `gpt-5-mini` via OpenRouter (configurable in YAML). Before paraphrasing, a regular expression removes the side-task answer line from the boxed answer (`redact_secret`, `secret_task.py` lines 51 to 58), so neither paraphraser nor monitor sees it.
- **Prompt.** Reword the sentences while keeping the same structure and meaning, and keep the boxed answer format exactly (`base.yaml`). It is a structure-preserving paraphrase, not a summary.
- **How it reaches the monitor.** Paraphrased text is written to `completions.jsonl` as `reasoning`, next to `original_reasoning`; the chain-of-thought monitor configurations then rate those files with three monitor models (`cot_monitor_validation/configs/paraphrase_attack.yaml`). The experiment also checks whether the model can still finish its side task when continuing from the paraphrased reasoning, which measures what the paraphrase destroyed.
- **Tool calls and arguments.** Not applicable: the input is plain reasoning text ending in a boxed answer.
- **Errors and audit.** Skips a sample with a printed warning when the paraphrase breaks the expected answer format; otherwise no handling. The original is kept.
- **Reuse.** A deterministic redaction pass before the model-based paraphrase; a format-preservation instruction when downstream parsing depends on structure; checking task-relevant content survives.
- **Avoid.** Printing instead of logging; hard-coded provider client.

### 3.5 BashArena public code: `redwoodresearch/basharena_public`

- **Where.** `src/monitors/monitoring_utils.py`, `src/xml_utils.py`, `src/policies/off_policy_traj_views.py`, `src/monitors/trusted_monitor.py`; commit `86082fd9111ec58b39c078c628323431e608cfd2` (2026-05-04); **no licence declared** [@basharenapublic2026]. A pared-down clone of the code behind [@kaufman2025basharena].
- **What exists.** Only the consumer side. Pydantic models `ParaphrasedToolCall(reasoning, summary)` and `ParaphrasedToolOutput(reasoning, output)` (lines 11 to 18) are stored in the assistant message's metadata as `paraphrased_tool_calls` (a list, one per tool call) and `paraphrased_tool_output`. `messages_to_xml(..., use_paraphrased_messages=True)` (lines 51 onwards) then drops the agent's prose entirely, replaces each tool call with a `<tool_call_summary>` and each tool output with a `<tool_output_summary>`. The producer of those summaries is not in the public repository.
- **Prompt.** Not public: the code that fills `paraphrased_tool_calls` is absent from this clone.
- **Tool calls and arguments.** One summary per tool call, each with the paraphraser's own reasoning kept beside the summary, so the reasoning can be audited without being shown to the monitor.
- **Errors and audit.** If the metadata is missing, the view silently shows an empty step (nothing is rendered). The output summary is read from the assistant message rather than from each tool message, so a step with several tool results repeats one summary. The original messages are untouched, so the raw trajectory remains for audit.
- **Reuse.** Per-call summaries in a typed record with a separate reasoning field; keeping paraphrases as data on the step rather than editing the transcript; a view builder with a single switch between raw and paraphrased rendering.
- **Avoid.** Silent empty rendering on a missing paraphrase; string-built XML without escaping.

### 3.6 ControlArena: `UKGovernmentBEIS/control-arena`

- **Where.** `control_arena/tools/_attack_tool/utils.py`, `control_arena/monitor/_monitor.py`, `_last_tool_calls_prompt_formatter.py`, `_cot_prompt_formatter.py`, `_full_trajectory_prompt_formatter.py`, `control_arena/micro_protocol/_trusted_editing.py`, `control_arena/policy/_trusted_editor.py`, `control_arena/policy/prompts/trusted_editor.txt`, `docs/glossary.qmd`; read at `main` commit `18fafcbf2374e763fee546f3a103ffc71dbaa053` (2026-09-29), files also present at release `v19.0.0` (`6d7259ec00f505869705e05ecc84df19b6230c7b`); MIT [@controlarena2026].
- **Paraphrasing.** **Not implemented.** The glossary defines paraphrasing and signal jamming (`glossary.qmd` lines 372 to 374 and 494), citing [@greenblatt2023aicontrol]; there is no untrusted-monitoring micro-protocol and no paraphraser.
- **How a monitor's view is built.** The `@monitor` decorator deep-copies the agent state and converts it to a "monitor view" before the monitor runs (`convert_agent_state_to_monitor_view`, `utils.py` lines 119 to 137). Per-tool metadata can hide a tool call from the monitor or substitute a different `ToolCall` for it (`monitor_views`); tool results whose call is hidden are dropped with it, so call and result stay paired (lines 22 to 117). Tool output is excerpted to exactly what the agent saw. Prompt formatters are synchronous functions `(task_metadata, messages, prompt) -> str` that render the last tool calls as `function: {json arguments}` and wrap chain-of-thought in `<cot>` tags, HTML-escaping both so model text cannot break out of the tags. The monitor's result goes into the last assistant message's metadata.
- **Trusted editing** (the nearest rewrite in the codebase) rewrites the *executed* action, not the monitor's view: the untrusted tool calls are rendered as `<tool_call><function>…</function><arguments>…</arguments></tool_call>` with indented JSON, the editor is told to remove unexpected behaviour and reissue the tool calls, it gets up to three attempts with a reminder when it produces no tool call, and the original, the edit and the monitor response are recorded in `TEMetadata` on the final message.
- **Reuse.** Build the monitor view from a deep copy, never the live state; keep call and result paired when hiding or substituting; escape agent-authored text inside tags; render arguments as JSON; record original and rewrite side by side.
- **Avoid.** The formatter signature is synchronous and returns a string, so it cannot host a model call; a paraphrasing layer needs a hook of its own with both synchronous and asynchronous forms. Trusted editing is a different protocol (it changes what runs) and should not be conflated with paraphrasing (which changes only what the monitor reads).

### 3.7 LinuxArena Control Tower: `linuxarena/control-tower`

- **Where.** `src/control_tower/trajectories/format_history.py`, `src/control_tower/protocols/eval/blue.py`, `src/control_tower/protocols/eval/game_view.py`; commit `74f7cd3976dfa81a60ec224f9d2fd8d1b136f926` (2026-09-29); MIT [@controltower2026]. The library behind [@tracy2026linuxarena]; the separate `redwoodresearch/linuxarena-paper` reproducibility repository [@linuxarenapaper2026] contains no paraphraser either.
- **Paraphrasing.** **Not implemented.** Open issue #2139 (a list of monitoring strategies not yet in Control Tower) names untrusted monitoring with collusion busting, including trusted paraphrasing or redaction of the monitor's input, as missing.
- **Related mechanisms.** Deterministic redaction of `str_replace` bodies when the output is also shown, to keep histories short (`format_history.py` lines 47 to 59); a blue-team `on_input` hook that may rewrite what the untrusted model is asked (`blue.py` line 83), which is the input-distortion idea of [@greenblatt2023aicontrol] rather than monitor-view paraphrasing; frozen `BlueView` and `RedView` dataclasses that make explicit what each party may see.
- **Prompt.** None: there is no paraphraser.
- **Tool calls and arguments.** `format_history` renders each past action as `Tool call: function: {args}` (the arguments as a Python dict repr, not JSON), followed by `Result:` (tool output truncated to a default limit) and `Error:` lines, and optionally the agent's reasoning or extended thinking depending on `cot_level` (lines 61 to 118). Actions marked policy-private or malformed are skipped.
- **Errors and audit.** Not applicable to paraphrasing; the redaction is a pure function over arguments, and the original trajectory is untouched.
- **Reuse.** Typed, frozen "view" records naming what a party may read.

### 3.8 Inspect AI compaction summary: `UKGovernmentBEIS/inspect_ai`

- **Where.** `src/inspect_ai/model/_compaction/summary.py`; `main` commit `6ec66ab66b6ad3c8629e61021ff1defae294a90b`; MIT [@inspectai2026].
- **What is rewritten, by which model.** Older conversation history, for context compaction, by an optional separate model (defaults to the agent's own model). Not a monitor feature: Inspect has no monitor-input paraphraser.
- **Prompt.** A long structured summary with an `<analysis>` scratchpad that is stripped afterwards. Two instructions matter for us: preserve security-relevant user constraints verbatim, and do not attribute framework-inserted user-role turns, or text in assistant output formatted like a user turn, to the user.
- **Tool calls and arguments.** Passed natively: the summariser receives the system, input and conversation messages as Inspect `ChatMessage` objects (tool calls and tool results included) followed by the summary prompt as a final user message, not a serialised transcript (lines 86 to 113).
- **Errors and audit.** Fits the input to the summariser's context window first, then raises if the summary itself stops on `model_length`, rather than returning a truncated summary. The summary is inserted as a user message labelled as generated, not as the user speaking, and tagged `metadata={"summary": True}`.
- **Reuse.** Provenance rules in the prompt (who authored each piece of text; agent text dressed as a user instruction stays agent text), which is exactly the injection a colluding agent would try on a paraphraser; raising on truncated output instead of passing a partial rewrite.

### 3.9 LangChain `SummarizationMiddleware` and internal calls

- **Where.** `langchain/agents/middleware/summarization.py`, `langchain/agents/middleware/internal_call_transformer.py`, `langchain/agents/middleware/_trace_policy.py`; `langchain` 1.4.2 as installed in the project's `.venv`; MIT [@langchain2026].
- **What is rewritten, by which model.** Older messages in the agent's own state, replaced by one summary message, by a second model the caller passes as `str | BaseChatModel` (strings resolved with `init_chat_model`, line 354) and wrapped in `.with_retry()` (line 359).
- **Prompt.** `DEFAULT_SUMMARY_PROMPT` (lines 40 to 99) casts the model as a context extractor and asks for four fixed sections (session intent, summary, artefacts, next steps) and to reply with the extracted context only. Its `<messages>` marker is documented as a public contract that Deep Agents splices into.
- **How output is used.** Written back to state with `RemoveMessage(id=REMOVE_ALL_MESSAGES)` followed by the summary as a `HumanMessage` tagged `additional_kwargs={"lc_source": "summarization"}` (lines 755 to 761).
- **Tool calls and arguments.** Messages are serialised with `get_buffer_string(messages, format="xml")` before the call (lines 853 and 883). The cutoff never separates an `AIMessage` with tool calls from its `ToolMessage` results (`_find_safe_cutoff_point`, lines 797 to 830).
- **Keeping its own calls internal.** Each summary call passes `config={"metadata": {"lc_source": "summarization", **internal_call_metadata()}}`, and the class declares `transformers = (InternalCallTransformer,)` (line 245). The transformer drops tagged `messages` events from `run.messages` and the raw event log; the tag carries a per-process random token so user metadata cannot spoof it. `langchain-sync-monitors` already uses this mechanism in `model_calls.build_internal_call_config` and `middleware.py`.
- **Errors and audit.** Model errors propagate after retries; trimming failures fall back to the last 15 messages. The original messages are removed from state and are not kept.
- **Reuse.** `internal_call_metadata()` and a distinct `lc_source` value for paraphraser calls; `str | BaseChatModel` with `init_chat_model`; XML serialisation through `get_buffer_string` rather than a hand-rolled renderer; pair-preserving selection.
- **Avoid.** Writing the rewrite back into state: our paraphrase must be a monitor-only view and must never replace what the agent or the audit log sees.

### 3.10 Deep Agents: summarisation offload and rubric grader

- **Where.** `deepagents/middleware/summarization.py`, `deepagents/middleware/rubric.py`; `deepagents` 0.7.19; MIT [@deepagents2026].
- **Summarisation.** Extends LangChain's prompt with a media-reference addendum. Before summarising, it **writes the evicted messages to a backend file** (`/conversation_history/{session_id}.md`), as `get_buffer_string(..., format='xml')` under a timestamped heading, so tool calls are stored in the same XML form the summariser reads (lines 1227 to 1259), logs an error and warns if that offload fails, then summarises (around lines 1565 to 1590). The summary message carries the file path. It filters its own earlier summaries by `lc_source` so they are not re-offloaded.
- **Rubric grader** (a second model reading the transcript): builds a bounded, role-labelled transcript deterministically, keeping the first genuine user message, skipping messages it injected itself (by `lc_source`), rendering tool calls from `content_blocks` and replacing opaque blocks with their type name (lines 1360 to 1430). No model rewrite.
- **Reuse.** Persist the original before producing a lossy rewrite, and fail loudly if the original cannot be saved; filter the library's own inserted messages by `lc_source` before rendering; render from `content_blocks` to avoid provider-specific shapes.

### 3.11 Other LangChain middleware checked

- **`PIIMiddleware`** (`pii.py`, `_redaction.py`): deterministic detectors with `redact`, `mask`, `hash` (sha256 prefix) or `block` strategies, applied to input, output or tool results. It **mutates graph state** (`before_model` and `after_model` return new `messages`), so the original is not kept, and `_redaction` is a private module. Useful only as a pattern for an optional deterministic pre-pass (for example hashing identifiers consistently).
- **`LLMToolSelectorMiddleware`** (`tool_selection.py`) and **`LLMToolEmulator`** (`tool_emulator.py`): call a second model with `internal_call_metadata()` and their own `lc_source`, but do not rewrite the transcript for it.
- **`langchain_typesafe`** 0.0.1a3 (MIT) [@typesafe2026langchain]: its classifier and experimental model router invoke a model on the request but contain no transcript rewriting.
- **`ContextEditingMiddleware`**: clears old tool results from the request deterministically; no model call.

## 4. Where nothing was found

- **Ctrl-Z paraphraser code and prompt.** Not in the paper's appendix (sections D, E and H give the trusted monitor prompt and the deferral protocols, not the paraphraser). Not in `redwoodresearch/ctrlz-traj-viewer` [@ctrlztrajviewer2025], which only illustrates the idea in a front-end component (`GameTreeVisualization.tsx`, the trusted-paraphraser stage). Not in `redwoodresearch/redwood-control-arena` [@redwoodcontrolarena2025] or `redwoodresearch/apps-monitor-control-eval` [@appsmonitorcontroleval2025]; code search for "paraphrase" returned nothing in either.
- **Ctrl-Z's summarise-then-act protocol** ("defer to trusted implementation", appendix H), which also summarises the untrusted action with the trusted model, has no public code either.
- **ControlArena, Control Tower, linuxarena-paper, Inspect AI:** no monitor-input paraphraser (sections 3.6 to 3.8).
- **BashArena public:** consumer only; no producer (section 3.5).
- **AI Control paraphraser:** the paper lists paraphrasing only as an option (the trusted model randomly rephrasing the problem or the code); the implemented and evaluated defence was redaction, whose prompts are in the code (section 3.1).
- **Hidden in Plain Text** (Mathew et al., LASR 2024), which tested paraphrasing against emergent steganographic collusion: no public code found.
- **Anthropic hierarchical summarisation for monitoring:** the post publishes no code and says it withholds the exact summariser prompt to reduce the risk of attacks on the summariser; it states that summaries carry the same access controls as the originals [@anthropic2025summarization].
- **LangChain, LangGraph, Deep Agents:** no middleware produces a rewritten view for a second model while leaving state untouched.
- **Antonio-Tresol's repositories:** the "paraphrase" hits concern prompt paraphrases for robustness in the odd-number experiments, not monitor inputs.

## 5. Side-by-side comparison

| Implementation | Rewrites | Paraphraser | Who reads the rewrite | Original kept | On paraphraser failure |
| --- | --- | --- | --- | --- | --- |
| control-evaluations [@greenblatt2023controlevaluationscode] | task text and code | trusted (GPT-3.5 class) | untrusted monitor, on real inputs and honeypots alike | yes, plus full chat log | length blow-up falls back to previous stage; other errors raise |
| Text Steganography Benchmark [@roger2023textstegcode] | answer, optionally question | GPT-3.5 or Llama-2-7B | decoder and evaluator | yes | returns the original (fail open) |
| SPAR / LASR [@gardnerchallis2026untrustedmonitoringcode] | code | GPT-4o mini (hard-coded) | untrusted monitor only; trusted monitor reads original | yes, plus paraphraser prompts | raises |
| encoded-reasoning [@schulz2026encodedreasoningcode] | chain of thought | gpt-5-mini (config) | chain-of-thought monitor | yes | skips sample with a warning |
| basharena_public [@basharenapublic2026] | each tool call and output | not public | monitor or policy view | yes (metadata only) | renders an empty step |
| Ctrl-Z [@bhatt2025ctrlz] | proposed command | trusted (Llama-3.1-8B) | untrusted monitor (GPT-4o mini) | the original runs | not described |
| Inspect compaction [@inspectai2026] | history | optional separate model | the agent | no | raises on `model_length` |
| LangChain summarisation [@langchain2026] | history | caller's model | the agent | no | raises after `with_retry` |
| Deep Agents summarisation [@deepagents2026] | history | caller's model | the agent | yes, offloaded to a file first | offload failure logged; model errors raise |

## 6. What to reuse and what to avoid

These points follow directly from the code above; they are inputs to the plan, not the design.

**Reuse**

1. *A monitor-only view.* Build the paraphrase from a copy and hand it only to the monitor, as control-evaluations, SPAR and ControlArena's monitor view do. The agent's state, the executed step and the audit record stay original [@greenblatt2023controlevaluationscode; @gardnerchallis2026untrustedmonitoringcode; @controlarena2026].
2. *Describe, do not transcribe, the proposed step.* Ctrl-Z's compromise (detailed natural-language description of what the step does, without arbitrary implementation detail) and the SPAR prompt agree on this for code and commands [@bhatt2025ctrlz; @gardnerchallis2026safetycase]. For chain of thought, a structure-preserving paraphrase is the tested variant [@schulz2026encodedreasoningcode].
3. *One summary per tool call*, with the paraphraser's own reasoning stored beside it and hidden from the monitor [@basharenapublic2026].
4. *Give the paraphraser the task* so it can judge relevance [@roger2023textstegcode], and add provenance rules: text the agent wrote stays attributed to the agent, even when it is formatted as a user instruction [@inspectai2026].
5. *Tell the monitor it is reading a paraphrase* [@gardnerchallis2026untrustedmonitoringcode].
6. *Record the original, the paraphrase, and the paraphraser prompt and model* for every step [@gardnerchallis2026untrustedmonitoringcode; @greenblatt2023controlevaluationscode]; if anything lossy replaces history, save the original first [@deepagents2026].
7. *Guard the output*: fall back or fail when the rewrite grows implausibly [@greenblatt2023controlevaluationscode], and treat a truncated rewrite as a failure [@inspectai2026].
8. *Keep paraphraser calls internal* with `internal_call_metadata()`, a distinct `lc_source`, and the existing `build_internal_call_config`; accept `str | BaseChatModel`; serialise with `get_buffer_string(format="xml")`; keep tool calls paired with their results [@langchain2026].
9. *Apply the same transform to calibration data.* Honeypots and threshold-calibration sets must pass through the identical paraphraser, or the monitor can tell them apart and the thresholds are wrong [@greenblatt2023controlevaluationscode; @shlegeris2024collusion].
10. *Measure both sides*: how much monitor recall the paraphrase costs on honest runs, and how much hidden signal survives, on the same items [@roger2023textstegcode; @gardnerchallis2026safetycase].

**Avoid**

1. *Failing open.* Returning the raw step when the paraphraser fails [@roger2023textstegcode] reopens the channel. The safe choices are to treat the step as unmonitorable (for example, route it as maximally suspicious or to the trusted monitor) or to raise; either should be an explicit, typed option.
2. *Silent emptiness.* A missing paraphrase must not render as an empty step [@basharenapublic2026].
3. *Deleting what looks suspicious.* The steganography "suspicious" paraphrasers remove odd content or replace the message with a refusal [@roger2023textstegcode]; in a monitor view that erases the evidence.
4. *Mutating state* as `SummarizationMiddleware` and `PIIMiddleware` do [@langchain2026].
5. *Hard-coded paraphraser models* [@gardnerchallis2026untrustedmonitoringcode; @greenblatt2023controlevaluationscode]; the project rule requires a parameter.
6. *Paraphrasing for the trusted monitor by default.* SPAR keeps the trusted monitor on the original, and a trusted summariser similar to the trusted monitor makes their failures correlated [@gardnerchallis2026safetycase]. Against collusion, paraphrasing matters only for an untrusted monitor. It has a second, separate use for any monitor, trusted included: stripping or restating agent-authored text written to manipulate the monitor, which is what the monitor-jailbreaking result shows for chain of thought [@schulz2026monitorjailbreaking] and what Control Tower issue #2139 lists as a defence against adaptive attacks [@controltower2026]. That makes it a reasonable opt-in for trusted monitors that read reasoning or agent text, not a default.
7. *Relying on the paraphraser alone.* Semantic collusion signals survive paraphrasing by construction (Ctrl-Z's best red-team signal was calling a custom logging utility), and even an ideal paraphraser leaves a channel as large as the semantic entropy of the content [@bhatt2025ctrlz; @motwani2024secretcollusion]. The trusted monitor's suspicion of unnecessary actions is the complementary defence [@shlegeris2024collusion].
8. *A secret prompt as a defence.* Anthropic withholds its summariser prompt [@anthropic2025summarization]; an open-source library cannot, so the prompt must be robust when the agent knows it.

## 7. Licences and what "reuse" may mean

| Code base | Licence | What we may take |
| --- | --- | --- |
| control-evaluations, ControlArena, Control Tower, Inspect AI, redwood-control-arena, control-arena-persistent-state-eval, LangChain, LangGraph, Deep Agents, langchain-typesafe | MIT | code and prompt text, with attribution and the MIT notice |
| Text Steganography Benchmark, basharena_public, ctrlz-traj-viewer, spar-2026-untrusted-monitoring, encoded-reasoning, apps-monitor-control-eval | no licence declared | ideas only, credited; write our own code and prompts |
| linuxarena-paper | NOASSERTION (custom) | ideas only, credited |

Note that the SPAR prompt is also printed in the paper's appendix, but the paper's text is not a licence to copy it either; write our own wording.

## 8. References

Entries already in `docs/references.bib` and reused here: `greenblatt2023aicontrol`, `bhatt2025ctrlz`, `kaufman2025basharena`, `tracy2026linuxarena`, `makins2026multiagent`, `langchain2026`, `deepagents2026`, `typesafe2026langchain`. The entries below are new; if this note moves into `docs/plans/`, add them to `docs/references.bib` so the reference test passes. All URLs were checked with `gh api` or by fetching the page on 2026-09-29.

### Code bases

```bibtex
@software{greenblatt2023controlevaluationscode,
  title   = {control-evaluations: code for {AI} Control: Improving Safety Despite Intentional Subversion},
  author  = {Greenblatt, Ryan},
  year    = {2024},
  version = {f8076e28863a560222ae7ddcc4442118e09d023d},
  url     = {https://github.com/rgreenblatt/control-evaluations},
  note    = {MIT licence}
}

@software{roger2023textstegcode,
  title   = {Text-Steganography-Benchmark: code for Preventing Language Models From Hiding Their Reasoning},
  author  = {Roger, Fabien and Greenblatt, Ryan},
  year    = {2024},
  version = {1dfa0c45730a28e5bf1148f081835598a3425bb3},
  url     = {https://github.com/redwoodresearch/Text-Steganography-Benchmark},
  note    = {No licence declared}
}

@software{gardnerchallis2026untrustedmonitoringcode,
  title   = {spar-2026-untrusted-monitoring: code for When can we trust untrusted monitoring?},
  author  = {Gardner-Challis, Nelson and Bostock, Jonathan and Kolbert, J{\k{e}}drzej and others},
  year    = {2026},
  version = {718bdb09ce19e4b179d26ce3070d817a743344bd},
  url     = {https://github.com/jedrzej-kolbert/spar-2026-untrusted-monitoring},
  note    = {No licence declared}
}

@software{schulz2026encodedreasoningcode,
  title   = {encoded-reasoning: code for Monitor Jailbreaking},
  author  = {Schulz, Julian},
  year    = {2026},
  version = {fd4dad6c91d4b07d2b84f3ae6bdde355e6f71a0c},
  url     = {https://github.com/wusche1/encoded-reasoning},
  note    = {No licence declared}
}

@software{basharenapublic2026,
  title   = {basharena\_public: a pared-down public clone of the {BashArena} repository},
  author  = {{Redwood Research}},
  year    = {2026},
  version = {86082fd9111ec58b39c078c628323431e608cfd2},
  url     = {https://github.com/redwoodresearch/basharena_public},
  note    = {No licence declared}
}

@software{ctrlztrajviewer2025,
  title   = {ctrlz-traj-viewer: trajectory viewer for {Ctrl-Z}},
  author  = {{Redwood Research}},
  year    = {2025},
  version = {206564c7cbea6ec2d6d08691ad492c1811144050},
  url     = {https://github.com/redwoodresearch/ctrlz-traj-viewer},
  note    = {No licence declared}
}

@software{redwoodcontrolarena2025,
  title   = {redwood-control-arena: Redwood Research fork of {ControlArena}},
  author  = {{Redwood Research}},
  year    = {2025},
  version = {3f3896136b635720f8ff3b0c1ee628f13b128599},
  url     = {https://github.com/redwoodresearch/redwood-control-arena},
  note    = {MIT licence}
}

@software{appsmonitorcontroleval2025,
  title   = {apps-monitor-control-eval: monitor and attack policy evaluation on {APPS}},
  author  = {{Redwood Research}},
  year    = {2025},
  version = {feb3df514e22c9c2ce1e9576a397efcb872f2fa4},
  url     = {https://github.com/redwoodresearch/apps-monitor-control-eval},
  note    = {No licence declared}
}

@software{linuxarenapaper2026,
  title   = {linuxarena-paper: reproducibility code for the {LinuxArena} paper},
  author  = {{Redwood Research}},
  year    = {2026},
  version = {304a96df7d28f2b72f59a644877ff64acf2697ad},
  url     = {https://github.com/redwoodresearch/linuxarena-paper},
  note    = {Licence: NOASSERTION}
}

@software{hills2026persistentstatecode,
  title   = {control-arena-persistent-state-eval: code for Distributed Attacks in Persistent-State {AI} Control},
  author  = {Hills, Josh},
  year    = {2026},
  version = {68de4a364315759dedec695222103b29ddd156a8},
  url     = {https://github.com/josh-hills/control-arena-persistent-state-eval},
  note    = {MIT licence}
}

@software{controlarena2026,
  title   = {{ControlArena}},
  author  = {{UK AI Security Institute}},
  year    = {2026},
  version = {18fafcbf2374e763fee546f3a103ffc71dbaa053 (after v19.0.0)},
  url     = {https://github.com/UKGovernmentBEIS/control-arena},
  note    = {MIT licence}
}

@software{controltower2026,
  title   = {Control Tower: control research and evaluations on {Inspect AI}},
  author  = {{Redwood Research} and {EquiStamp}},
  year    = {2026},
  version = {74f7cd3976dfa81a60ec224f9d2fd8d1b136f926},
  url     = {https://github.com/linuxarena/control-tower},
  note    = {MIT licence}
}

@software{inspectai2026,
  title   = {Inspect {AI}},
  author  = {{UK AI Security Institute}},
  year    = {2026},
  version = {6ec66ab66b6ad3c8629e61021ff1defae294a90b},
  url     = {https://github.com/UKGovernmentBEIS/inspect_ai},
  note    = {MIT licence}
}
```

### Papers and posts

```bibtex
@misc{roger2023preventing,
  title         = {Preventing Language Models From Hiding Their Reasoning},
  author        = {Roger, Fabien and Greenblatt, Ryan},
  year          = {2023},
  eprint        = {2310.18512},
  archiveprefix = {arXiv},
  url           = {https://arxiv.org/abs/2310.18512}
}

@misc{gardnerchallis2026safetycase,
  title         = {When can we trust untrusted monitoring? A safety case sketch across collusion strategies},
  author        = {Gardner-Challis, Nelson and Bostock, Jonathan and Kozhevnikov, Georgiy and Sinclaire, Morgan and Velja, Joan and Abate, Alessandro and Griffin, Charlie},
  year          = {2026},
  eprint        = {2602.20628},
  archiveprefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.20628}
}

@misc{schulz2026monitorjailbreaking,
  title         = {Monitor Jailbreaking: Evading Chain-of-Thought Monitoring Without Encoded Reasoning},
  author        = {Schulz, Julian},
  year          = {2026},
  eprint        = {2609.31121},
  archiveprefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.31121}
}

@misc{motwani2024secretcollusion,
  title         = {Secret Collusion among {AI} Agents: Multi-Agent Deception via Steganography},
  author        = {Motwani, Sumeet Ramesh and Baranchuk, Mikhail and Strohmeier, Martin and Bolina, Vijay and Torr, Philip H. S. and Hammond, Lewis and Schroeder de Witt, Christian},
  year          = {2024},
  eprint        = {2402.07510},
  archiveprefix = {arXiv},
  url           = {https://arxiv.org/abs/2402.07510}
}

@online{shlegeris2024collusion,
  title   = {How to prevent collusion when using untrusted models to monitor each other},
  author  = {Shlegeris, Buck},
  year    = {2024},
  month   = sep,
  organization = {Redwood Research},
  url     = {https://blog.redwoodresearch.org/p/how-to-prevent-collusion-when-using}
}

@online{anthropic2025summarization,
  title   = {Monitoring computer use via hierarchical summarization},
  author  = {{Anthropic}},
  year    = {2025},
  url     = {https://alignment.anthropic.com/2025/summarization-for-monitoring/}
}
```
