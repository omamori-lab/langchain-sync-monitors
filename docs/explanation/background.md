# Where the ideas come from

This explanation credits every paper, post and codebase the library draws on,
and says what each one contributed. The full entries are in the
[bibliography](../references.bib); code and docs cite them at the point of use
with the same keys.

[TOC]

## At a glance

| Part of the library | What it takes | From |
|---|---|---|
| `DeferToResample` with no resamples: trusted monitoring and defer to trusted | A trusted model rates each output, and a suspicious one is audited or replaced | AI Control [@greenblatt2023aicontrol] |
| `DeferToResample` | A suspicious step is sampled again, and the least suspicious sample runs | Ctrl-Z [@bhatt2025ctrlz] |
| `BlockAndFeedBack`, the default view, `CascadeMonitor` | Block and continue, the limits of three and twenty, a classifier that does not read the agent's own messages, two stages | Claude Code's auto mode [@hughes2026automode] |
| Every monitor, protocol, view, prompt and wrapper | The sync and async monitors and protocols each class builds, the rendering of the trajectory, the protocols' decisions, and the rules for failing closed | hyoshigi [@hyoshigi2026] |
| `DEFAULT_PROMPT` | Reasoning before the score, on an anchored scale | Apollo Research [@apollo2026monitoringprompt] |
| Verdicts read only where the format puts them | The threat of prompt injections aimed at the monitor | Terekhov et al. [@terekhov2025adaptive] |
| `RepeatedMonitor`, `CalibratedMonitor` | Averaged calls, smoothed conformal p-values and random tie-breaking | [Thresholds and scores](#thresholds-and-scores) |
| `GuardMonitor` | The share of sampled replies that carry the unsafe label, or the probability of the label from log-probabilities | [Guard models](#guard-models) |
| `DecisionMonitor` | The probability that each statement holds, from Jev | [Decision models](#decision-models) |

## Control protocols

Trusted Monitoring and Defer to Trusted come from the original AI control
paper (sections 3.2 and 3.3) [@greenblatt2023aicontrol]: an untrusted model
does the work, a trusted model rates how suspicious each output is, and
suspicious outputs are either audited or replaced by the trusted model's own
output. In the library both are `DeferToResample` with no resamples.

Defer to Resample comes from Ctrl-Z (section 3.2) [@bhatt2025ctrlz]. When a
step looks suspicious, the protocol draws up to two more samples of it, runs
the least suspicious one below the threshold, and falls back to the trusted
model when every sample looks suspicious. The paper flags a step for audit on
the lowest score across its samples (section 3.2, footnote 6)
[@bhatt2025ctrlz], and finds that several samples of a step help tell an
attack from honest work (section 3.2.1; figure 4) [@bhatt2025ctrlz], which is
why the library keeps every sample in the record. The authors' post stresses
that resampling is stateless, so a model that keeps no state between calls
does not know it is being resampled (note "Note on statelessness")
[@bhatt2025ctrlzpost], and that the more deterministic the models are, the
less resampling helps (section "Important Limitations")
[@bhatt2025ctrlzpost].

The agent writes the steps its monitor reads, so it can aim a prompt
injection at the monitor. Terekhov et al. show that injections embedded in
the model's outputs let frontier models evade diverse monitors, and that
Defer to Resample backfires: its resampling amplifies the injection into a
best-of-n attack (abstract; sections 4.2 and 6) [@terekhov2025adaptive]. The
same threat is why the default prompt tells the monitor to read everything as
data, and why an LLM monitor's score and a guard's label are read only where
the reply's format puts them.

## Claude Code's auto mode

Claude Code's auto mode reviews each action with a transcript classifier
before the action runs (section "How it works") [@hughes2026automode]. The
library takes four things from Hughes's account of it:

- **What the classifier reads.** It sees the user's messages and the executable
  payload of the agent's tool calls. The agent's own messages, tool-call
  descriptions and all tool outputs are stripped, which makes it
  reasoning-blind by design (section "Why we strip assistant text and tool
  results"; figure 3) [@hughes2026automode]. `Channel.DEFAULT`, the
  default view, follows it, with differences: it shows a call's name and all
  its arguments, and it also reads `<tool_error>` entries, subagent reports
  and the text of the action it rates. When a subagent returns,
  the post's classifier reviews its whole action history, and a flag only
  adds a warning (section "Multi-agent handoffs") [@hughes2026automode]; the
  library shows the report to the parent's monitor as a `<subagent_report>`.
- **Deny and continue.** A denied action comes back to the agent with the
  reason and an instruction to find a safer path rather than route around
  the block (section "Deny-and-continue"; figure 4) [@hughes2026automode].
  `BlockAndFeedBack` does the same, with the middleware's `FeedbackTemplate`.
- **The limits.** After three consecutive denials or twenty in all within a
  session, Claude Code stops the model and escalates to a person, or ends the
  process when there is no one to ask (section "Deny-and-continue")
  [@hughes2026automode]. `BlockAndFeedBack` uses the same numbers; here a
  session is the monitor's trajectory of the conversation thread, the twenty
  include the blocks inside its subagents, and at either limit the next
  proposal is audited, which halts the run, since no person can answer inside
  the model node.
- **Two stages.** A fast single-token filter runs first, and reasoning
  follows only when it flags the action (section "Two-stage classification:
  efficiency and recall") [@hughes2026automode]. `CascadeMonitor` runs a
  fast monitor on every step and a careful one on the steps the fast one
  finds suspicious.

The rule that everything the agent chooses on its own is unauthorised until
the user asks for it comes from the same post, and `DEFAULT_PROMPT`
states it.

## What a monitor reads

The default view leaves the chain of thought out, as auto mode does, and one
setting adds it. Studies of reading the chain of thought found this:

- Baker et al. find that a monitor reading the chain of thought catches far
  more reward hacking than one reading actions alone [@baker2025monitoring].
- Arnav et al. find that reading the chain of thought helps most on subtle
  sabotage, that its rationalisations can mislead the monitor on obvious
  sabotage, and that a hybrid scoring the chain of thought and the actions
  separately beats both [@arnav2025cotredhanded].
- Arike et al. find that monitors with access to the chain of thought beat
  action-only monitors in almost all their settings. Monitors also often do
  better with less information, mostly when a monitor reading only the chain
  of thought beats one reading it with the actions [@arike2026information].

## Prompts and scores

A monitor's prompt follows Apollo Research's ablations. Structured reasoning
before the score is the component whose removal hurts most, and a rubric with
a worked example calibrates the absolute level of the score rather than its
ranking [@apollo2026monitoringprompt].

### Thresholds and scores

Control evaluations set thresholds as percentiles of the scores a monitor
gives to honest runs, so an audit budget of 1% means flagging for review the
top 1% of honest steps. LinuxArena puts its 1% budget at the 99th percentile
of honest scores (section 5.1) [@tracy2026linuxarena], and
[Combine and calibrate monitors](../how-to/combine-and-calibrate-monitors.md)
gives the budgets of AI Control and Ctrl-Z.

`RepeatedMonitor` draws on BashArena, LinuxArena, Ctrl-Z and Koran et al.
[@kaufman2025basharena; @tracy2026linuxarena; @bhatt2025ctrlz; @koran2026ensemble];
`RepeatedMonitor` in hyoshigi's
[API reference](https://omamori-lab.github.io/hyoshigi/reference/api/) says
what each contributed. `CalibratedMonitor`, which reports a step's smoothed
conformal p-value against honest scores and breaks the remaining ties at
random, draws on Vovk et al., Makins et al. and LinuxArena
[@vovk2016criteria; @makins2026multiagent; @tracy2026linuxarena];
[hyoshigi's page on calibration](https://omamori-lab.github.io/hyoshigi/explanation/calibration/)
defines the p-value.

Hua et al. combine monitors under a cost budget, and their best two-monitor
protocol calls the expensive monitor only when the cheap score falls in an
uncertain band (section 6) [@hua2025combining]. `CascadeMonitor` uses a
simpler rule, one threshold on the cheap score, which is the multi-stage
classifier of Cunningham et al. (section Multi-stage classifiers)
[@cunningham2025cheapmonitors].

In LLM-as-a-judge evaluation [@zheng2023judging], weighting a judge's score
by the probabilities of its levels gives a finer, continuous score
[@liu2023geval], and taking the mean of the judgement distribution
outperforms taking its most likely level [@wang2025judgment];
the same argument favours the continuous probabilities of decision models.

## Guard models

Guard models such as Llama Guard [@inan2023llamaguard], ShieldGemma
[@zeng2024shieldgemma] and Granite Guardian [@padhi2024graniteguardian] score
with the probability of the label that marks content unsafe, `unsafe` for Llama
Guard and `Yes` for the other two, and Llama Guard 3's model card takes the
probability of the first generated token as the unsafe class's
[@meta2024llamaguard3].

The share of several samples that carry the unsafe label estimates the same
probability without log-probabilities, and `GuardMonitor` scores that share, or
the probability itself when `top_log_probabilities` is above 0.
How many samples to draw draws on Lyu et al., Xiong et al. and Tian et al.,
and how guard models are calibrated on Liu et al.
[@lyu2025sampleconsistency; @xiong2024uncertainty; @tian2023justask; @liu2025guardcalibration];
[A guard monitor scores the share of its samples labelled suspicious](https://omamori-lab.github.io/hyoshigi/how-to/choosing-a-monitor/#a-guard-monitor-scores-the-share-of-its-samples-labelled-suspicious)
gives the evidence.

Llama Guard 3 writes its label on the first line and, for unsafe content, the
codes of the violated categories on the next [@meta2024llamaguard3format].
Llama Guard 4 also lists the violated categories after an unsafe label
[@meta2025llamaguard4]. `GuardMonitor` reads a label on the reply's last
line, or on its first line when that is a suspicious label followed only by
category codes, or after the key a guard's labels name, and nowhere else.

## Decision models

Jev gives the probability that a statement holds. It is served through
OpenRouter's Decisions API [@openrouter2026decisions] and through TypeSafe's
own API, which TypeSafe's SDK reaches [@typesafe2026sdk]. TypeSafe advises
asking atomic questions, one factor each, and combining the answers in code
[@typesafe2026docs].

## The research behind the first plan

Three research notes weighed the evidence for the library's first decisions
before any code was written. They are kept in the repository, under
`docs/plans/initial-implementation/research/`, and cite these sources beyond
the ones above:

| Note | Sources |
|---|---|
| How a guard's label becomes a score | AEGIS scores from log-probabilities [@ghosh2024aegis]; WildGuard and gpt-oss-safeguard report labels only [@han2024wildguard; @openai2025gptosssafeguard; @openai2025gptosssafeguardguide]; Qwen3Guard adds a controversial level [@qwen2025qwen3guard]; verbalised confidence [@tian2023justask] and what models know about their own answers [@kadavath2022know]; two papers named ProbGuard, both on other problems [@huang2026probguard; @wang2026probguard]; reading a logit gap through logit bias, which the library does not implement [@carlini2024stealing]. |
| Which question a decision model answers | Rubric grading [@kim2024prometheus]; single-answer against pairwise grading [@zheng2023judging]; checklists [@lee2025checkeval; @cook2024ticking]; scale length [@godfrey2025likert; @li2026gradingscale]; proper scoring rules [@gneiting2007scoring]; language models as risk scores [@cruz2024riskscores]; saturated yes or no probabilities [@qi2026verdi]; a calibrated open reimplementation of Jev [@ren2026openjev]; Jev in practice [@typesafe2026jev; @openrouter2026gatetoolcalls]. |
| Defer to Resample as Ctrl-Z specifies it | The protocol, its parameters and its caveats, from the paper and the authors' post [@bhatt2025ctrlz; @bhatt2025ctrlzpost]. |

## The research behind the score export

The research under `docs/plans/monitor-scores/research/` asks how LangSmith
and Langfuse each expect a guardrail's or an evaluator's result to be
recorded, which led to the `export_scores` option. It cites these sources
beyond the ones above.

### LangSmith

Feedback's shape, sources and targets come from LangSmith's feedback pages
[@langsmith2026feedback; @langsmith2026feedbackformat; @langsmith2026feedbackcriteria].
What its filters accept comes from the trace query syntax
[@langsmith2026querysyntax], and the `ls_` metadata keys from their reference
[@langsmith2026metadataparameters]. The cost of feedback that extends a
trace's retention is in the administration overview
[@langsmith2026retention], and what a dashboard can chart is in the dashboards
page [@langsmith2026dashboards]. Server-side scoring draws on code online
evaluators [@langsmith2026codeevaluators], automation rules
[@langsmith2026rules] and an evaluator's retention setting
[@langsmith2026evaluatorretention]. LangSmith's OpenTelemetry mapping is
described in its own guide [@langsmith2026otel].

### Langfuse

Scores and when to prefer them to tags come from the scores overview
[@langfuse2026scores], and the ways to score a LangChain run from the
integration guide [@langfuse2026langchain]. The research also read the pages on
observation types [@langfuse2026observationtypes], log levels
[@langfuse2026levels], trace ids [@langfuse2026traceids], naming
[@langfuse2026bestpractices], code evaluators [@langfuse2026codeevaluators],
LLM-as-a-judge [@langfuse2026llmjudge], custom dashboards
[@langfuse2026dashboards], guardrails [@langfuse2026guardrails] and
OpenTelemetry [@langfuse2026otel].

### OpenTelemetry and comparable libraries

The GenAI semantic conventions define the evaluation event
[@otel2026genai], and the Python package marks their constants as moved
[@otel2026semconvpython]. Two open proposals would add guardrail spans
[@otel2026guardrailproposal] and a decision event for a proposed tool call
[@otel2026tooldecisionproposal]. For how other libraries report a verdict,
the research read openevals [@openevals2026], NeMo Guardrails
[@nemoguardrails2026], Guardrails AI [@guardrailsai2026], the OpenAI Agents
SDK [@openaiagents2026] and OpenInference's instrumentation of it
[@openinference2026].

## Code we learned from or build on

The monitors are hyoshigi's [@hyoshigi2026], the framework-free control core
extracted from this library: each monitor class builds hyoshigi's sync and
async monitors, which rate an action against a trajectory, and the views,
prompts, guard labels, decision statements, wrappers and rules for failing
closed are hyoshigi's own. hyoshigi writes each rating once, free of I/O, as
a plan that a sync and an async driver run, after the sans-I/O pattern
[@cannon2016sansio].

The middleware follows LangChain's own middleware [@langchain2026]: how
`LLMToolSelectorMiddleware` calls a second model, how the human-in-the-loop
middleware rejects a tool call with an error result, and how
`internal_call_metadata()` marks an internal model call, which
`InternalCallTransformer` then drops from the `stream_events(version="v3")`
projection. Messages, content
blocks, prompts and the response cache come from langchain-core
[@langchaincore2026]. LangGraph [@langgraph2026] runs the agent graph: its
reducers merge `monitor_log` across subagents, its `nostream` tag keeps
unvetted samples out of `stream_mode="messages"`, and its rerun of a node on
resume is why nothing in the model node may interrupt. Deep Agents [@deepagents2026]
contributes the subagent model, and its `RubricMiddleware` the convention of
tagging injected messages. TypeSafe's `AutoModeMiddleware`
[@typesafe2026langchain] was the starting point for the tool-call classifier.

The `openrouter` extra installs langchain-openrouter
[@langchainopenrouter2026], which reaches OpenRouter through its Python SDK
[@openrouterpythonsdk2026] and maps a reply's reasoning into LangChain's
content blocks. An `openrouter:` decision model calls the Decisions API with
httpx [@httpx2024], validates the response with pydantic [@pydantic2026], and
retries network failures with stamina [@schlawack2026stamina] rather than
hand-rolled loops.

## Built with

The repository layout, documentation structure and quality gate follow LaNorme
[@lanorme2026], which also checks the code and prose. uv [@uv2025] manages the
environment and builds the package with hatchling [@hatchling2026], whose
hatch-fancy-pypi-readme plugin [@hatchfancypypireadme2025] turns the README's
relative links into absolute ones for PyPI. Ruff [@ruff2026] lints and formats
the code, and ty [@ty2026] checks its types. The tests run on pytest
[@krekel2004pytest] with pytest-asyncio [@pytestasyncio2026], and the citation
test reads this bibliography with bibtexparser [@bibtexparser2026]. pre-commit
[@precommit2026] runs the fast gates before each commit: ruff, ty, LaNorme and
the tests.

These pages are built with MkDocs [@mkdocs2024] and Material for MkDocs
[@mkdocsmaterial2026], with PyMdown Extensions [@pymdownextensions2026] for
the Markdown. The diagrams are the lab's own figures, SVGs drawn in its house
style in a light and a dark version. mkdocstrings
[@mkdocstrings2026] renders the API reference from the docstrings, and
mkdocs-bibtex [@mkdocsbibtex2025] turns the citations into footnotes.

## References
