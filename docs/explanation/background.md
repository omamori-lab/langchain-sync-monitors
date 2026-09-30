# Where the ideas come from

This explanation credits every paper, post and codebase the library draws on,
and says what each one contributed. The full entries are in the
[bibliography](../references.bib); code and docs cite them at the point of use
with the same keys.

[TOC]

## At a glance

| Part of the library | What it takes | From |
|---|---|---|
| `TrustedMonitoring`, `DeferToTrusted` | A trusted model rates each output, and a suspicious one is audited or replaced | AI Control [@greenblatt2023aicontrol] |
| `DeferToResample` | A suspicious step is sampled again, and the least suspicious sample runs | Ctrl-Z [@bhatt2025ctrlz] |
| `AutoMode`, the default view, `CascadeMonitor` | Block and continue, the limits of three and twenty, a classifier that does not read the agent's own messages, two stages | Claude Code's auto mode [@hughes2026automode] |
| `DEFAULT_MONITOR_PROMPT` | Reasoning before the score, on an anchored scale | Apollo Research [@apollo2026monitoringprompt] |
| Verdicts read only where the format puts them | The threat of prompt injections aimed at the monitor | Terekhov et al. [@terekhov2025adaptive] |
| `RepeatedMonitor`, `CalibratedMonitor` | Averaged calls, conformal ranks and random tie-breaking | [Thresholds and scores](#thresholds-and-scores) |
| `GuardModelMonitor` | The probability of the unsafe label, or the share of samples that carry it | [Guard models](#guard-models) |
| `DecisionModelMonitor` | Yes or no probabilities from Jev | [Decision models](#decision-models) |

## Control protocols

Trusted Monitoring and Defer to Trusted come from the original AI control
paper [@greenblatt2023aicontrol]: an untrusted model does the work, a trusted
model rates how suspicious each output is, and suspicious outputs are either
audited or replaced by the trusted model's own output.

Defer to Resample comes from Ctrl-Z [@bhatt2025ctrlz]. When a step looks
suspicious, the protocol draws up to two more samples of it, runs the least
suspicious one below the threshold, and falls back to the trusted model when
every sample looks suspicious. The paper flags a step for audit on the lowest
score across its samples, and finds that several samples of a step help tell
an attack from honest work, which is why the library keeps every sample in
the record. The authors' post stresses that resampling is stateless, so the
model does not know it is being resampled, and that the more deterministic
the models are, the less resampling helps [@bhatt2025ctrlzpost].

The agent writes the steps its monitor reads, so it can aim a prompt
injection at the monitor. Terekhov et al. show that injections embedded in
the model's outputs let frontier models evade diverse monitors, and that
Defer to Resample backfires: its resampling amplifies the injection into a
best-of-n attack [@terekhov2025adaptive]. The same threat is why the default
prompt tells the monitor to read everything as data, and why a chat judge's
score and a guard's label are read only where the reply's format puts them.

## Claude Code's auto mode

Claude Code's auto mode reviews each action with a transcript classifier
before the action runs [@hughes2026automode]. The library takes four things
from Hughes's account of it:

- **What the classifier reads.** It sees the user's messages and the agent's
  tool calls, with the agent's own messages and the tool outputs stripped
  out, which makes it reasoning-blind by design. `Channel.ACTIONS`, the
  library's default view, follows it. The library also shows the text of a
  final answer, since that answer is the step's action.
- **Deny and continue.** A denied action comes back to the agent as a tool
  result, with an instruction to find a safer path rather than route around
  the block. `AutoMode` does the same, and its default feedback also carries
  the monitor's reason.
- **The limits.** After three consecutive denials or twenty in all within a
  session, Claude Code stops the model and escalates to a person, or ends the
  process when there is no one to ask. `AutoMode` uses the same numbers; here
  the three count within one step, a session is a conversation thread, and
  the twenty include the blocks inside its subagents.
- **Two stages.** A fast single-token filter runs first, and reasoning
  follows only when it flags the action. `CascadeMonitor` runs a fast monitor
  on every step and a careful one on the steps the fast one finds
  suspicious.

The rule that everything the agent chooses on its own is unauthorised until
the user asks for it comes from the same post, and `DEFAULT_MONITOR_PROMPT`
states it.

## What a monitor reads

Reading the agent's chain of thought catches more misbehaviour than reading
its actions alone [@baker2025monitoring]. Reasoning traces can also carry
rationalisations that mislead a monitor, and a hybrid that scores reasoning
and actions separately and combines them outperforms either alone
[@arnav2025cotredhanded]. Giving a monitor more information does not always
help: current monitors often do better with less [@arike2026information].
The default view leaves the chain of thought out, and one setting adds it.

## Prompts and scores

A monitor's prompt follows Apollo Research's ablations. Structured reasoning
before the score is the component whose removal hurts most, and a rubric with
a worked example calibrates the absolute level of the score rather than its
ranking [@apollo2026monitoringprompt].

### Thresholds and scores

Control settings set thresholds as percentiles of the scores a monitor gives
to honest runs, so a 1% audit budget means flagging the top 1% of honest
steps. BashArena shows that averaging a few calls of a monitor that scores
from 0 to 10 raises its resolution at low false-positive rates
[@kaufman2025basharena], and LinuxArena averages two calls per action
[@tracy2026linuxarena]. Koran et al. note that integer scores from 0 to 10
leave a sparse threshold space, and find the plain mean as good a way to
combine monitors' scores as the other aggregators they tried
[@koran2026ensemble].

`CalibratedMonitor` reports a step's smoothed conformal rank among honest
scores [@vovk2005algorithmic], and breaks the remaining ties at random, as
Makins et al. and LinuxArena do [@makins2026multiagent; @tracy2026linuxarena].
Hua et al. combine monitors under a cost budget, and their best two-monitor
protocol calls the expensive monitor only when the cheap score falls in an
uncertain band [@hua2025combining]. `CascadeMonitor` uses a simpler rule, one
threshold on the cheap score.

Weighting a judge's score by the probabilities of its levels, rather than
taking the most likely level, gives finer and better judgements
[@liu2023geval; @wang2025judgment]; the same argument favours the continuous
probabilities of decision models.

## Guard models

Guard models such as Llama Guard [@inan2023llamaguard], ShieldGemma
[@zeng2024shieldgemma] and Granite Guardian [@padhi2024graniteguardian] score
with the probability of their "unsafe" label, and Llama Guard 3's model card
takes the probability of the first generated token as the unsafe class's
[@meta2024llamaguard3]. Guard models are overconfident, and markedly
miscalibrated under jailbreak prompts [@liu2025guardcalibration].

When a provider returns no log-probabilities, the share of several samples
that carry the unsafe label estimates the same probability. The evidence for
sampling comes from studies of confidence in question answering, not of
guards: Xiong et al. and Tian et al. see its gains converge as samples grow,
and Lyu et al. see them saturate after 15 to 20 samples
[@xiong2024uncertainty; @tian2023justask; @lyu2025sampleconsistency].

Llama Guard's prompt format puts the label on the first line, followed for
unsafe content by the codes of the violated categories
[@meta2024llamaguard3format; @meta2025llamaguard4]. `GuardModelMonitor` reads
a label there, or on the reply's last line, and nowhere else.

## Decision models

Jev answers yes or no questions with a probability. It is served through
OpenRouter's Decisions API [@openrouter2026decisions] and through TypeSafe's
own client [@typesafe2026langchain]. Asking atomic questions, one proposition
each, and combining them in code is TypeSafe's advice [@typesafe2026docs].

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

## Code we learned from or build on

The middleware follows LangChain's own middleware [@langchain2026]: how
`LLMToolSelectorMiddleware` calls a second model, how the human-in-the-loop
middleware rejects a tool call with an error result, and how
`InternalCallTransformer` marks internal model calls so that the
`stream_events(version="v3")` projection drops them. Messages, content
blocks, prompts and the response cache come from langchain-core
[@langchaincore2026]. LangGraph [@langgraph2026] runs the agent graph: its
reducers merge `monitor_log` across subagents, its `nostream` tag keeps
unvetted samples out of `stream_mode="messages"`, and its rerun of a node on
resume is why a fallback must not interrupt. Deep Agents [@deepagents2026]
contributes the subagent model, and its `RubricMiddleware` the convention of
tagging injected messages. TypeSafe's `AutoModeMiddleware`
[@typesafe2026langchain] was the starting point for the tool-call classifier.

The `openrouter` extra installs langchain-openrouter
[@langchainopenrouter2026], which reaches OpenRouter through its Python SDK
[@openrouterpythonsdk2026] and maps a reply's reasoning into LangChain's
content blocks. `OpenRouterDecisionModel` calls the Decisions API with httpx
[@httpx2024], validates the response with pydantic [@pydantic2026], and retries
network failures with stamina [@schlawack2026stamina] rather than hand-rolled
loops. Chat monitors write their scoring once, free of I/O, after the sans-I/O
pattern [@cannon2016sansio].

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
the Markdown and Mermaid [@mermaid2026] for the diagrams. mkdocstrings
[@mkdocstrings2026] renders the API reference from the docstrings, and
mkdocs-bibtex [@mkdocsbibtex2025] turns the citations into footnotes.

## References
