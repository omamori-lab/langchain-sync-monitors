# Where the ideas come from

This explanation credits every paper, post and codebase the library draws on,
and says what each one contributed. The full entries are in the
[bibliography](../references.bib); code and docs cite them at the point of use
with the same keys.

## Control protocols

Trusted Monitoring and Defer to Trusted come from the original AI control
paper [@greenblatt2023aicontrol]: an untrusted model does the work, a trusted
model rates how suspicious each output is, and suspicious outputs are either
audited or replaced by the trusted model's own output.

Defer to Resample comes from Ctrl-Z [@bhatt2025ctrlz]. When a step looks
suspicious, the protocol draws fresh samples of the same step and runs the
least suspicious acceptable one, falling back to the trusted model when every
sample looks suspicious. The agent never sees the rejected samples, as the
authors' post on the paper stresses [@bhatt2025ctrlzpost], and the rejected
samples become evidence for auditing.

The Auto Mode protocol follows Claude Code's auto mode [@hughes2026automode]: a
classifier reviews each action before it runs, a blocked action is answered
with the reason so the agent can find a safer path, and repeated blocks
escalate. Its limits of three blocks in a row and twenty in a session are the
ones Claude Code uses; here a session is a conversation thread, and the
twenty count the blocks inside its subagents too.

## Attacks on the monitor

The agent writes the steps its monitor reads, so it can aim a prompt injection
at the monitor. Terekhov et al. show that such injections subvert trusted
monitors, and that Defer to Resample makes them worse: running the least
suspicious of several samples gives an injection that sometimes works another
try with every sample, a best-of-n attack [@terekhov2025adaptive]. The same
threat is why the default prompt tells the monitor to read everything as data,
and why a chat judge's score and a guard's label are read only where the
reply's format puts them: a score or label the agent planted, and the monitor
quoted, never becomes the verdict.

## Thresholds and scores

Control settings set thresholds as percentiles of the scores a monitor gives to
honest runs, so a 1% audit budget means flagging the top 1% of honest steps.
BashArena [@kaufman2025basharena] and LinuxArena [@tracy2026linuxarena] show
that integer scores tie at low false-positive rates and that averaging a few
monitor calls restores resolution; Koran et al. [@koran2026ensemble] measure
how much. `CalibratedMonitor` reports a step's smoothed conformal rank among
honest scores [@vovk2005algorithmic], and breaking the remaining ties at random
follows Makins et al. [@makins2026multiagent]. Combining monitors under a
cost budget, and the cascade of a cheap monitor in front of a careful one, draw
on Hua et al. [@hua2025combining].

A monitor's prompt follows Apollo Research's ablations [@apollo2026monitoringprompt]:
reasoning before the score matters most, an
anchored scale calibrates the level, and the task and user messages come
first.

## What a monitor reads

Reading the agent's chain of thought catches more misbehaviour than reading its
actions alone [@baker2025monitoring], but persuasive reasoning can also talk a
monitor out of a correct flag, which is why hybrids that score the two
separately do well [@arnav2025cotredhanded]. Giving a monitor more context does
not always help [@arike2026information]. The default view is therefore
reasoning-blind, with the chain of thought one setting away.

## Guard models and decision models

Guard models such as Llama Guard [@inan2023llamaguard], ShieldGemma
[@zeng2024shieldgemma] and Granite Guardian [@padhi2024graniteguardian] score
with the probability of their "unsafe" label. They are overconfident,
especially under jailbreaks [@liu2025guardcalibration]. When a provider does
not return log-probabilities, the share of several samples that carry the
unsafe label is the standard fallback, and its gains level off after a handful
of samples [@xiong2024uncertainty; @lyu2025sampleconsistency; @tian2023justask].
Llama Guard 3's model card reads its score from the probability of the first
token [@meta2024llamaguard3]. Llama Guard's prompt format puts the label on the
first line, followed for unsafe content by the codes of the violated
categories [@meta2024llamaguard3format; @meta2025llamaguard4].
`GuardModelMonitor` reads a label there or on the reply's last line, and
nowhere else.

Weighting a judge's score by its probabilities, rather than taking the most
likely level, improves ranking [@liu2023geval; @wang2025judgment], which is
the same argument for continuous scores from decision models. Jev is served
through OpenRouter's Decisions API [@openrouter2026decisions] and through
TypeSafe's own client [@typesafe2026langchain]. Asking atomic questions, one
proposition each, and combining them in code is TypeSafe's advice
[@typesafe2026docs].

## The research behind the first plan

Three research notes weighed the evidence for the library's first decisions
before any code was written. They are kept in the repository, under
`docs/plans/initial-implementation/research/`, and cite these sources beyond
the ones above.

### How a guard's label becomes a score

The note compared how guard papers report a score: AEGIS [@ghosh2024aegis]
from log-probabilities, WildGuard [@han2024wildguard] and gpt-oss-safeguard
[@openai2025gptosssafeguard; @openai2025gptosssafeguardguide] with labels
only, and Qwen3Guard [@qwen2025qwen3guard] with a third, controversial
level. It weighed verbalised confidence [@tian2023justask] and
what language models know about their own answers [@kadavath2022know]. It
checked two papers named ProbGuard and found that both address other problems
[@huang2026probguard; @wang2026probguard]. It also describes reading a logit
gap through logit bias [@carlini2024stealing], a technique the library does not
implement.

### Which question a decision model answers

The note weighed rubric grading
[@kim2024prometheus], single-answer against pairwise grading
[@zheng2023judging], checklists [@lee2025checkeval; @cook2024ticking], scale
length [@godfrey2025likert; @li2026gradingscale], proper scoring rules
[@gneiting2007scoring], language models as risk scores [@cruz2024riskscores],
saturated yes or no probabilities [@qi2026verdi], and a calibrated open
reimplementation of Jev [@ren2026openjev]. TypeSafe's launch post
[@typesafe2026jev] and OpenRouter's recipe for gating tool calls with Jev
[@openrouter2026gatetoolcalls] showed how Jev is used in practice.

### Defer to Resample as Ctrl-Z specifies it

The note extracted the protocol, its parameters and its caveats from the paper and from the authors' post
[@bhatt2025ctrlz; @bhatt2025ctrlzpost].

## Code we learned from or build on

The middleware follows LangChain's own middleware [@langchain2026]: how
`LLMToolSelectorMiddleware` calls a second model, how the human-in-the-loop
middleware rejects a tool call with an error result, and how
`InternalCallTransformer` keeps internal model calls out of the message
stream. Messages, content blocks, prompts and the response cache come from
langchain-core [@langchaincore2026]. LangGraph [@langgraph2026] runs the agent
graph: its reducers merge `monitor_log` across subagents, its `nostream` tag
keeps unvetted samples out of the message stream, and its rerun of a node on
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
relative links into absolute ones for PyPI. Ruff
[@ruff2026] lints and formats the code, and ty [@ty2026] checks its types. The
tests run on pytest [@krekel2004pytest] with pytest-asyncio
[@pytestasyncio2026], and the citation test reads this bibliography with
bibtexparser [@bibtexparser2026]. pre-commit [@precommit2026] runs the same
gates before each commit.

These pages are built with MkDocs [@mkdocs2024] and Material for MkDocs
[@mkdocsmaterial2026], with PyMdown Extensions [@pymdownextensions2026] for
the Markdown and Mermaid [@mermaid2026] for the diagrams. mkdocstrings
[@mkdocstrings2026] renders the API reference from the docstrings, and
mkdocs-bibtex [@mkdocsbibtex2025] turns the citations into footnotes.
