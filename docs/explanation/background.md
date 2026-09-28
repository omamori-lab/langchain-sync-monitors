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
sample looks suspicious. The agent never sees the rejected samples, and the
rejected samples become evidence for auditing.

The Auto Mode protocol follows Claude Code's auto mode [@hughes2026automode]: a
classifier reviews each action before it runs, a blocked action is answered
with the reason so the agent can find a safer path, and repeated blocks
escalate. Its limits of three blocks in a row and twenty in a run are the ones
Claude Code uses.

## Thresholds and scores

Control settings set thresholds as percentiles of the scores a monitor gives to
honest runs, so a 1% audit budget means flagging the top 1% of honest steps.
BashArena [@kaufman2025basharena] and LinuxArena [@tracy2026linuxarena] show
that integer scores tie at low false-positive rates and that averaging a few
monitor calls restores resolution; Koran et al. [@koran2026ensemble] measure
how much. Breaking the remaining ties at random follows Makins and Angelini
[@makins2026multiagent]. Combining monitors under a cost budget, and the
cascade of a cheap monitor in front of a careful one, draw on Hua et al.
[@hua2025combining].

A monitor's prompt follows Apollo Research's ablations
[@apollo2026monitoringprompt]: reasoning before the score matters most, an
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
of samples [@xiong2024uncertainty; @lyu2025sampleconsistency].

Weighting a judge's score by its probabilities, rather than taking the most
likely level, improves ranking [@liu2023geval; @wang2025judgment], which is
the same argument for continuous scores from decision models. Jev is served
through OpenRouter's Decisions API [@openrouter2026decisions] and through
TypeSafe's own client [@typesafe2026langchain].

## Code we learned from or build on

The middleware follows LangChain's own middleware [@langchain2026]: how
`LLMToolSelectorMiddleware` calls a second model, how the human-in-the-loop
middleware rejects a tool call with an error result, and how
`InternalCallTransformer` keeps internal model calls out of the message
stream. Deep Agents [@deepagents2026] contributes the subagent model, and its
`RubricMiddleware` the convention of tagging injected messages. TypeSafe's
`AutoModeMiddleware` [@typesafe2026langchain] was the starting point for the
tool-call classifier.

The repository layout, documentation structure and quality gate follow LaNorme
[@lanorme2026]. Retries of network calls use stamina
[@schlawack2026stamina] rather than hand-rolled loops.
