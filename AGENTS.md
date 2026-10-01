# AGENTS.md

Guidance for coding agents working in this repository (the
[agents.md](https://agents.md/) standard). `CLAUDE.md` is a generated copy kept
in sync by `scripts/sync-agents.sh`; edit this file, then run that script.

langchain-sync-monitors adds control monitors to LangChain agents and Deep
Agents as middleware. A monitor judges every step an agent proposes before
any of the agent's own tools run; a control protocol decides what happens
with that judgement.

## Before you finish a change

Run the gates and make sure they pass:

```console
scripts/check.sh
```

It runs the agent-file sync check, ruff, ty (Astral's type checker, warnings
treated as errors), the offline test suite, the lanorme standards, a strict
docs build, a package build and an import of the built wheel without any
extra. CI runs the same script. The pre-commit hooks run only its fast gates,
ruff, ty, lanorme and the offline suite, plus a gitleaks secrets scan and file
checks, so they do not replace the script. Do not finish with a red gate.
`uv run --group dev ruff check --fix . && uv run --group dev ruff format .`
fixes what ruff reports; `uvx --python 3.13 lanorme==0.21.0 rule CODE` explains
a lanorme finding.

## Project facts

- Python 3.12 or newer. Setup: `uv sync --group dev --group docs --all-extras`,
  then `uvx pre-commit install`.
- Core dependencies: `langchain`, `langchain-core`, `langgraph`, `pydantic`,
  `httpx`, `stamina`. Providers sit behind extras: `deepagents`, `openrouter`,
  `typesafe`.
- Plans live in `docs/plans/<feature>/`, one directory per feature: the plan
  (diagrams and pseudocode) and the research behind its decisions. The first
  is `docs/plans/initial-implementation/`. A new feature gets a new directory.
- Shared interfaces live in `src/langchain_sync_monitors/contracts.py`. Code
  against them; change them only in a dedicated pull request.
- Errors live in `errors.py` and derive from `MonitorError`. Diagnostics go
  through `logging.getLogger(__name__)`, never `print`.

## How the code is designed

Keep to these patterns; reviewers check them.

- **Simple for the user, configurable for the expert.** A monitored agent is
  one line, `MonitorMiddleware(monitor=..., protocol=...)`. Every behaviour
  is an option with a sensible default (never a default model), checked when
  the object is built, and anything beyond the options is an implementation
  of a contract.
- **Program to the contracts.** `contracts.py` defines the interfaces:
  `Monitor`, `ControlProtocol`, `Fallback` and `PendingStep`, with
  `DecisionModel` in `monitors/decision.py`. Code depends on these, not on
  concrete classes, so any monitor works with any protocol.
- **A strategy for every choice.** Behaviour that can vary is an object or an
  enum passed in: a protocol, a fallback, a monitor, a decision model,
  `GuardScoring`, `Combine`, `MonitorView`, `FeedbackVisibility`,
  `SubagentHalt`, `Resampling`. No flag switches behaviour inside one class.
- **Composition over inheritance.** Monitors compose: `RepeatedMonitor`,
  `CalibratedMonitor` and `CascadeMonitor` are monitors that hold monitors.
  Inherit only from a contract, or from a template with named hooks such as
  `ChatModelMonitor`.
- **A small core of functions, a thin shell.** The middleware wires
  LangChain's hooks; the decisions live in small functions over typed records
  (`halts.py`, `records.py`, `transcript.py`, `task_authorship.py`), each
  testable on its own, and shared by the sync and async paths.
- **One layer touches LangChain's untyped surfaces.** `_langchain.py` reads
  and writes them; everything else sees typed values.
- **Wrong states cannot be built.** Typed records, enums and frozen dataclasses
  carry the data; external payloads are validated with pydantic where they
  enter; a value the library cannot read fails closed.
- **One source of truth.** Each state key is a named constant, gathered in
  `state_keys.py`; the messages the monitor writes into a run, such as
  `DEFAULT_HALT_MESSAGE` and `STANDING_HALT_MESSAGE`, are named constants; and
  each documented fact lives on one page.

## Rules that the gates do not fully catch

- **Every model is a parameter.** No monitor, protocol or example picks a model
  by default. Accept `str | BaseChatModel` and resolve strings with
  `init_chat_model`.
- **Names.** Functions and methods are verbs, named for what they do, verb
  first (`build_`, `render_`, `read_`, `is_`); classes, modules and packages
  are nouns, named for what they are. Use full words: no shorthands or
  abbreviations beyond standard ones such as `id` or `url`.
  LangChain's fixed hook names, such as `wrap_model_call`, `awrap_tool_call`
  and `aafter_model`, are the only exception.
- **Keyword-only parameters** for every public constructor and function after
  the first positional one, except LangChain hooks marked with `@override`.
- **Intentional types.** No `Any` and no `dict[str, Any]` for data whose shape
  we know: use `StrEnum`, `Literal`, frozen dataclasses, `TypedDict` or pydantic
  models. Untyped LangChain surfaces are confined to `_langchain.py`.
- **Prefer a canonical library to hand-rolled logic**: stamina for retries of
  network calls, pydantic for validating external payloads, httpx for HTTP,
  the standard library `statistics` and `bisect` for numbers. Chat models
  retry network and server errors on their own (`max_retries`), so do not wrap
  them in stamina. HTTP 429 is the exception: `ChatOpenRouter` does not retry
  it, so a chat monitor retries its own calls on one (`monitors/chat.py`).
- **Cite sources at the point of use.** When code or a docs page takes an idea,
  a protocol, a number or a code pattern from a paper, a post or another
  codebase, cite it with `[@key]` in the docstring or text and add the entry to
  `docs/references.bib`. Cite code bases as `@software`. The unit tests fail on
  a cited key that is missing from the bibliography and on an entry nothing
  cites.
- **State records hold plain values**: `str`, `int`, `float`, `bool`, `None`,
  lists and `TypedDict`; enums are stored as `Literal` strings; lists, not
  tuples. Declare `monitor_log` as `Annotated[list[StepRecord], OmitFromInput,
  operator.add]`, with the reducer last: LangGraph only reads the reducer from
  the last position.
- **Every message the monitor inserts gets a fresh id** (`monitor-<uuid4>`), and
  feedback is marked in its text, because some providers drop `name`.
- **No per-run state on `self`.** Middleware instances are shared across
  parallel subagent runs; keep run state in the graph state.
- **The monitor sits last** in a `create_agent` middleware list, so no other
  middleware returns commands from inside it.

## How we build features

1. Understand and design before writing: read the relevant code, and weigh
   more than one design where the approach is open.
2. Implement the coherent change in one place.
3. Test end to end: a positive case, a negative case, the boundary, and a
   regression test for the exact behaviour, under both `invoke` and `ainvoke`.
   Tests use `# Arrange`, `# Act`, `# Assert` markers.
4. Review adversarially for correctness, resilience, duplication and test
   strength, and reproduce each finding before acting on it.
5. Open one pull request per issue group, with `Closes #N` in the description.

## Documentation

Docs follow the Diataxis layout under `docs/`: tutorials, how-to guides,
reference and explanation. Every page has one level-1 heading and opens with a
line that starts "This page", "This tutorial", "This guide", "This reference",
"This how-to" or "This explanation". Prose uses British spelling, no em dashes
and no emoji. Docs state current truth only; history lives in `CHANGELOG.md`.
Docs change with the code in the same pull request: a change to behaviour, an
option, a name or a message updates every page, docstring, example and figure
that states it. Add a `## [Unreleased]` entry for anything a user would
notice.
