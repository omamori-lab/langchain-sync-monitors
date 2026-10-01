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
a lanorme finding. lanorme's limits in `pyproject.toml` are a ratchet: meet them
by naming steps or sharing a helper, not by raising a limit, and keep inline
ignores within the `max_total` budget there.

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

## Rules that the gates do not fully catch

- **Every model is a parameter.** No monitor, protocol or example picks a model
  by default. Accept `str | BaseChatModel` and resolve strings with
  `init_chat_model`.
- **Names.** A function is named for what it does, verb first (`build_`,
  `render_`, `read_`, `is_`); modules and classes are nouns. Use full words: no
  shorthands or abbreviations beyond standard ones such as `id` or `url`.
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
Add a `## [Unreleased]` entry for anything a user would notice.
