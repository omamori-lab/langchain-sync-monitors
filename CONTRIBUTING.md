# Contributing to langchain-sync-monitors

Thank you for helping. This guide covers how the project is set up, the gates a
change must pass, and how to send a pull request.

## Where to start

- Read `docs/plans/initial-implementation/plan.html` in a browser for the design: the diagrams, the
  pseudocode and the decisions taken so far.
- Pick an open issue. Issues are grouped into milestones by phase and labelled
  by area (`monitors`, `protocols`, `middleware`, `deepagents`, `docs`).
- For anything larger than a fix, comment on the issue first so the design can
  be agreed before the code.

## Principles

- **Every model is a parameter.** The library never picks a model for you.
- **Core dependencies stay small.** Providers sit behind extras.
- **Types say what the data is.** No `Any` where the shape is known.
- **Canonical libraries over hand-rolled logic.** Retries, validation and HTTP
  use stamina, pydantic and httpx.
- **Credit the source.** Code and docs cite the papers and codebases they draw
  on, and the bibliography is checked by the tests.

## Setup

```console
git clone https://github.com/Antonio-Tresol/langchain-sync-monitors
cd langchain-sync-monitors
uv sync --group dev --group docs --all-extras
uvx pre-commit install
```

Live tests call real model providers. They read `OPENROUTER_API_KEY` from the
environment or from a local `.env` file, which git ignores. Run them with
`uv run --group dev pytest -m live`.

## The gates

One script runs every gate:

```console
scripts/check.sh
```

It checks that `CLAUDE.md` matches `AGENTS.md`, then runs ruff, ty (type checking),
the unit tests, the lanorme standards (naming, complexity, docstrings, prose,
docs layout), a strict docs build and a package build. The pre-commit hooks and
CI run the same checks, so a green run locally means a green pull request.

lanorme needs Python 3.13, so the script runs it with `uvx --python 3.13`; it
only parses the code, so it checks 3.12 source correctly. To see why a rule
fired, run `uvx --python 3.13 lanorme==0.21.0 rule CODE`.

## Adding a monitor or a protocol

- Implement the abstract base class from `contracts.py`: `Monitor` for a new way
  to score a step, `ControlProtocol` for a new way to act on scores.
- Implement both the synchronous and the asynchronous method.
- Keep every model a constructor parameter.
- Cite the paper or codebase the idea comes from, and add a changelog entry.

## Citing sources

Cite at the point of use with `[@key]`, in a docstring or on a docs page:

```python
class DeferToResample(ControlProtocol):
    """Defer to Resample, the basic protocol of Ctrl-Z [@bhatt2025ctrlz]."""
```

Add the entry to `docs/references.bib` in the same change. Papers use `@misc`
or `@inproceedings`, posts use `@online`, and codebases and libraries use
`@software` with the version you learned from. `tests/unit/test_references.py`
fails on a cited key missing from the bibliography and on an entry nothing
cites.

## Tests

Unit tests live in `tests/unit/` and run offline with scripted fake models.
Mark each section of a test with `# Arrange`, `# Act` and `# Assert`. Cover
both `invoke` and `ainvoke` for anything that touches the middleware. Tests
that call real providers are marked `live` and skipped by default.

## Documentation

Docs live under `docs/` in the Diataxis layout: tutorials, how-to guides,
reference and explanation. Each page opens with a line that starts "This
page", "This guide" or similar, uses British spelling, and avoids em dashes and
emoji; lanorme checks all of this. Diagrams use Mermaid. Preview the site with
`uv run --group docs mkdocs serve`.

## Sending a pull request

1. Branch off an up-to-date `main`: `feat/...` for a feature, `fix/...` for a
   bug, `docs/...` for documentation.
2. Make one focused change, with tests, docs and a changelog entry.
3. Run `scripts/check.sh` until it passes.
4. Commit with a message that describes the effect, in the imperative mood,
   and reference the issue: `Add Defer to Resample (#15)`.
5. Push and open a pull request against `main` with `Closes #N` in the
   description. CI runs the same gate across Python 3.12, 3.13 and 3.14.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence in `LICENSE`.
