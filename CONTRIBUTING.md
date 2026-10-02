# Contributing to langchain-sync-monitors

Thank you for helping. This guide covers how the project is set up, the gates a
change must pass, how to send a pull request and how to cut a release.

## Where to start

- Read [How the library is built](docs/explanation/design.md) for the design:
  what happens on one step, where the monitor sits and how the pieces share
  the work.
- `docs/plans/` holds the plan written before each feature and the research
  behind its decisions. The plans are historical records of what was decided
  at the time; the code and the rest of `docs/` describe what the library does
  now.
- Pick an open issue. Issues are grouped into milestones by phase and labelled
  by area (`monitors`, `protocols`, `middleware`, `deepagents`, `docs`).
- For anything larger than a fix, comment on the issue first so the design can
  be agreed before the code.

## Principles

- **Simple to use, open to configure.** A monitored agent takes one line;
  every behaviour is an option, with a default where one is safe; anything
  beyond the options is your own implementation of a contract in
  `contracts.py`. `AGENTS.md` ("How the code is designed") lists the design
  patterns the code keeps to.
- **Every model is a parameter.** The library never picks a model for you.
- **Core dependencies stay small.** Providers sit behind extras.
- **Types say what the data is.** No `Any` where the shape is known.
- **Canonical libraries over hand-rolled logic.** Retries, validation and HTTP
  use stamina, pydantic and httpx.
- **Standard words.** Names use the terms software engineering, machine
  learning, technical AI safety and AI control already have, in full words.
  A new term is coined only when none exists.
- **Protocols rest on the literature.** A protocol follows AI control or
  technical AI safety research, or a published control evaluation or
  benchmark. A behaviour no source supports says so.
- **Credit the source.** Code and docs cite the papers and codebases they draw
  on, and the bibliography is checked by the tests.

## Setup

```console
git clone https://github.com/omamori-lab/langchain-sync-monitors
cd langchain-sync-monitors
uv sync --group dev --group docs --all-extras
uvx pre-commit install
```

The pre-commit hooks include a secrets scan, which needs
[gitleaks](https://github.com/gitleaks/gitleaks) on your `PATH`, for example
from `brew install gitleaks`. CI scans the whole history with the same
`.gitleaks.toml`.

Live tests call real model providers. They read `OPENROUTER_API_KEY` from the
environment only, and are skipped without it. Export the key, then run them
with `uv run --group dev pytest -m live`.

## The gates

One script runs every gate:

```console
scripts/check.sh
```

It checks that `CLAUDE.md` matches `AGENTS.md`, then runs ruff, ty (type checking),
the offline test suite, the lanorme standards (naming, complexity, docstrings,
prose, docs layout), a strict docs build and a package build, and imports the
built wheel without any extra. CI runs the same script. CI also runs the
offline suite at the lowest versions the dependency bounds in `pyproject.toml`
allow, and again without any extra.

The pre-commit hooks run the fast gates only: ruff, ty, lanorme, the offline
test suite, the gitleaks secrets scan and a few file checks. They skip the
agent-file sync check, the docs build, the package build and the wheel import,
and ruff, ty and the test suite run only when a commit stages a Python file,
so a docs-only commit skips the docs-standard tests. Run `scripts/check.sh`
before you push.

lanorme needs Python 3.13, so the script runs it with `uvx --python 3.13`; it
only parses the code, so it checks 3.12 source correctly. To see why a rule
fired, run `uvx --python 3.13 lanorme==0.21.0 rule CODE`. Its limits in
`pyproject.toml` are a ratchet: meet them by naming steps or sharing a helper,
not by raising a limit, and keep inline ignores within the `max_total` budget
there.

## Adding a monitor or a protocol

- A new way to score a step implements `Monitor` from `contracts.py`, with both
  `evaluate` and `evaluate_sync`.
- A new way to act on scores implements `ControlProtocol`: one `decide`
  coroutine that awaits nothing that needs an event loop: the `PendingStep`'s
  methods and coroutines built only on them, such as the built-in fallbacks'
  `take_over` or a helper of your own. The same copy then runs under both
  `invoke()` and `ainvoke()`. Under `invoke()`, awaiting work that needs an
  event loop raises `SynchronousRunError`.
- A new decision model implements `DecisionModel`, with both
  `estimate_probabilities` and `estimate_probabilities_sync`; a new fallback
  implements `Fallback.take_over`, under the same rule as `decide`.
- Keep no run state on the instance: one copy serves parallel subagent runs.
- Keep every model a constructor parameter.
- Cite the research, evaluation or codebase it rests on (see "Citing
  sources"), and add a changelog entry.

## Citing sources

Cite at the point of use in pandoc's citation syntax, `[@key]` or
`[@first; @second]`, straight after the claim, in the docstring of the code
that applies the source or in the docs text:

```python
class DeferToResample(ControlProtocol):
    """Replace a suspicious step with a fresh sample of it, or hand it to a fallback.

    This is the basic Defer to Resample protocol of Ctrl-Z [@bhatt2025ctrlz].
    """
```

When a claim takes a specific number, result or definition from a source,
give a locator (a section, appendix, table, figure, equation or footnote) in
the prose, in full lowercase words, as in `(section 3.2, footnote 6)
[@bhatt2025ctrlz]`. Never write `§`, `Sec.`, `App.`, `Eq.` or `Tab.`, and
never put the locator inside the brackets. On a docs page the strict build
fails on one there; in a docstring nothing checks it, so reviewers do.

Add the entry to `docs/references.bib` in the same change, under the `%`
heading for its topic. A key is the first author's surname or the
organisation, the year and a short title stem (`bhatt2025ctrlz`). A codebase
is keyed by its name and year (`langgraph2026`), or by author, year and name
when it is better known that way (`schlawack2026stamina`). Papers use `@misc`,
`@inproceedings` or `@article`, reports `@techreport`, books `@book`, posts
`@online`, and codebases and libraries `@software` with the version you
learned from. `tests/unit/test_references.py`
fails on a cited key missing from the bibliography and on an entry nothing
cites.

## Tests

The offline suite runs with scripted fake models: unit tests in `tests/unit/`,
whole-agent tests in `tests/integration/`, and the live harness's offline
checks in `tests/live/test_harness_offline.py` and
`tests/live/test_eval_checks_offline.py`. Mark each section of a test
with `# Arrange`, `# Act` and `# Assert`. Cover both `invoke` and `ainvoke` for
anything that touches the middleware. Tests that call real providers are
marked `live` and skipped by default.

## Documentation

Docs live under `docs/` in the Diataxis layout: tutorials, how-to guides,
reference and explanation. Each page opens with a line that starts "This
page", "This guide" or similar, uses British spelling, and avoids em dashes and
emoji; lanorme checks all of this. Do not add Mermaid: describe the figure you
need in your pull request, and a maintainer draws it as a brand figure.
Preview the site with `uv run --group docs mkdocs serve`.

## Sending a pull request

1. Branch off an up-to-date `main`: `feat/...` for a feature, `fix/...` for a
   bug, `docs/...` for documentation.
2. Make one focused change, with tests, docs and a changelog entry.
3. Run `scripts/check.sh` until it passes.
4. Commit with a message that describes the effect, in the imperative mood,
   and reference the issue: `Add Defer to Resample (#15)`.
5. Push and open a pull request against `main` with `Closes #N` in the
   description, and work through the template's checklist. CI runs the same
   gate across Python 3.12, 3.13 and 3.14.

## Releasing

A release is a commit on `main` tagged `vX.Y.Z`, with a GitHub Release for
that tag. `scripts/release.sh` makes both; publishing the GitHub Release starts
the Release workflow, which checks the tagged commit, publishes it to PyPI and
deploys the documentation. The release skill,
`.claude/skills/release/SKILL.md`, adds the versioning rules, the gotchas and
how to finish a half-done release.

### One-time setup

A repository administrator does this once, before the first release:

1. Make the repository public. On GitHub's Free plan, an environment can only
   require a reviewer in a public repository.
2. Check the pending trusted publisher on PyPI: owner `omamori-lab`, repository
   `langchain-sync-monitors`, workflow `release.yml`, environment `pypi`. The
   first publish makes it the project's publisher.
3. Create the `pypi` environment (Settings, Environments). Add the maintainers
   as required reviewers, and limit its deployment branches and tags to the
   tag pattern `v*`. `scripts/release.sh` and the Release workflow both refuse
   until the environment has a required reviewer. Administrators may bypass
   it, by the owner's choice: leave "Allow administrators to bypass configured
   protection rules" ticked.
4. Enable GitHub Pages with "GitHub Actions" as the source (Settings, Pages).
   That creates the `github-pages` environment. Limit its deployment branches
   and tags to `main` and the tag pattern `v*`: the docs deploy from the
   release tag, so a rule that allows only `main` refuses them.
5. Enable private vulnerability reporting in the repository's security
   settings; `SECURITY.md` relies on it.
6. Add a tag ruleset for `v*` that lets only maintainers create, move or
   delete release tags. If `main` requires pull requests, let the maintainer
   who releases bypass that rule, because the script pushes the release commit
   to `main`.

The required reviewers and the tag ruleset are the controls that hold. An
administrator can bypass the reviewer, and the ruleset too when its bypass list
includes them. The
Release workflow's own checks catch mistakes, but they run from the tagged
commit's copy of the workflow, which anyone with write access can change.

### Release steps

1. **Date the changelog in a pull request.** Move the `## [Unreleased]` entries
   under a new `## [X.Y.Z] - YYYY-MM-DD` heading, leave an empty
   `## [Unreleased]` above it, and update the link references at the bottom of
   `CHANGELOG.md`. In `README.md`, `docs/index.md` and
   `docs/tutorials/first-monitored-agent.md`, rewrite the text each
   `release-check` comment marks so it describes the released package, and
   delete the comments. `scripts/check-release.sh X.Y.Z` previews the release
   checks; before the bump it should report only `__version__` and
   `CITATION.cff`. Merge the pull request.
2. **Run the release script** on a clean `main` that matches `origin/main`:
   `scripts/release.sh X.Y.Z`. It sets `__version__` and `CITATION.cff`, runs
   the release checks, every gate and `twine check --strict`, commits
   `Release X.Y.Z`, tags it `vX.Y.Z`, pushes `main` and the tag, and creates
   the GitHub Release with the CHANGELOG section as its notes.
3. **Approve the deployment.** The workflow's `Publish to PyPI` job waits for
   the `pypi` environment; approve it once every check before it has passed.
4. **Check the result**: install `langchain-sync-monitors==X.Y.Z` from PyPI in
   a fresh environment, and open the docs site.
5. **Start the next version** in a pull request that sets `__version__` to the
   next development version, such as `0.1.1.dev0`. `CITATION.cff` keeps the
   released version.

### What the Release workflow checks

Before it publishes anything, the workflow:

- refuses a GitHub Release marked as a pre-release, which would otherwise go
  to PyPI as a final version;
- refuses unless the `pypi` environment has a required reviewer, so a
  Release published from the web page, which skips `scripts/release.sh`,
  still waits for approval;
- checks that the tag is `vX.Y.Z`, names a commit on `main`, and agrees with
  `__version__`, `CITATION.cff` and a dated CHANGELOG section, and that
  `README.md`, `docs/index.md` and the tutorial hold no pre-release text
  (`scripts/check-release.sh`);
- runs the whole CI workflow on the tagged commit: every gate on Python 3.12,
  3.13 and 3.14, the offline suite at the lowest allowed dependency versions,
  and the offline suite without any extra;
- builds the sdist and the wheel once, records their SHA-256 digests, checks
  that their names carry the version, and runs `twine check --strict` on them;
- installs the wheel into fresh environments on Python 3.12, 3.13 and 3.14,
  with and without the extras, and imports it.

Once approved, it checks the files against the recorded digests, publishes
exactly those files to PyPI through trusted publishing, checks them again and
attaches them to the GitHub Release, and deploys the tagged commit's
documentation to GitHub Pages. A Release published from the web page skips
what `scripts/release.sh` runs on the maintainer's machine, so release with
the script.

### Recovering from a failed release

- **A failure before publishing** leaves PyPI untouched. Delete the GitHub
  Release and its tag with `gh release delete vX.Y.Z --cleanup-tag --yes`, and
  the local tag with `git tag -d vX.Y.Z`. Fix the cause in a pull request, and
  run `scripts/release.sh X.Y.Z` again.
- **A failed publish**, such as a PyPI outage or a wrong setting: fix the cause
  and re-run only the failed jobs with `gh run rerun <run-id> --failed`, so the
  files published are the ones that were checked.
- **A failed attach or docs job** after publishing: re-run the failed jobs, or
  run the Docs workflow by hand from the release tag.
- **A broken release on PyPI** cannot be replaced, and its version number
  cannot be used again. Yank it on PyPI and release the fix as `X.Y.(Z+1)`.
- **The release script stopped after its commit**: finish by hand, as the
  release skill describes.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence in `LICENSE`.
