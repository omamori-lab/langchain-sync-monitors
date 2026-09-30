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

- **Every model is a parameter.** The library never picks a model for you.
- **Core dependencies stay small.** Providers sit behind extras.
- **Types say what the data is.** No `Any` where the shape is known.
- **Canonical libraries over hand-rolled logic.** Retries, validation and HTTP
  use stamina, pydantic and httpx.
- **Credit the source.** Code and docs cite the papers and codebases they draw
  on, and the bibliography is checked by the tests.

## Setup

```console
git clone https://github.com/omamori-lab/langchain-sync-monitors
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
CI run the same checks. CI also runs the offline suite at the lowest versions
the dependency bounds in `pyproject.toml` allow, and again without any extra.

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
   as required reviewers, untick "Allow administrators to bypass configured
   protection rules", and limit its deployment branches and tags to the tag
   pattern `v*`. `scripts/release.sh` and the Release workflow both refuse
   until the environment has a required reviewer whom administrators cannot
   bypass.
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

The required reviewers and the tag ruleset are the controls that hold. The
Release workflow's own checks catch mistakes, but they run from the tagged
commit's copy of the workflow, which anyone with write access can change.

### Release steps

1. **Date the changelog in a pull request.** Move the `## [Unreleased]` entries
   under a new `## [X.Y.Z] - YYYY-MM-DD` heading, leave an empty
   `## [Unreleased]` above it, and update the link references at the bottom of
   `CHANGELOG.md`. In `README.md` and `docs/index.md`, rewrite the text each
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
- refuses unless the `pypi` environment has a required reviewer whom
  administrators cannot bypass, so a Release published from the web page,
  which skips `scripts/release.sh`, still waits for approval;
- checks that the tag is `vX.Y.Z`, names a commit on `main`, and agrees with
  `__version__`, `CITATION.cff` and a dated CHANGELOG section, and that
  `README.md` and `docs/index.md` hold no pre-release text
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
