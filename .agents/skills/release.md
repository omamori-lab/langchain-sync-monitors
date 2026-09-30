---
name: release
description: Use when cutting, shipping, releasing or publishing a new langchain-sync-monitors version (a "0.x.y" bump, a vX.Y.Z tag, a PyPI release), or when finishing or recovering a release that failed part way. Runs every gate on the bumped tree, sets __version__ and CITATION.cff, tags a commit on main, and creates the GitHub Release whose workflow re-runs CI, checks and installs the built files, and publishes them to PyPI through trusted publishing once a maintainer approves.
license: MIT
compatibility: Requires bash, perl, git, uv and a signed-in gh with push access to omamori-lab/langchain-sync-monitors, run from a clean main checkout that matches origin/main.
metadata:
  project: langchain-sync-monitors
---

# Release langchain-sync-monitors

This page explains how to cut a release, the discipline every release follows,
and how to recover when a step fails. `scripts/release.sh` tags a commit on
`main` and creates the GitHub Release; publishing that Release starts
`.github/workflows/release.yml`, which publishes to PyPI through trusted
publishing (OIDC, no token anywhere) and then deploys the docs. Nobody runs
`uv publish` or `twine upload` by hand.

## The release discipline

Every release passes the same checks, in two places, and leaves the same
evidence behind.

`scripts/release.sh` checks, on your machine, that the approval gate exists
and that the bumped tree is ready, before it commits:

1. The `pypi` environment has a required reviewer (read with `gh api`).
   Administrators may bypass it, by the owner's choice.
2. `scripts/check-release.sh X.Y.Z`: `__version__`, `CITATION.cff` (version and
   `date-released`) and a dated, non-empty `## [X.Y.Z] - YYYY-MM-DD` section of
   `CHANGELOG.md` with its link reference agree, and `README.md` and
   `docs/index.md` hold no pre-release text: no `release-check` comment, and
   none of the phrases such text uses, such as "not on PyPI", "Pre-release" or
   a `.devN` version.
3. `scripts/check.sh`: every gate, from the agent-file sync check to the package
   build.
4. A clean `uv build` and `twine check --strict`, with the Markdown renderer, on
   the sdist and the wheel.

The Release workflow checks the tagged commit again before it publishes:

1. The GitHub Release is not marked as a pre-release, the `pypi` environment
   has a required-reviewer rule, the tag is
   `vX.Y.Z` and names a commit on `main`, and the tree passes
   `scripts/check-release.sh`.
2. The whole CI workflow: the gates on Python 3.12, 3.13 and 3.14, the offline
   suite at the lowest allowed dependency versions, and without any extra.
3. One build of the sdist and the wheel, whose names must carry the version,
   whose SHA-256 digests are recorded, and which pass `twine check --strict`.
4. The wheel, installed into fresh environments on Python 3.12, 3.13 and 3.14,
   with and without the extras, imports and reports the version.
5. A maintainer approves the `pypi` environment. The publish job then checks
   the files against the recorded digests, and only then publishes exactly
   those files. The attach job checks them again before it adds them to the
   GitHub Release.

These workflow checks catch mistakes, not a determined insider: they run from
the tagged commit's own copy of `release.yml`, which anyone with write access
can change. The controls that hold are the required reviewers on the `pypi`
environment and the `v*` tag ruleset.

The evidence a release leaves:

- the Release workflow run, whose summaries hold the release notes and the
  SHA-256 digest of each built file, and whose publish log prints the digests
  PyPI received;
- the approval in the `pypi` environment, naming who approved it;
- a PEP 740 attestation for each file on PyPI, tying it to that workflow run;
- the GitHub Release, with the CHANGELOG section as its notes and the same
  files attached.

## Versioning

The public API is what the header of `CHANGELOG.md` says: the names exported
from `langchain_sync_monitors` (monitors, control protocols, the middleware and
their constructor keywords) and the shape of the `monitor_log` records. Keep
this page and that header in step. The deciding question is whether a working
integration could fail on upgrade.

Before 1.0:

- **Minor, `0.Y.0`**: anything breaking. A removed or renamed export, a
  removed, renamed or newly required keyword, a record field removed, renamed
  or changed in type or meaning, a new exception where there was none. A changed
  default that decides which steps run, such as a threshold, the default view
  or prompt, or an Auto Mode limit, is a minor release too, with a `Changed`
  entry, because an integration that relies on the default then behaves
  differently.
- **Patch, `0.y.Z`**: nothing a working integration relies on changes. Fixes
  that make the library do what its documentation says, new names, new
  keywords whose defaults keep today's behaviour, and documentation. A security
  fix that makes the monitor fail closed where it failed open ships as a patch,
  under `Security`, so every 0.y user can take it.
- **Major, `1.0.0`**: the commitment to a stable API.

## Steps

1. **Pick the version** `X.Y.Z` by the rules above.
2. **Prepare it in a pull request** from a branch such as `release/X.Y.Z`:
   - In `CHANGELOG.md`, give the entries under `## [Unreleased]` their own
     `## [X.Y.Z] - YYYY-MM-DD` heading, dated the day you will release, and
     leave an empty `## [Unreleased]` above it. The notes of the GitHub Release
     are taken verbatim from this section.
   - Update the link references at the bottom of `CHANGELOG.md`:

     ```
     [Unreleased]: https://github.com/omamori-lab/langchain-sync-monitors/compare/vX.Y.Z...HEAD
     [X.Y.Z]: https://github.com/omamori-lab/langchain-sync-monitors/compare/vA.B.C...vX.Y.Z
     ```

     For the first release, `[0.1.0]` points at
     `https://github.com/omamori-lab/langchain-sync-monitors/releases/tag/v0.1.0`.
   - In `README.md` and `docs/index.md`, rewrite the text each
     `release-check` comment marks so it describes the released package, then
     delete the comments.
   - Leave `__version__` and `CITATION.cff` alone: the script sets them.
   - Preview with `scripts/check-release.sh X.Y.Z`. On this branch it should
     report only the `__version__` and `CITATION.cff` problems.
   - Run `scripts/check.sh`, then merge the pull request.
3. **Run the script** on an up-to-date `main`:

   ```console
   git checkout main && git pull --ff-only
   scripts/release.sh X.Y.Z
   ```

   It refuses unless you are on `main`, the tree is clean, `main` matches
   `origin/main`, the dated CHANGELOG section exists, the tag `vX.Y.Z` is new
   locally and on `origin`, and the `pypi` environment has a required reviewer.
   On GitHub's Free plan that needs a public
   repository; the API shows no reviewers on a private one. Then it sets `__version__` and `CITATION.cff`
   (version and `date-released`, from the CHANGELOG heading), runs `uv lock`,
   runs the checks listed under "The release discipline", commits
   `Release X.Y.Z`, tags it `vX.Y.Z`, pushes `main` and the tag, and creates the
   GitHub Release with `gh release create --repo omamori-lab/langchain-sync-monitors --verify-tag`.
   It deletes its temporary notes file however it ends.
4. **Approve the deployment.** Open the Release run in the Actions tab. Once
   every check passes, the `Publish to PyPI` job waits for the `pypi`
   environment; review the run and approve it.
5. **Verify** the package and the docs:

   ```console
   gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
   uv run --isolated --no-project --with langchain-sync-monitors==X.Y.Z python -c "import langchain_sync_monitors as package; print(package.__version__)"
   ```

   The docs are at https://omamori-lab.github.io/langchain-sync-monitors/.
6. **Start the next version** in a pull request that sets `__version__` to
   `X.Y.(Z+1).dev0`, so a build from `main` is never mistaken for the release.
   `CITATION.cff` keeps the released version.

## Gotchas

- The script commits the bump straight to `main`. If `main` requires pull
  requests, the maintainer running it needs permission to bypass that rule.
- The version lives in `src/langchain_sync_monitors/__init__.py`, which hatch
  reads, and in `CITATION.cff`. `uv.lock` does not record it, because the
  version is dynamic, so `uv lock` normally leaves the lock alone.
- PyPI never accepts a file name twice, even after the file is deleted, so a
  version number is spent once any file of it is uploaded. Never move or reuse
  a tag whose files reached PyPI.
- The workflow runs from the workflow file at the tagged commit. A fix to
  `release.yml` reaches a release only through a new commit on `main`.
- A GitHub Release created with the workflow token, `GITHUB_TOKEN`, starts no
  workflow. Create it with your own `gh`, as the script does.
- A Release published from the GitHub web page also starts the workflow, and
  passes the workflow's checks, the `pypi` environment check included. It skips
  everything `scripts/release.sh` does on your machine: the gates on the bumped
  tree, the build and twine check, and the version bump itself, so the
  workflow's version checks refuse it unless the tag's commit already carries
  `X.Y.Z`.
- A Release marked as a pre-release is refused, because its `vX.Y.Z` would go to
  PyPI as a final version. Unticking the mark afterwards fires `released`, not
  `published`, so it starts nothing: delete the Release with
  `gh release delete vX.Y.Z --yes`, which keeps the tag, and create it again as
  "Finish a half-done release" shows.
- The approval is the gate that holds. `scripts/release.sh` and the workflow's
  first job both refuse until the `pypi` environment has a required reviewer,
  and the `v*` tag ruleset keeps other people from tagging releases.
  Administrators may bypass the reviewer, by the owner's choice, so the
  approval holds against a leaked token or another maintainer, not against an
  administrator. The workflow's copy of that check can be edited
  away like its other checks; the environment's own rule cannot.
- The PyPI page links pages and diagrams at the tag `vX.Y.Z`, through
  hatch-fancy-pypi-readme's `$HFPR_VERSION`. They resolve once the tag is
  pushed, which happens before anything is published.
- The docs deploy from the tag, so the `github-pages` environment needs a `v*`
  tag rule; without it GitHub refuses the deployment.
- The Release workflow runs only in `omamori-lab/langchain-sync-monitors`, so
  a fork never tries to publish. If the repository is renamed or moved, update
  the `if:` of its first job, or every run is skipped: grey, not red.
- `pypa/gh-action-pypi-publish` is pinned to a commit SHA, with its version in a
  comment. Dependabot (`.github/dependabot.yml`) proposes updates to it and to
  the other actions every week; nothing else moves the pin.
- setup-uv turns its cache off on release events, so the Release run is slower
  than CI. That is on purpose: it keeps a poisoned cache out of a publishing
  run.

## If something fails

- **The script refuses in its preflight**: nothing changed. Fix what it names
  and run it again.
- **The script fails before the commit** (a check, a gate, the build or twine):
  it restores `__init__.py`, `CITATION.cff` and `uv.lock`. Fix the cause in a
  pull request, then run it again.
- **The script fails at the commit itself**, for example because a pre-commit
  hook or commit signing fails: it restores the three files from `HEAD`, which
  also unstages them. Fix the cause and run it again.
- **The script fails after the commit** (a rejected push, or `gh`): see
  "Finish a half-done release".
- **The workflow's first job finds no reviewer on `pypi`**: nothing ran after
  it. Fix the environment, then re-run the workflow.
- **The Release workflow fails before publishing** (the tag checks, CI, the
  build or the install check): nothing is on PyPI. Delete the Release and the
  tag with `gh release delete vX.Y.Z --cleanup-tag --yes` and
  `git tag -d vX.Y.Z`, fix the cause in a pull request, and run the script
  again once it merges, before anything else does. `main` already carries
  `X.Y.Z`, so the script checks and tags the new `HEAD` without a second
  release commit.
- **The publish job fails** (a PyPI outage, or a publisher setting): fix the
  setting, then re-run only the failed jobs with `gh run rerun <run-id> --failed`,
  so the files published are the ones the checks passed on.
- **PyPI holds only some of the files**: a re-run fails with "File already
  exists" for the ones it has. Yank the version on PyPI and release
  `X.Y.(Z+1)`.
- **The attach or docs job fails after publishing**: re-run the failed jobs,
  or run the Docs workflow by hand from the tag.
- **A broken release reached PyPI**: yank it on PyPI (the project's Releases
  page, Options, Yank), fix it on `main`, and release `X.Y.(Z+1)`. Keep the tag
  and the GitHub Release.

## Finish a half-done release

The script cannot resume once it has committed: a second run refuses, because
the tag exists. Finish by hand, from the step that failed.

- **The push of `main` was rejected**, for example because `main` moved or is
  protected: the commit and the tag exist only on your machine. Drop them with
  `git tag -d vX.Y.Z` and `git reset --hard origin/main`, resolve the cause, and
  run the script again.
- **`main` was pushed but the tag was not**: `git push origin vX.Y.Z`, then
  continue with the next case.
- **The tag was pushed but no GitHub Release exists**: create it with the
  CHANGELOG section as its notes:

  ```console
  notes_file="$(mktemp)"
  scripts/check-release.sh X.Y.Z > "$notes_file"
  gh release create vX.Y.Z --repo omamori-lab/langchain-sync-monitors --verify-tag --title vX.Y.Z --notes-file "$notes_file"
  rm "$notes_file"
  ```

  The notes file lives outside the repository, so the tree stays clean for a
  later run of the script.

  Publishing it starts the Release workflow; continue from step 4 of Steps.
