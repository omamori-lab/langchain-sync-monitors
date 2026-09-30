## What this changes

<!-- What the change does and why, in a few sentences. -->

Closes #

## Checklist

- [ ] `scripts/check.sh` passes: agent files in sync, ruff, ty, the unit tests,
      lanorme, the strict docs build and the package build.
- [ ] Tests cover a positive case, a negative case, the boundary and a
      regression test for the exact behaviour, under both `invoke` and
      `ainvoke`, with `# Arrange`, `# Act` and `# Assert` markers.
- [ ] Every model is a parameter, and public constructors and functions take
      keyword-only parameters after the first.
- [ ] Ideas taken from a paper, post or code base are cited with `[@key]`, with
      the entry in `docs/references.bib`.
- [ ] Docstrings and the docs pages the change affects are updated, in British
      spelling, with no em dashes.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` for anything a user
      would notice.
