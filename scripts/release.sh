#!/usr/bin/env bash
# Cut a release: check that main carries the version, run every gate, tag, push
# the tag and create the GitHub Release.
#
#   scripts/release.sh X.Y.Z
#
# Run it on a clean, up-to-date main, after the release pull request has merged:
# the one that dates the "## [X.Y.Z] - YYYY-MM-DD" section of CHANGELOG.md and
# sets __version__ and CITATION.cff to X.Y.Z. It refuses unless you are on main,
# the tree is clean, main matches origin/main, the tag vX.Y.Z is new, and the
# pypi environment has a required reviewer.
# Then it:
#   1. runs scripts/check-release.sh, syncs the environment with every extra,
#      runs scripts/check.sh on that tree, builds it and runs
#      twine check --strict on the result;
#   2. refuses if origin/main moved meanwhile, then tags main's commit vX.Y.Z
#      and pushes the tag;
#   3. creates the GitHub Release, with that CHANGELOG section as its notes.
#
# It changes no tracked file, and deletes its temporary notes file however it
# ends. Publishing the GitHub Release starts .github/workflows/release.yml,
# which publishes to PyPI through trusted publishing; this script never uploads
# anything to PyPI.
set -euo pipefail

version="${1:-}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/release.sh X.Y.Z" >&2
  exit 2
fi
tag="v${version}"
repository="omamori-lab/langchain-sync-monitors"

cd "$(git rev-parse --show-toplevel)"

refuse() {
  echo "refusing: $1" >&2
  exit 1
}

# --- preflight: nothing changes until all of these hold --------------------
branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "${branch}" == "main" ]] || refuse "not on main (on ${branch})."
[[ -z "$(git status --porcelain)" ]] || refuse "the working tree is not clean."
git fetch --quiet --tags origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] ||
  refuse "main is not at origin/main; pull first."
if git rev-parse --quiet --verify "refs/tags/${tag}" >/dev/null; then
  refuse "the tag ${tag} already exists."
fi
if [[ -n "$(git ls-remote --tags origin "refs/tags/${tag}")" ]]; then
  refuse "the tag ${tag} already exists on origin."
fi
command -v gh >/dev/null || refuse "the GitHub CLI, gh, is not installed."
gh auth status >/dev/null 2>&1 || refuse "gh is not signed in; run gh auth login."
# The approval gate: the publish job waits for a person only if the pypi
# environment has a required reviewer. Administrators may bypass it, by the
# owner's choice.
pypi_rules='[.protection_rules[]? | select(.type == "required_reviewers") | .reviewers | length] | add // 0'
reviewer_count="$(gh api "repos/${repository}/environments/pypi" --jq "${pypi_rules}" 2>/dev/null)" ||
  refuse "could not read the pypi environment of ${repository}; create it as the one-time setup in CONTRIBUTING.md says."
[[ "${reviewer_count}" =~ ^[0-9]+$ && "${reviewer_count}" -ge 1 ]] ||
  refuse "the pypi environment has no required reviewer, so nothing would hold the publish for approval. Add one under Settings, Environments, pypi; on GitHub's Free plan that needs a public repository, and the API shows no reviewers on a private one."

notes_file="$(mktemp)"
trap 'rm -f "${notes_file}"' EXIT

# --- check main: release checks, every gate, the built files ---------------
scripts/check-release.sh "${version}" >"${notes_file}" ||
  refuse "main is not ready to release as ${version}. The release pull request dates CHANGELOG.md and sets __version__ and CITATION.cff; CONTRIBUTING.md lists its steps."
# The gates expect the environment CONTRIBUTING.md sets up, every extra included.
uv sync --locked --group dev --group docs --all-extras
scripts/check.sh
rm -rf dist
uv build
uvx --with "readme_renderer[md]" twine check --strict \
  "dist/langchain_sync_monitors-${version}-py3-none-any.whl" \
  "dist/langchain_sync_monitors-${version}.tar.gz"

# --- tag, push the tag, and create the GitHub Release -----------------------
git fetch --quiet origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] ||
  refuse "main moved on origin while the checks ran; pull and run the script again."
git tag -a "${tag}" -m "${tag}"
git push origin "${tag}"
gh release create "${tag}" --repo "${repository}" --verify-tag --title "${tag}" \
  --notes-file "${notes_file}"

echo
echo "Released ${tag} on GitHub. The Release workflow now checks and builds it,"
echo "then waits for approval in the pypi environment. Watch it with:"
echo "  gh run watch \"\$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')\" --exit-status"
echo "Once it is published, check the package from PyPI:"
echo "  uv run --isolated --no-project --with langchain-sync-monitors==${version} python -c \"import langchain_sync_monitors as package; print(package.__version__)\""
