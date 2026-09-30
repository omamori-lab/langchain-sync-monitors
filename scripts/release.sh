#!/usr/bin/env bash
# Cut a release: bump the version, run every gate, tag, push and create the
# GitHub Release.
#
#   scripts/release.sh X.Y.Z
#
# Run it on a clean, up-to-date main, after the pull request that dates the
# "## [X.Y.Z] - YYYY-MM-DD" section of CHANGELOG.md has merged. It refuses
# unless you are on main, the tree is clean, main matches origin/main, the
# dated CHANGELOG section exists, the tag vX.Y.Z is new, and the pypi
# environment has a required reviewer that administrators cannot bypass.
# Then it:
#   1. sets __version__ and CITATION.cff to X.Y.Z, dated as the CHANGELOG
#      section, and lets uv refresh uv.lock;
#   2. runs scripts/check-release.sh, syncs the environment with every extra,
#      runs scripts/check.sh on that tree, builds it and runs
#      twine check --strict on the result;
#   3. commits "Release X.Y.Z", tags it vX.Y.Z, and pushes main and the tag;
#   4. creates the GitHub Release, with that CHANGELOG section as its notes.
#
# A failure before the commit restores the files it bumped, and the script
# deletes its temporary notes file however it ends. Publishing the
# GitHub Release starts .github/workflows/release.yml, which publishes to PyPI
# through trusted publishing; this script never uploads anything to PyPI.
set -euo pipefail

version="${1:-}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/release.sh X.Y.Z" >&2
  exit 2
fi
tag="v${version}"
repository="omamori-lab/langchain-sync-monitors"
escaped_version="${version//./\\.}"

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
  refuse "main is not at origin/main; pull or push first."
if git rev-parse --quiet --verify "refs/tags/${tag}" >/dev/null; then
  refuse "the tag ${tag} already exists."
fi
if [[ -n "$(git ls-remote --tags origin "refs/tags/${tag}")" ]]; then
  refuse "the tag ${tag} already exists on origin."
fi
release_date="$(sed -n -E "s/^## \[${escaped_version}\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})$/\1/p" CHANGELOG.md | head -n 1)"
[[ -n "${release_date}" ]] ||
  refuse "CHANGELOG.md has no '## [${version}] - YYYY-MM-DD' section. Date it in a pull request first."
command -v gh >/dev/null || refuse "the GitHub CLI, gh, is not installed."
gh auth status >/dev/null 2>&1 || refuse "gh is not signed in; run gh auth login."
# The approval gate: the publish job waits for a person only if the pypi
# environment has a required reviewer whom administrators cannot bypass.
pypi_rules='[([.protection_rules[]? | select(.type == "required_reviewers") | .reviewers | length] | add // 0), .can_admins_bypass] | @tsv'
pypi_protection="$(gh api "repos/${repository}/environments/pypi" --jq "${pypi_rules}" 2>/dev/null)" ||
  refuse "could not read the pypi environment of ${repository}; create it as the one-time setup in CONTRIBUTING.md says."
read -r reviewer_count admins_can_bypass <<<"${pypi_protection}"
[[ "${reviewer_count}" =~ ^[0-9]+$ && "${reviewer_count}" -ge 1 ]] ||
  refuse "the pypi environment has no required reviewer, so nothing would hold the publish for approval. Add one under Settings, Environments, pypi; on GitHub's Free plan that needs a public repository, and the API shows no reviewers on a private one."
[[ "${admins_can_bypass}" == "false" ]] ||
  refuse "administrators can bypass the pypi environment's reviewers. Untick \"Allow administrators to bypass configured protection rules\" under Settings, Environments, pypi."

# --- bump: restore these files if anything fails before the commit ---------
bumped_files=(src/langchain_sync_monitors/__init__.py CITATION.cff uv.lock)
notes_file="$(mktemp)"
restore_on_exit=yes
finish() {
  if [[ "${restore_on_exit}" == yes ]]; then
    echo "release: restoring ${bumped_files[*]}" >&2
    # From HEAD, not the index: a failed commit leaves the bump staged.
    git checkout --quiet HEAD -- "${bumped_files[@]}"
  fi
  rm -f "${notes_file}"
}
trap finish EXIT

export RELEASE_VERSION="${version}" RELEASE_DATE="${release_date}"
perl -i -pe 's/^__version__ = .*/__version__ = "$ENV{RELEASE_VERSION}"/' \
  src/langchain_sync_monitors/__init__.py
perl -i -pe 's/^version:.*/version: "$ENV{RELEASE_VERSION}"/' CITATION.cff
if grep -q '^date-released:' CITATION.cff; then
  perl -i -pe 's/^date-released:.*/date-released: "$ENV{RELEASE_DATE}"/' CITATION.cff
else
  perl -i -pe '$_ .= qq{date-released: "$ENV{RELEASE_DATE}"\n} if /^version:/' CITATION.cff
fi
uv lock
echo "release: bumped to ${version}, dated ${release_date}"

# --- check the bumped tree: release checks, every gate, the built files ----
scripts/check-release.sh "${version}" >"${notes_file}"
# The gates expect the environment CONTRIBUTING.md sets up, every extra included.
uv sync --locked --group dev --group docs --all-extras
scripts/check.sh
rm -rf dist
uv build
uvx --with "readme_renderer[md]" twine check --strict \
  "dist/langchain_sync_monitors-${version}-py3-none-any.whl" \
  "dist/langchain_sync_monitors-${version}.tar.gz"

# --- commit, tag, push, and create the GitHub Release ------------------------
git add -- "${bumped_files[@]}"
if git diff --cached --quiet; then
  echo "release: HEAD already carries ${version}; tagging it as it is."
else
  git commit --quiet -m "Release ${version}"
fi
restore_on_exit=no
git tag -a "${tag}" -m "${tag}"
git push origin main
git push origin "${tag}"
gh release create "${tag}" --repo "${repository}" --verify-tag --title "${tag}" \
  --notes-file "${notes_file}"

echo
echo "Released ${tag} on GitHub. The Release workflow now checks and builds it,"
echo "then waits for approval in the pypi environment. Watch it with:"
echo "  gh run watch \"\$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')\" --exit-status"
echo "Once it is published, check the package from PyPI:"
echo "  uv run --isolated --no-project --with langchain-sync-monitors==${version} python -c \"import langchain_sync_monitors as package; print(package.__version__)\""
