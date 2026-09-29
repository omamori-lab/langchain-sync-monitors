#!/usr/bin/env bash
# Cut a release: bump the version, run every gate, tag, push and create the
# GitHub Release.
#
#   scripts/release.sh X.Y.Z
#
# Run it on a clean, up-to-date main, after the pull request that dates the
# "## [X.Y.Z] - YYYY-MM-DD" section of CHANGELOG.md has merged. It refuses
# unless you are on main, the tree is clean, main matches origin/main, the
# dated CHANGELOG section exists and the tag vX.Y.Z is new. Then it:
#   1. sets __version__ and CITATION.cff to X.Y.Z, dated as the CHANGELOG
#      section, and lets uv refresh uv.lock;
#   2. runs scripts/check-release.sh, syncs the environment with every extra,
#      runs scripts/check.sh on that tree, builds it and runs
#      twine check --strict on the result;
#   3. commits "Release X.Y.Z", tags it vX.Y.Z, and pushes main and the tag;
#   4. creates the GitHub Release, with that CHANGELOG section as its notes.
#
# A failure before the commit restores the files it bumped. Publishing the
# GitHub Release starts .github/workflows/release.yml, which publishes to PyPI
# through trusted publishing; this script never uploads anything to PyPI.
set -euo pipefail

version="${1:-}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/release.sh X.Y.Z" >&2
  exit 2
fi
tag="v${version}"
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

# --- bump: restore these files if anything fails before the commit ---------
bumped_files=(src/langchain_sync_monitors/__init__.py CITATION.cff uv.lock)
restore_bumped_files() {
  echo "release: restoring ${bumped_files[*]}" >&2
  git checkout --quiet -- "${bumped_files[@]}"
}
trap restore_bumped_files EXIT

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
notes_file="$(mktemp)"
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
trap - EXIT
git tag -a "${tag}" -m "${tag}"
git push origin main
git push origin "${tag}"
gh release create "${tag}" --verify-tag --title "${tag}" --notes-file "${notes_file}"

echo
echo "Released ${tag} on GitHub. The Release workflow now checks and builds it,"
echo "then waits for approval in the pypi environment. Watch it with:"
echo "  gh run watch \"\$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')\" --exit-status"
echo "Once it is published, check the package from PyPI:"
echo "  uv run --isolated --no-project --with langchain-sync-monitors==${version} python -c \"import langchain_sync_monitors as package; print(package.__version__)\""
