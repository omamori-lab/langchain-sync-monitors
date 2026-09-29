#!/usr/bin/env bash
# Check that the tree is ready to be released as version X.Y.Z, and print the
# release notes. It changes nothing.
#
#   scripts/check-release.sh X.Y.Z
#
# scripts/release.sh runs it on the bumped tree before it commits, and the
# Release workflow runs it on the tagged commit before anything is built. It
# checks that:
#   - X.Y.Z is a final version, not a development or pre-release one;
#   - __version__ in src/langchain_sync_monitors/__init__.py is X.Y.Z;
#   - CHANGELOG.md has a non-empty "## [X.Y.Z] - YYYY-MM-DD" section and an
#     "[X.Y.Z]: " link reference;
#   - CITATION.cff has version X.Y.Z and date-released equal to that date;
#   - README.md and docs/index.md hold no pre-release text: "not on PyPI", or
#     a development version such as 0.1.0.dev0.
#
# It reports every problem it finds and exits 1 if there was any. Otherwise it
# prints the CHANGELOG section, without its heading, to standard output: the
# notes of the GitHub Release.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

version="${1:-}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/check-release.sh X.Y.Z" >&2
  exit 2
fi
escaped_version="${version//./\\.}"

problems=0
report() {
  # On GitHub Actions each problem becomes an error annotation on the run.
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::error::$1" >&2
  else
    echo "check-release: $1" >&2
  fi
  problems=$((problems + 1))
}

package_version="$(sed -n -E 's/^__version__ = "([^"]*)"$/\1/p' src/langchain_sync_monitors/__init__.py)"
if [[ "${package_version}" != "${version}" ]]; then
  report "__version__ is '${package_version}', not ${version}."
fi

release_date="$(sed -n -E "s/^## \[${escaped_version}\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})$/\1/p" CHANGELOG.md | head -n 1)"
notes=""
if [[ -z "${release_date}" ]]; then
  report "CHANGELOG.md has no '## [${version}] - YYYY-MM-DD' section."
else
  # The section runs to the next "## [" heading. Link references, which Keep a
  # Changelog puts at the end of the file, and outer blank lines are dropped.
  # The awk sticks to what gawk, mawk and BSD awk all read the same way.
  notes="$(awk -v heading="## [${version}] - " '
    index($0, heading) == 1 { grab = 1; next }
    grab && /^## \[/ { exit }
    grab && /^\[.+\]: http/ { next }
    grab { lines[++count] = $0 }
    END {
      first = 1
      while (first <= count && lines[first] ~ /^[ \t]*$/) first++
      last = count
      while (last >= first && lines[last] ~ /^[ \t]*$/) last--
      for (line = first; line <= last; line++) print lines[line]
    }
  ' CHANGELOG.md)"
  if [[ -z "${notes}" ]]; then
    report "the CHANGELOG.md section for ${version} is empty."
  fi
fi
if ! grep -q -E "^\[${escaped_version}\]: https://" CHANGELOG.md; then
  report "CHANGELOG.md has no '[${version}]: https://...' link reference."
fi

citation_version="$(sed -n -E 's/^version: "?([^"]*)"?$/\1/p' CITATION.cff)"
if [[ "${citation_version}" != "${version}" ]]; then
  report "CITATION.cff has version '${citation_version}', not ${version}."
fi
citation_date="$(sed -n -E 's/^date-released: "?([^"]*)"?$/\1/p' CITATION.cff)"
if [[ -n "${release_date}" && "${citation_date}" != "${release_date}" ]]; then
  report "CITATION.cff has date-released '${citation_date}', not ${release_date} as in CHANGELOG.md."
fi

pre_release_text="$(grep -n -i -E 'not on PyPI|[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+' README.md docs/index.md || true)"
if [[ -n "${pre_release_text}" ]]; then
  while IFS= read -r line; do
    report "pre-release text left in ${line}"
  done <<<"${pre_release_text}"
fi

if [[ "${problems}" -gt 0 ]]; then
  echo "check-release: ${problems} problem(s); ${version} is not ready to release." >&2
  exit 1
fi
printf '%s\n' "${notes}"
