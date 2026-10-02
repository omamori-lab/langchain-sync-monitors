#!/usr/bin/env bash
# Run every gate: agent files in sync, the gitleaks secrets scan, ruff, ty, the
# offline test suite, the lanorme standards, the docs build, a package build,
# and an import of the built wheel without any extra.
# Run this before committing or finishing a change. No arguments.
#
#   scripts/check.sh
#
# Exits non-zero on the first failing gate. CI runs this same script; CI also
# runs the offline suite at the lowest versions the dependency bounds allow, and
# without any extra. The pre-commit hooks run only the fast gates: ruff, ty,
# lanorme and the offline suite, plus a gitleaks scan of the staged changes.
#
# The secrets scan is the command CI's secrets job runs: gitleaks reads the
# history of every ref in the repository, which a git worktree shares with its
# main checkout, against .gitleaks.toml. Without gitleaks on PATH, the script
# says so and skips the scan, as in CI's gates job, which has no gitleaks; CI's
# secrets job still scans every pull request.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> agent files in sync"
scripts/sync-agents.sh --check

echo "==> secrets (gitleaks)"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks git . --config .gitleaks.toml --redact --no-banner
else
  echo "skipped: gitleaks is not on PATH; CI's secrets job runs this scan."
fi

echo "==> ruff"
uv run --group dev ruff check .
uv run --group dev ruff format --check .

echo "==> ty (type checking)"
uv run --group dev ty check

echo "==> offline test suite"
# The docs standard tests render pages with the docs group. They skip without it,
# as in the dev-only CI jobs; here REQUIRE_DOCS_GROUP=1 makes a missing module fail.
REQUIRE_DOCS_GROUP=1 uv run --group dev --group docs pytest -q

echo "==> lanorme standards"
# lanorme needs Python 3.13+; it only parses the code, so it checks 3.12 source correctly.
uvx --python 3.13 lanorme==0.21.0 check .

echo "==> docs build"
uv run --group docs mkdocs build --strict --site-dir "$(mktemp -d)"

echo "==> build"
uv build >/dev/null

echo "==> import the built wheel without any extra"
wheel="$(ls -t dist/langchain_sync_monitors-*.whl | head -n 1)"
uv run --isolated --no-project --python 3.12 --with "$wheel" \
  python -c "import langchain_sync_monitors as package; print(len(package.__all__), 'public names')"

echo
echo "OK: all gates passed."
