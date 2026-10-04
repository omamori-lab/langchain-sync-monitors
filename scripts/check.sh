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
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> agent files in sync"
scripts/sync-agents.sh --check

echo "==> secrets (gitleaks)"
if command -v gitleaks >/dev/null 2>&1; then
  # --verbose names the rule, file and commit of a finding; --redact hides the secret.
  gitleaks_options=(--config .gitleaks.toml --redact --verbose --no-banner)
  # The history of the checked-out commit.
  gitleaks git . --log-opts=HEAD "${gitleaks_options[@]}"
  # The staged changes, then the unstaged ones.
  gitleaks git . --pre-commit --staged "${gitleaks_options[@]}"
  gitleaks git . --pre-commit "${gitleaks_options[@]}"
  # Each untracked file git does not ignore: gitleaks dir takes one path, and
  # ./ keeps a name that starts with a dash from reading as an option.
  git ls-files --others --exclude-standard -z |
    while IFS= read -r -d '' path; do
      gitleaks dir "./$path" "${gitleaks_options[@]}"
    done
else
  echo "skipped: gitleaks is not on PATH; CI's secrets job scans the whole history."
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
# A fresh environment: `uv run --with` reuses a cached install of a wheel whose
# name and version have not changed, and would import an older build.
wheel_env="$(mktemp -d)"
uv venv --quiet --python 3.12 "$wheel_env"
uv pip install --quiet --python "$wheel_env/bin/python" "$wheel"
"$wheel_env/bin/python" -c "import langchain_sync_monitors as package; print(len(package.__all__), 'public names')"
rm -rf "$wheel_env"

echo
echo "OK: all gates passed."
