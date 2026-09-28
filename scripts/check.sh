#!/usr/bin/env bash
# Run every gate: agent files in sync, ruff, mypy, the unit tests, the lanorme
# standards, the docs build, and a package build.
# Run this before committing or finishing a change. No arguments.
#
#   scripts/check.sh
#
# Exits non-zero on the first failing gate. CI and the pre-commit hooks run this
# same script, so a green run here means a green pull request.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> agent files in sync"
scripts/sync-agents.sh --check

echo "==> ruff"
uv run --group dev ruff check .
uv run --group dev ruff format --check .

echo "==> mypy (strict)"
uv run --group dev mypy

echo "==> unit tests"
uv run --group dev pytest -q

echo "==> lanorme standards"
# lanorme needs Python 3.13+; it only parses the code, so it checks 3.12 source correctly.
uvx --python 3.13 lanorme==0.21.0 check .

echo "==> docs build"
uv run --group docs mkdocs build --strict --quiet --site-dir "$(mktemp -d)"

echo "==> build"
uv build >/dev/null

echo
echo "OK: all gates passed."
