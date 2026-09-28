"""Fixtures for the Deep Agents tests, which are skipped without the `deepagents` extra."""

import importlib.util

from tests.support.fixtures import run_mode as run_mode

collect_ignore_glob = [] if importlib.util.find_spec("deepagents") else ["test_*.py"]
