"""Construction warnings reach the user under Python's default warning filters.

By default Python shows a `UserWarning` wherever it is raised, but ignores a
`DeprecationWarning` raised outside `__main__`. The test builds every option
that warns from a module of its own, in a fresh interpreter with no warning
filter changed, and reads what the interpreter printed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BUILDS_EVERY_OPTION_THAT_WARNS = """
from langchain_core.caches import InMemoryCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_sync_monitors import (
    GuardLabels,
    GuardMonitor,
    LLMMonitor,
    MonitorMiddleware,
    MonitorPrompt,
    TrustedMonitoring,
)
from langchain_sync_monitors.protocols import AutoMode

AutoMode()
GuardMonitor(
    model=FakeListChatModel(responses=["violation"]),
    prompt=MonitorPrompt(system="", request="{trajectory}{proposed_action}"),
    labels=GuardLabels(suspicious=frozenset({"violation"}), safe=frozenset({"no_violation"})),
    samples=1,
)
MonitorMiddleware(
    monitor=LLMMonitor(model=FakeListChatModel(responses=["unused"], cache=InMemoryCache())),
    protocol=TrustedMonitoring(audit_threshold=0.5),
)
"""


def test_every_construction_warning_shows_under_the_default_filters(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "user_code.py").write_text(BUILDS_EVERY_OPTION_THAT_WARNS, encoding="utf-8")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONWARNINGS", "PYTHONDEVMODE"}
    }

    # Act
    finished = subprocess.run(
        [sys.executable, "-c", "import user_code"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert finished.returncode == 0, finished.stderr
    assert "UncalibratedThresholdWarning: block_threshold uses the uncalibrated" in finished.stderr
    assert "HardLabelWarning: samples=1 gives every action a suspicion of 0 or 1" in finished.stderr
    assert "CachedResampleWarning: A LangChain response cache is active for a monitor" in (
        finished.stderr
    )
