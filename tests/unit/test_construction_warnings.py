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
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_sync_monitors.monitors import (
    Combine,
    DecisionModel,
    DecisionModelMonitor,
    GuardModelMonitor,
    GuardScoring,
)
from langchain_sync_monitors.protocols import AutoMode


class UnusedDecisionModel(DecisionModel):
    async def estimate_probabilities(self, *, context, questions):
        return {}

    def estimate_probabilities_sync(self, *, context, questions):
        return {}


AutoMode()
GuardModelMonitor(
    model=FakeListChatModel(responses=["violation"]),
    policy_prompt=ChatPromptTemplate.from_messages([("human", "{transcript}{proposed_step}")]),
    suspicious_labels={"violation"},
    safe_labels={"no_violation"},
    scoring=GuardScoring.HARD_LABEL,
)
DecisionModelMonitor(decision_model=UnusedDecisionModel(), combine=Combine.MEAN)
"""


def test_every_construction_warning_shows_under_the_default_filters(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "user_code.py").write_text(BUILDS_EVERY_OPTION_THAT_WARNS, encoding="utf-8")
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONWARNINGS"}

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
    assert "HardLabelWarning: GuardScoring.HARD_LABEL gives every step" in finished.stderr
    assert "UserWarning: Combine.MEAN dilutes a single strong hit" in finished.stderr
