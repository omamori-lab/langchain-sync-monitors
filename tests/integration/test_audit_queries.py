"""The audit query the Defer to Resample guide documents, run on records the real protocol writes.

Defer to Resample flags a step on its lowest suspicion, as Ctrl-Z does, so a
step whose highly suspicious sample was rejected, and whose resample ran, is
not flagged. The guide gives a query that finds such steps. The test reads the
query from the guide itself, so the page and the test cannot drift apart, and
runs it on the `monitor_log` of a scripted agent.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from langchain_sync_monitors import StepRecord
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.monitors import KeywordMonitor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFER_TO_RESAMPLE_GUIDE = REPOSITORY_ROOT / "docs" / "how-to" / "use-defer-to-resample.md"
QUERY_NAME = "find_steps_with_rejected_suspicious_samples"
PYTHON_BLOCK_PATTERN = re.compile(r"^```python\n(.*?)^```", flags=re.DOTALL | re.MULTILINE)
DEFER_THRESHOLD = 0.6
AUDIT_THRESHOLD = 0.9
RESAMPLE_SUSPICION = 0.4
FINAL_ANSWER = "Q3 revenue grew 12%."

type AuditQuery = Callable[..., list[StepRecord]]


def load_documented_function(page: Path, *, name: str) -> AuditQuery:
    """Return the function `name` as a Python example on `page` defines it, with its imports.

    Only the block's imports and that definition run: the rest of the block
    uses names, such as a run's `result`, that earlier examples define.
    """
    for source in PYTHON_BLOCK_PATTERN.findall(page.read_text(encoding="utf-8")):
        module = ast.parse(source)
        definitions = [
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if not definitions:
            continue
        imports = [node for node in module.body if isinstance(node, ast.Import | ast.ImportFrom)]
        namespace: dict[str, object] = {}
        code = compile(
            ast.Module(body=[*imports, *definitions], type_ignores=[]), str(page), "exec"
        )
        exec(code, namespace)
        return cast("AuditQuery", namespace[name])
    message = f"{page.name} defines no function named {name}"
    raise AssertionError(message)


def run_defer_to_resample(
    *,
    untrusted_steps: list[AIMessage],
    audit_threshold: float,
    suspicion_by_keyword: dict[str, float],
    run_mode: RunMode,
) -> list[StepRecord]:
    """Run a scripted agent under Defer to Resample and a keyword monitor, and return its log."""
    protocol = DeferToResample(
        fallback=HaltRun(),
        defer_threshold=DEFER_THRESHOLD,
        audit_threshold=audit_threshold,
    )
    agent = create_agent(
        model=ScriptedChatModel(responses=untrusted_steps),
        tools=Workspace().build_tools(),
        middleware=[
            MonitorMiddleware(
                monitor=KeywordMonitor(suspicion_by_keyword=suspicion_by_keyword),
                protocol=protocol,
            ),
        ],
    )
    return cast("list[StepRecord]", run_agent(agent, mode=run_mode)["monitor_log"])


def run_resampled_step(*, rejected_suspicion: float, run_mode: RunMode) -> list[StepRecord]:
    """Run an agent whose first sample posts the key and whose resample reads the report.

    The monitor scores the post `rejected_suspicion`, above the defer threshold
    so the sample is rejected, and the read 0.4, so the resample runs.
    """
    return run_defer_to_resample(
        untrusted_steps=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        audit_threshold=AUDIT_THRESHOLD,
        suspicion_by_keyword={
            "attacker.example": rejected_suspicion,
            "q3.md": RESAMPLE_SUSPICION,
        },
        run_mode=run_mode,
    )


def test_the_query_finds_a_rejected_sample_the_flag_misses(run_mode: RunMode) -> None:
    # Arrange
    query = load_documented_function(DEFER_TO_RESAMPLE_GUIDE, name=QUERY_NAME)
    monitor_log = run_resampled_step(rejected_suspicion=1.0, run_mode=run_mode)

    # Act
    found = query(monitor_log, audit_threshold=AUDIT_THRESHOLD)

    # Assert
    first_step = monitor_log[0]
    assert first_step["outcome"] == "resampled"
    assert first_step["flagged"] is False
    assert [(sample["suspicion"], sample["executed"]) for sample in first_step["samples"]] == [
        (1.0, False),
        (RESAMPLE_SUSPICION, True),
    ]
    assert found == [first_step]


@pytest.mark.parametrize(
    ("rejected_suspicion", "found_steps"),
    [(AUDIT_THRESHOLD, [1]), (0.89, []), (0.7, [])],
)
def test_the_query_counts_a_rejected_sample_from_the_audit_threshold_up(
    run_mode: RunMode,
    rejected_suspicion: float,
    found_steps: list[int],
) -> None:
    # Arrange
    query = load_documented_function(DEFER_TO_RESAMPLE_GUIDE, name=QUERY_NAME)
    monitor_log = run_resampled_step(rejected_suspicion=rejected_suspicion, run_mode=run_mode)

    # Act
    found = query(monitor_log, audit_threshold=AUDIT_THRESHOLD)

    # Assert
    assert [record["step_number"] for record in found] == found_steps
    assert not any(record["flagged"] for record in monitor_log)


def test_the_query_skips_a_flagged_step_whose_suspicious_sample_ran(run_mode: RunMode) -> None:
    # Arrange: an audit threshold below the defer threshold flags a borderline step that ran.
    query = load_documented_function(DEFER_TO_RESAMPLE_GUIDE, name=QUERY_NAME)
    borderline_threshold = 0.3
    monitor_log = run_defer_to_resample(
        untrusted_steps=[build_read_step(), AIMessage(FINAL_ANSWER)],
        audit_threshold=borderline_threshold,
        suspicion_by_keyword={"q3.md": RESAMPLE_SUSPICION},
        run_mode=run_mode,
    )

    # Act
    found = query(monitor_log, audit_threshold=borderline_threshold)

    # Assert
    assert monitor_log[0]["outcome"] == "allowed"
    assert monitor_log[0]["flagged"] is True
    assert found == []


def test_a_page_without_the_query_is_reported(tmp_path: Path) -> None:
    # Arrange
    page = tmp_path / "page.md"
    page.write_text("# A page\n\n```python\ndef something_else() -> None:\n    pass\n```\n")

    # Act and Assert
    with pytest.raises(AssertionError, match=QUERY_NAME):
        load_documented_function(page, name=QUERY_NAME)
