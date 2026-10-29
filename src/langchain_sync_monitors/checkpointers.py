"""The warning for a step held for a person to audit in a run without a checkpointer.

LangGraph saves a paused run in a checkpointer and resumes it from there
[@langgraph2026]. A run without one stops at the audit, so the audit can
never be answered.
"""

from __future__ import annotations

import functools
import warnings


class MissingCheckpointerWarning(UserWarning):
    """A step is held for a person to audit in a run without a checkpointer.

    The run stops at the audit with the request under `__interrupt__`, and
    cannot be resumed [@langgraph2026]. The middleware emits this warning
    once per process, at the first such audit.
    """


@functools.cache
def warn_about_missing_checkpointer() -> None:
    """Emit the `MissingCheckpointerWarning`, on the first call in the process.

    The cache on this function is what keeps the warning to one per process.
    """
    warnings.warn(
        "A step is held for a person to audit, but the agent runs without a checkpointer, "
        "so the run stops at the audit and cannot be resumed. Build the agent with "
        "create_agent(checkpointer=...) to answer audits, or set on_audit=AuditResponse.FLAG "
        "to run and flag such steps instead.",
        MissingCheckpointerWarning,
        stacklevel=2,
    )
