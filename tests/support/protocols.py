"""Protocol settings the agent-level tests share, as the library's configurations."""

from __future__ import annotations

from langchain_sync_monitors import (
    AuditResponse,
    BlockAndFeedBack,
    DeferToResample,
    FeedbackTemplate,
)

SUSPICION_THRESHOLD = 0.5
FEEDBACK_PREFIX = "[Safety monitor] Blocked: "
FEEDBACK = FeedbackTemplate(text=FEEDBACK_PREFIX + "{reason}")
"""Feedback that is the prefix and the monitor's reason, so a test can read the reason back."""

RUN_EVERY_STEP = DeferToResample(
    defer_threshold=1.0,
    audit_threshold=1.0,
    max_resamples=0,
    on_audit=AuditResponse.FLAG,
)
"""Trusted monitoring at 1.0: every step runs, and one rated 1.0 is flagged."""

RESAMPLE = DeferToResample(
    defer_threshold=SUSPICION_THRESHOLD,
    audit_threshold=1.0,
    max_resamples=2,
)
"""Resamples a suspicious step up to twice; the trusted model, or an audit, takes the rest."""

DEFER = DeferToResample(
    defer_threshold=SUSPICION_THRESHOLD,
    audit_threshold=1.0,
    max_resamples=0,
)
"""Hands a suspicious step to the trusted model, or, without one, audits it."""

HALT_EVERY_STEP = DeferToResample(defer_threshold=0.0, audit_threshold=0.0, max_resamples=0)
"""Audits every step, so each halts while no person can answer."""

BLOCK = BlockAndFeedBack(block_threshold=SUSPICION_THRESHOLD)
"""Blocks a suspicious step and feeds the reason back; three in a row audit the next."""
