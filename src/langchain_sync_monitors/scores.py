"""What the suspicion score export passes around: the tools, a waiting score and a delivery report.

`MonitorMiddleware(export_scores=...)` names the tools with `Tracer`. Each
committed step becomes one `PendingScore` per tool its run is traced to, and a
`ScoreSender` writes a batch of them, saying in a `DeliveryReport` which were
written, which must wait and which the service refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

SCORE_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/omamori-lab/langchain-sync-monitors")
"""The namespace of every score id, so the same score always gets the same id."""


class Tracer(StrEnum):
    """A tracing tool that `export_scores` can send each step's suspicion to.

    `LANGSMITH` writes feedback on the step's run, and `LANGFUSE` a numeric
    score on the step's observation.
    """

    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingScore:
    """One step's highest suspicion, waiting to be written to one tracing tool.

    `step_id` is the step span's run id, which the step's spans also carry
    as `monitor_step_id`. `project` is the LangSmith project the step was
    traced to, and None for Langfuse. `queued_at` is the worker clock's
    reading when the score was queued, from which the worker counts how long
    it has waited.
    """

    step_id: UUID
    name: str
    value: float
    tracer: Tracer
    project: str | None
    queued_at: float

    @property
    def score_id(self) -> UUID:
        """The score's fixed id, so a score written twice is stored once."""
        return uuid5(
            SCORE_ID_NAMESPACE,
            f"{self.tracer}/{self.project or ''}/{self.step_id}/{self.name}",
        )


@dataclass(slots=True, kw_only=True)
class DeliveryReport:
    """What came of one attempt to write a batch of scores to one tool.

    `waiting` scores are tried again in a later window, such as a step the
    service has not ingested yet, or one a failed request left unwritten.
    `refused` scores are dropped, since the service would refuse them again,
    and `refusal` says why. `pause_seconds` asks the worker not to call the
    service again for that long, as a `429` answer's `Retry-After` does.
    """

    written: list[PendingScore] = field(default_factory=list)
    waiting: list[PendingScore] = field(default_factory=list)
    refused: list[PendingScore] = field(default_factory=list)
    refusal: str | None = None
    pause_seconds: float | None = None


class ScoreSender(Protocol):
    """Writes scores to one tracing tool over its public HTTP API."""

    def send(self, scores: Sequence[PendingScore]) -> DeliveryReport:
        """Write the scores in as few requests as the API allows, and report each one's fate."""
        ...

    def close(self) -> None:
        """Release the sender's HTTP connections."""
        ...
