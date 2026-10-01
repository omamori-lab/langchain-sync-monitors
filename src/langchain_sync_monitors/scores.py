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

from pydantic import SecretStr

SCORE_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/omamori-lab/langchain-sync-monitors")
"""The namespace of every score id, so the same score always gets the same id."""


class Tracer(StrEnum):
    """A tracing tool that `MonitorMiddleware(export_scores=...)` sends each step's suspicion to.

    `LANGSMITH` writes feedback on the step's run, and `LANGFUSE` a numeric
    score on the step's observation. The score is named `<label>_suspicion`
    and holds the step's highest suspicion; a step with no sample gets none.

    - **Where it goes.** Only a run traced to the tool sends to it. LangSmith
      feedback goes to the tracer's project, through its client's endpoint,
      key and workspace. Langfuse scores use `LANGFUSE_PUBLIC_KEY`,
      `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`, and go only on steps
      found in that project, so a handler built with other keys or another
      host gets none. Building the monitor needs `LANGSMITH_API_KEY` for
      LangSmith, and the Langfuse keys and the `langfuse` package for
      Langfuse.
    - **What sends nothing.** A run with LangSmith tracing turned off by
      `tracing_context(enabled=False)`, a LangSmith client in OpenTelemetry
      mode, whose run ids are not the steps' ids and which is warned once
      [@langsmithsdk2026], and a Langfuse handler whose tracing is off, by
      `LANGFUSE_TRACING_ENABLED=false` or by a client built with
      `tracing_enabled=False` or `sample_rate=0`.
    - **What cannot be seen.** LangSmith accepts feedback on a run it never
      ingested, such as one its sampling rate dropped, so such a score is
      lost without a sign. A Langfuse step never found, such as one a sample
      rate between 0 and 1 dropped, is given up after five minutes with a
      warning, and a process that exits with one waiting spends the whole
      exit drain on it.
    - **When.** A background thread sends the scores, so the agent never
      waits: LangSmith's within about ten seconds, Langfuse's once Langfuse
      has ingested the step, ten to twenty-five seconds later in our checks.
      At exit it keeps sending for up to thirty seconds, every five, then
      logs what it drops; a process that exits right after its last step may
      drop Langfuse scores still waiting.
    - **When scores are lost.** When the process ends without running
      `atexit`: on `os._exit`, which a `multiprocessing` child started by fork
      calls, on SIGKILL, on SIGTERM without a handler, and when a Jupyter
      kernel is killed.
    - **Forks.** A forked child sends its own steps' scores, through a worker
      of its own. On macOS it reads proxies only from the `*_proxy`
      variables, never from System Settings, since that lookup kills a
      forked child. That holds for a child forked after this library was
      imported; one that imports it only after the fork reads System
      Settings as any process does.
    - **Cost and privacy.** Only numbers and ids leave the process, never the
      judge's reason. LangSmith feedback is sent with
      `extend_trace_retention` false, which by LangSmith's retention docs
      leaves the trace's retention, and so the bill, unchanged
      [@langsmith2026retention].
    """

    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"


@dataclass(frozen=True, slots=True, kw_only=True)
class LangSmithCredentials:
    """What reaches LangSmith's API: the key, the endpoint, and the workspace a key may need.

    Its repr hides the key, and two equal connections hash alike, so the
    LangSmith sender keeps one HTTP client per connection.
    """

    api_key: SecretStr
    endpoint: str
    workspace_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingScore:
    """One step's highest suspicion, waiting to be written to one tracing tool.

    `step_id` is the step span's run id, which the step's spans also carry
    as `monitor_step_id`. For LangSmith, `project` is the project the step
    was traced to and `connection` the one its tracer sends through; both
    are None for Langfuse. `queued_at` is the worker clock's reading when the
    score was queued, from which the worker counts how long it has waited.
    """

    step_id: UUID
    name: str
    value: float
    tracer: Tracer
    project: str | None
    queued_at: float
    connection: LangSmithCredentials | None = None

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
