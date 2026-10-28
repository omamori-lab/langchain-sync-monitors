"""The `export_scores` option: each step's highest and lowest suspicion, as scores in tracing tools.

`MonitorMiddleware(export_scores={Tracer.LANGSMITH, Tracer.LANGFUSE})` sends
two scores for each committed step that judged a sample, off by default:
`<label>_suspicion`, the highest, and `<label>_lowest_suspicion`, the lowest.
Each is the recorded `SuspicionScore`, sent by hyoshigi's report classes
with its explanation as the comment [@hyoshigi2026]. They sit on the step's
`monitor step` run: feedback in LangSmith, a numeric score on the
observation in Langfuse [@langsmith2026dashboards; @langfuse2026scores].

- **Where it goes.** The handlers of the step span name the tools the run is
  traced to. For LangSmith's `LangChainTracer`, the feedback goes to the
  tracer's project, through the tracer's own client. For Langfuse's
  LangChain handler, recognised by its package without importing it, the
  scores go through the handler's own client, on the observation of the
  step. Neither tool's key is read from the environment.
- **The review flag.** The lowest score's report also marks a step flagged
  for review, once: the report does so itself when the lowest score reaches
  `flag_threshold`, and `flag` does so for any other flagged step, one with
  no sample included.
- **A held step.** A step an audit holds for a person sends its scores when
  the run pauses, before the answer. The answer sends no score, and a stop is
  flagged on the decision span of the answer.
- **What sends nothing.** A run traced to neither tool, a run with
  LangSmith tracing turned off by `tracing_context(enabled=False)`, a
  Langfuse handler whose client traces nothing, and a LangSmith client in
  OpenTelemetry mode, which warns once: its run ids are derived from span
  ids, so no feedback could find the step [@langsmithsdk2026]. A Langfuse
  handler of a shape this library cannot read warns once.
- **The agent never waits.** LangSmith's project lookup and its posts run on
  a pool of daemon threads, and Langfuse's client queues its scores. At exit
  the pool keeps running its calls for up to 30 seconds. Nothing here raises
  into a run; a failure is logged by the error's type alone.
"""

from __future__ import annotations

import logging
import operator
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, assert_never
from uuid import UUID

from hyoshigi import (
    LangfuseScores,
    LangSmithFeedback,
    ProtocolDecision,
    ReviewFlag,
    SuspicionScore,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.run_helpers import get_tracing_context
from langsmith.utils import get_tracer_project

from langchain_sync_monitors._langchain import TracedRun
from langchain_sync_monitors.errors import (
    ConfigurationError,
    MissingExtraError,
    build_missing_extra_message,
)
from langchain_sync_monitors.model_calls import is_package_installed
from langchain_sync_monitors.options import check_enum_option, describe_option_value
from langchain_sync_monitors.records import find_rated, reaches_flag_threshold
from langchain_sync_monitors.request_pool import PROCESS_POOL

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langsmith import Client

logger = logging.getLogger(__name__)

SCORE_NAME_SUFFIX = "_suspicion"
"""What follows a monitor's label in the name of its highest score."""

LOWEST_SCORE_NAME_SUFFIX = "_lowest_suspicion"
"""What follows a monitor's label in the name of its lowest score."""

FLAG_NAME = "flagged"
"""The name of the mark that puts a step in a person's review queue."""

LANGFUSE_PACKAGE = "langfuse"
"""The top-level package of Langfuse's SDK, whose LangChain handler traces a run to Langfuse."""

PROJECT_ID_CACHE_SIZE = 64
"""How many LangSmith projects' ids are remembered, by client and name."""

OTEL_MODE_NOTICE = (
    "score export: LangSmith's tracer sends this run through OpenTelemetry, whose run ids are "
    "not the steps' ids, so no LangSmith score is sent in this process while it does"
)
"""Said once per process when a LangSmith client runs in OpenTelemetry mode."""

UNREADABLE_HANDLER_NOTICE = (
    "score export: a Langfuse handler of a shape this library cannot read, which keeps neither "
    "its client nor its open observations where Langfuse's does, gets no score"
)
"""Said once per process when a handler from `langfuse` does not hold the step's observation."""


class Tracer(StrEnum):
    """A tracing tool that `MonitorMiddleware(export_scores=...)` sends each step's scores to.

    `LANGSMITH` writes feedback on the step's run, and `LANGFUSE` a numeric
    score on the step's observation. A step that judged a sample gets
    `<label>_suspicion`, its highest suspicion, and `<label>_lowest_suspicion`,
    its lowest, each with the explanation as its comment; a flagged step also
    gets a `flagged` mark. Only a run traced to the tool sends to it. The
    tool's client is the tracer's own. What a report sends is
    [hyoshigi's](https://omamori-lab.github.io/hyoshigi/how-to/reporting-scores-to-langfuse-and-langsmith/).
    """

    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"


@dataclass(slots=True, kw_only=True)
class ProcessNotices:
    """The warnings this process says once, such as that LangSmith's OTel mode gets no score."""

    said: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def warn_once(self, message: str) -> None:
        """Log the warning the first time it comes up in this process."""
        with self.lock:
            if message in self.said:
                return
            self.said.add(message)
        logger.warning(message)


PROCESS_NOTICES = ProcessNotices()
"""The warnings this process has said."""


def forget_process_state_after_fork() -> None:
    """In a forked child, start afresh: no inherited pool, and no lock another thread held."""
    global LANGSMITH_PROJECT_IDS_LOCK
    PROCESS_POOL.forget_after_fork()
    PROCESS_NOTICES.lock = threading.Lock()
    LANGSMITH_PROJECT_IDS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=forget_process_state_after_fork)


def check_score_export(tracers: object) -> frozenset[Tracer]:
    """Return the tools as a frozenset, or raise `ConfigurationError` for any that cannot work.

    `tracers` must be a set of `Tracer` members; a plain string is refused.
    Langfuse also needs the `langfuse` extra, without which no run is traced
    to it and `MissingExtraError` is raised. No credential is checked: each
    tool's tracer holds its own.
    """
    if not isinstance(tracers, AbstractSet):
        message = (
            "export_scores must be a set of Tracer members, such as "
            f"{{Tracer.LANGSMITH, Tracer.LANGFUSE}}, got {describe_option_value(tracers)}"
        )
        raise ConfigurationError(message)
    for tracer in tracers:
        check_enum_option(tracer, option_type=Tracer, parameter_name="export_scores")
    if Tracer.LANGFUSE in tracers and not is_package_installed(LANGFUSE_PACKAGE):
        feature = "export_scores with Tracer.LANGFUSE"
        raise MissingExtraError(build_missing_extra_message(feature, extra=LANGFUSE_PACKAGE))
    return frozenset(tracers)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreNames:
    """The names of a monitor's two scores."""

    highest: str
    lowest: str


def build_score_names(label: str) -> ScoreNames:
    """Return a monitor's score names: its label with `_suspicion`, and with `_lowest_suspicion`."""
    return ScoreNames(
        highest=f"{label}{SCORE_NAME_SUFFIX}",
        lowest=f"{label}{LOWEST_SCORE_NAME_SUFFIX}",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Extremes:
    """The scores of a step's most and least suspicious samples, as recorded."""

    highest: SuspicionScore
    lowest: SuspicionScore


@dataclass(frozen=True, slots=True, kw_only=True)
class StepScores:
    """What one step sends: its extreme scores, none for a step with no sample, and its flag."""

    extremes: Extremes | None
    flagged: bool


def read_step_scores(decisions: Sequence[ProtocolDecision], *, flagged: bool) -> StepScores:
    """Return the step's highest and lowest recorded scores, none when it judged no sample."""
    scores = [rated.score for rated in find_rated(decisions)]
    if not scores:
        return StepScores(extremes=None, flagged=flagged)
    value = operator.attrgetter("value")
    extremes = Extremes(highest=max(scores, key=value), lowest=min(scores, key=value))
    return StepScores(extremes=extremes, flagged=flagged)


def needs_flag(scores: StepScores, *, flag_threshold: float | None) -> bool:
    """Tell whether the step must be flagged by a call, since its report would not flag it.

    The report flags a score at or above the threshold when it sends it. A
    step flagged for another reason, such as a halt, is flagged by a call.
    """
    lowest = None if scores.extremes is None else scores.extremes.lowest.value
    return scores.flagged and not reaches_flag_threshold(lowest, flag_threshold=flag_threshold)


@dataclass(frozen=True, slots=True, kw_only=True)
class LangSmithDestination:
    """The run a step's scores go to in LangSmith: the tracer's client and project, and its id."""

    client: Client
    project_name: str
    run_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class LangfuseDestination:
    """The observation a step's scores go to in Langfuse: the handler's client, and its ids."""

    client: Langfuse
    trace_id: str
    observation_id: str


type Destination = LangSmithDestination | LangfuseDestination

LANGSMITH_PROJECT_IDS: OrderedDict[tuple[Client, str], str] = OrderedDict()
"""The id of each LangSmith project looked up, by client and project name, oldest first."""

LANGSMITH_PROJECT_IDS_LOCK = threading.Lock()


def read_project_id(client: Client, *, project_name: str) -> str:
    """Return the id of the project, from LangSmith the first time and from memory after.

    The lookup runs on the pool. A lookup that raises remembers nothing.
    """
    key = (client, project_name)
    with LANGSMITH_PROJECT_IDS_LOCK:
        known = LANGSMITH_PROJECT_IDS.get(key)
    if known is not None:
        return known
    project_id = str(client.read_project(project_name=project_name).id)
    with LANGSMITH_PROJECT_IDS_LOCK:
        LANGSMITH_PROJECT_IDS[key] = project_id
        while len(LANGSMITH_PROJECT_IDS) > PROJECT_ID_CACHE_SIZE:
            LANGSMITH_PROJECT_IDS.popitem(last=False)
    return project_id


def is_langfuse_handler(handler: BaseCallbackHandler) -> bool:
    """Tell whether a handler is Langfuse's LangChain handler, or built on it, without an import."""
    return any(
        cls.__module__.partition(".")[0] == LANGFUSE_PACKAGE for cls in type(handler).__mro__
    )


def read_hex_id(value: object, *, digits: int) -> str | None:
    """Return the value if it is a lower-case hexadecimal string of `digits` digits, else None."""
    if (
        isinstance(value, str)
        and len(value) == digits
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def read_langfuse_destination(
    handler: BaseCallbackHandler,
    *,
    step_id: UUID,
) -> LangfuseDestination | None:
    """Return the observation of the step in Langfuse, or None when it has none to score.

    Langfuse's handler keeps the client it traces through in
    `_langfuse_client` and each open run's observation in `_runs`, under the
    run's id, and an observation has its `trace_id`, 32 hexadecimal digits,
    and its `id`, 16 [@langfuse2026]. The handler exposes none publicly, so
    these are private reads, by `getattr`, all in this function, and the
    step span must still be open. A handler that holds no such observation
    warns once. A client whose tracing is off gives its handler an
    observation with ids of zeros, and drops the score itself.
    """
    client = getattr(handler, "_langfuse_client", None)
    runs = getattr(handler, "_runs", None)
    observation = runs.get(step_id) if isinstance(runs, Mapping) else None
    trace_id = read_hex_id(getattr(observation, "trace_id", None), digits=32)
    observation_id = read_hex_id(getattr(observation, "id", None), digits=16)
    if client is None or trace_id is None or observation_id is None:
        PROCESS_NOTICES.warn_once(UNREADABLE_HANDLER_NOTICE)
        return None
    return LangfuseDestination(client=client, trace_id=trace_id, observation_id=observation_id)


def read_langsmith_destination(
    tracer: LangChainTracer,
    *,
    step_id: UUID,
) -> LangSmithDestination | None:
    """Return where a LangSmith tracer sends the step's run, or None when no score can go there.

    None, quietly, when tracing is off for the run, as `tracing_context(enabled=False)`
    turns it off, which the tracer itself checks [@langchaincore2026]. None,
    warned once, when the client runs in OpenTelemetry mode, whose run ids are
    derived from span ids, so that no feedback could find its step
    [@langsmithsdk2026].
    """
    if get_tracing_context().get("enabled") is False:
        return None
    if tracer.client.tracing_mode == "otel":
        PROCESS_NOTICES.warn_once(OTEL_MODE_NOTICE)
        return None
    # A tracer built without a project traces to the environment's, as the SDK reads it.
    project_name = tracer.project_name or get_tracer_project()
    if project_name is None:
        return None
    return LangSmithDestination(client=tracer.client, project_name=project_name, run_id=step_id)


def read_destination(
    handler: BaseCallbackHandler,
    *,
    step_id: UUID,
    tracers: AbstractSet[Tracer],
) -> Destination | None:
    """Return where the handler traces the step to, if it is a tool's tracer among `tracers`."""
    if Tracer.LANGSMITH in tracers and isinstance(handler, LangChainTracer):
        return read_langsmith_destination(handler, step_id=step_id)
    if Tracer.LANGFUSE in tracers and is_langfuse_handler(handler):
        return read_langfuse_destination(handler, step_id=step_id)
    return None


def find_destinations(
    traced_step: TracedRun,
    *,
    tracers: AbstractSet[Tracer],
) -> list[Destination]:
    """Return where the step span's handlers trace the step to, among `tracers`, once each."""
    step_id = traced_step.run_id
    if step_id is None:
        return []
    destinations = (
        read_destination(handler, step_id=step_id, tracers=tracers)
        for handler in traced_step.handlers
    )
    return list(dict.fromkeys(item for item in destinations if item is not None))


def send_each(*sends: Callable[[], None]) -> None:
    """Run every send in turn, and log each that raises by the error's type alone."""
    for send in sends:
        try:
            send()
        except Exception as error:
            logger.warning("score export: a score could not be sent: %s", type(error).__name__)


def post_to_langsmith(
    destination: LangSmithDestination,
    *,
    scores: StepScores,
    names: ScoreNames,
    flag_threshold: float | None,
) -> None:
    """Post the step's scores, and its flag, as feedback on its run: the pool's call.

    The lowest score's report carries the review flag, which it posts itself
    for a score at or above `flag_threshold`. Each post is made whether or not
    the one before it failed.
    """
    project_id = read_project_id(destination.client, project_name=destination.project_name)
    run_id = str(destination.run_id)
    highest = LangSmithFeedback(
        client=destination.client,
        project_id=project_id,
        name=names.highest,
    )
    lowest = LangSmithFeedback(
        client=destination.client,
        project_id=project_id,
        name=names.lowest,
        review_flag=ReviewFlag(name=FLAG_NAME, threshold=flag_threshold),
    )
    sends: list[Callable[[], None]] = []
    if scores.extremes is not None:
        sends.append(partial(highest.send, scores.extremes.highest, run_id=run_id))
        sends.append(partial(lowest.send, scores.extremes.lowest, run_id=run_id))
    if needs_flag(scores, flag_threshold=flag_threshold):
        sends.append(partial(lowest.flag, run_id=run_id))
    send_each(*sends)


def send_to_langfuse(
    destination: LangfuseDestination,
    *,
    scores: StepScores,
    names: ScoreNames,
    flag_threshold: float | None,
) -> None:
    """Queue the step's scores, and its flag, as scores on its observation: Langfuse sends them.

    The lowest score's report carries the review flag, which it queues itself
    for a score at or above `flag_threshold`. Each is queued whether or not
    the one before it failed.
    """
    highest = LangfuseScores(client=destination.client, name=names.highest)
    lowest = LangfuseScores(
        client=destination.client,
        name=names.lowest,
        review_flag=ReviewFlag(name=FLAG_NAME, threshold=flag_threshold),
    )
    ids = {"trace_id": destination.trace_id, "observation_id": destination.observation_id}
    sends: list[Callable[[], None]] = []
    if scores.extremes is not None:
        sends.append(partial(highest.send, scores.extremes.highest, **ids))
        sends.append(partial(lowest.send, scores.extremes.lowest, **ids))
    if needs_flag(scores, flag_threshold=flag_threshold):
        sends.append(partial(lowest.flag, **ids))
    send_each(*sends)


def send_to_destination(
    destination: Destination,
    *,
    scores: StepScores,
    names: ScoreNames,
    flag_threshold: float | None,
) -> None:
    """Send the step's scores to the tool: LangSmith's on the process's pool, Langfuse's now.

    Langfuse's client queues what it is given and sends it in the background.
    """
    match destination:
        case LangSmithDestination():
            PROCESS_POOL.read_pool().submit(
                partial(
                    post_to_langsmith,
                    destination,
                    scores=scores,
                    names=names,
                    flag_threshold=flag_threshold,
                )
            )
        case LangfuseDestination():
            send_to_langfuse(destination, scores=scores, names=names, flag_threshold=flag_threshold)
        case unreachable:
            assert_never(unreachable)


def send_step_scores(
    traced_step: TracedRun,
    *,
    decisions: Sequence[ProtocolDecision],
    flagged: bool,
    label: str,
    flag_threshold: float | None,
    tracers: AbstractSet[Tracer],
) -> None:
    """Send the step's highest and lowest suspicion, and its flag, to the tools it is traced to.

    `decisions` are the decisions the trajectory recorded for the step, and
    `flagged` whether the step is flagged for review. Nothing is sent when
    the monitor exports to no tool or when the step span had no handler. It
    never raises: a failure is logged by the error's type alone, and the run
    goes on. A tool that fails does not stop the tools after it, nor a score
    the scores after it.
    """
    if not tracers:
        return
    scores = read_step_scores(decisions, flagged=flagged)
    names = build_score_names(label)
    try:
        destinations = find_destinations(traced_step, tracers=tracers)
    except Exception as error:
        log_unsent_scores(error)
        return
    for destination in destinations:
        try:
            send_to_destination(
                destination, scores=scores, names=names, flag_threshold=flag_threshold
            )
        except Exception as error:
            log_unsent_scores(error)


def log_unsent_scores(error: Exception) -> None:
    """Log that a step's scores could not be sent to a tool, by the error's type alone."""
    logger.warning("score export: the step's scores could not be sent: %s", type(error).__name__)
