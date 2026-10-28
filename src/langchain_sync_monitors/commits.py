"""How a monitored step ends: committed once decided, or reported when it fails first.

`build_step_response` gives the messages a decided step commits, and
`commit_step` hands them to the agent state and appends the step's record to
`monitor_log`. `report_failed_step` covers a step in which a call raised
before the step was decided: that step is never committed, so its rated
proposals are reported instead. All run inside `MonitorMiddleware`'s
model-call hooks, and log under `langchain_sync_monitors.middleware`, the
logger the docs give for committed and failed steps.
"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import assert_never

from hyoshigi import Audit, Flag, Halt, Replace, Run
from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentModelRequest,
    AgentModelResponse,
    MonitorStepEvent,
    MonitorStepFailedEvent,
    StructuredOutput,
    TracedRun,
    write_stream_event,
)
from langchain_sync_monitors.contracts import SampleRecord, StepRecord
from langchain_sync_monitors.feedback import build_halt_message
from langchain_sync_monitors.halts import build_halt_inputs_update
from langchain_sync_monitors.protocols import DEFAULT_HALT_MESSAGE
from langchain_sync_monitors.records import (
    build_sample_records,
    redact_failed_samples,
    redact_text,
)
from langchain_sync_monitors.spans import StepIdentity, build_step_span_inputs
from langchain_sync_monitors.steps import Step, StepResult, find_reply, find_reply_position
from langchain_sync_monitors.subagent_returns import build_answered_update
from langchain_sync_monitors.task_authorship import build_note_update
from langchain_sync_monitors.trajectories import (
    LoadedTrajectory,
    add_reply_results,
    build_trajectory_update,
    mark_committed,
)

# The middleware's logger, which docs/how-to/read-the-monitor-log.md names for these messages.
logger = logging.getLogger("langchain_sync_monitors.middleware")


def render_sample_suspicions(samples: Sequence[SampleRecord]) -> str:
    """Render each judged sample's suspicion for a log line, as in `0.950, 0.100`."""
    return ", ".join(f"{sample['suspicion']:.3f}" for sample in samples)


def describe_failure(step: Step, *, error: BaseException) -> tuple[str, list[SampleRecord]]:
    """Return the error as text, and the samples the failed step had rated.

    The samples are those of the decisions recorded in the step and of the
    protocol's latest answer. With the trajectory's redaction, the error and
    each sample's reason pass through its text function, and the proposals are
    left out, since the latest answer was not recorded.
    """
    samples = build_sample_records([*step.decisions, *step.outcomes[-1:]], executed_id=None)
    description = f"{type(error).__name__}: {error}"
    redaction = step.trajectory.trajectory.redaction
    if redaction is None:
        return description, samples
    return (
        redact_text(description, redaction=redaction),
        redact_failed_samples(samples, redaction=redaction),
    )


@contextmanager
def report_failed_step(
    step: Step,
    *,
    identity: StepIdentity,
    traced_step: TracedRun,
    middleware_name: str,
) -> Generator[None, None, None]:
    """Report the proposals rated in a step whose block raises, before the error propagates.

    The step is never committed, so this is the only trace of its samples:
    a `MonitorStepFailedEvent` on `stream_mode="custom"`, which holds each
    rated proposal of the decisions recorded in the step and of the latest
    outcome of its protocol, and the error, and the step span's inputs, which
    name the first sample rated. When the monitor had rated anything, a warning also
    gives the error's type and each sample's suspicion, and says the event
    holds the samples. It quotes no proposal, reason or error message,
    since any of them can hold the transcript. `identity` names the step as
    its span does. A redaction on the trajectory redacts the event, as
    `describe_failure` says. LangGraph's own control flow, such as an
    interrupt, is not a failed step, and passes through unreported.
    """
    try:
        yield
    except GraphBubbleUp:
        raise
    except BaseException as error:
        step_number = identity.step_number
        description, samples = describe_failure(step, error=error)
        traced_step.inputs_at_end = build_step_span_inputs(step_number=step_number, samples=samples)
        event = MonitorStepFailedEvent(
            type="monitor_step_failed",
            agent=identity.agent,
            monitor=identity.monitor,
            step_number=step_number,
            error=description,
            samples=samples,
        )
        if identity.delegation_id is not None:
            event["delegation_id"] = identity.delegation_id
        write_stream_event(step.request.runtime, event=event)
        if samples:
            logger.warning(
                "%s: step %d failed with %s before it was committed, so the %d sample(s) the "
                "monitor judged are not in monitor_log. Their suspicions: %s. The step's "
                "monitor_step_failed event on stream_mode='custom' holds the samples.",
                middleware_name,
                step_number,
                type(error).__name__,
                len(samples),
                render_sample_suspicions(samples),
            )
        raise


def build_step_response(result: StepResult, *, step: Step) -> AgentModelResponse:
    """Return the messages a decided step commits, as its last decision says.

    Each block's reply and the feedback on it come first. A run, a flag or a
    replace commits the draw of the action it runs, whole, with its
    structured response, and adds what follows the reply in the draw to the
    trajectory. An audit commits the held reply alone, for a person to
    answer before any of the agent's own tools run. A halt commits
    `DEFAULT_HALT_MESSAGE`.
    """
    match result.final:
        case Run(action=action) | Flag(action=action) | Replace(action=action):
            drawn = step.drawn[action.id].response
            add_reply_results(step.trajectory, messages=drawn.result[find_reply_position(drawn) :])
            return ModelResponse(
                result=[*result.blocked, *drawn.result],
                structured_response=drawn.structured_response,
            )
        case Audit(action=action):
            return ModelResponse(
                result=[*result.blocked, find_reply(step.drawn[action.id].response)]
            )
        case Halt():
            return ModelResponse(result=[*result.blocked, build_halt_message(DEFAULT_HALT_MESSAGE)])
        case unreachable:
            assert_never(unreachable)


def commit_step(
    request: AgentModelRequest,
    *,
    response: AgentModelResponse,
    record: StepRecord,
    middleware_name: str,
    trajectory: LoadedTrajectory,
) -> ExtendedModelResponse[StructuredOutput]:
    """Commit the step's messages and append the step's record to `monitor_log`.

    The record is also written to `stream_mode="custom"`, once the response
    and the update are built, so a commit that fails streams no record. A
    step held for a person writes neither: the answer writes its record. The
    untagged human messages in the state that the monitor had not seen are
    recorded as seen, so the next run does not take them for its input, and
    the subagent halts the step answered are removed. The monitor's
    trajectory gains what the step added to it, every committed message is
    marked seen there, and the step answers the trajectory's lost writes,
    for which it halted.
    """
    mark_committed(trajectory, committed=response.result)
    is_held = record["outcome"] == "audited"
    update = {
        **({} if is_held else {MONITOR_LOG_KEY: [record]}),
        # A middleware listed after the monitor runs its `before_model` hook after the
        # monitor's own, so a human message it wrote is first recorded here.
        **build_note_update(request.state),
        **build_halt_inputs_update(record, state=request.state, middleware_name=middleware_name),
        **build_answered_update(request.state),
        **build_trajectory_update(trajectory, answered_lost_writes=trajectory.lost_writes),
    }
    if not is_held:
        event = MonitorStepEvent(type="monitor_step", record=record)
        write_stream_event(request.runtime, event=event)
    logger.debug(
        "%s committed step %d: %s", middleware_name, record["step_number"], record["outcome"]
    )
    return ExtendedModelResponse(model_response=response, command=Command(update=update))
