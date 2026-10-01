"""How a monitored step ends: committed once decided, or reported when it fails first.

`commit_step` hands the agent state the decided messages and appends the
step's record to `monitor_log`. `report_failed_step` covers a step in which a
call raised before the protocol decided: that step is never committed, so its
judged samples are reported instead. Both run inside `MonitorMiddleware`'s
model-call hooks, and both log under `langchain_sync_monitors.middleware`, the
logger the docs give for committed and failed steps.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentModelRequest,
    MonitorStepEvent,
    MonitorStepFailedEvent,
    StructuredOutput,
    TracedRun,
    write_stream_event,
)
from langchain_sync_monitors.contracts import (
    ControlProtocol,
    FeedbackVisibility,
    SampleRecord,
    StepDecision,
    StepRecord,
)
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.feedback import build_blocked_attempt_messages
from langchain_sync_monitors.halts import build_halt_inputs_update
from langchain_sync_monitors.pending_steps import MonitoredStep
from langchain_sync_monitors.records import build_sample_record, render_value
from langchain_sync_monitors.run_inputs import build_refresh_update
from langchain_sync_monitors.spans import StepIdentity, build_step_span_inputs
from langchain_sync_monitors.subagent_returns import build_answered_update
from langchain_sync_monitors.task_authorship import build_note_update

# The middleware's logger, which docs/how-to/read-the-monitor-log.md names for these messages.
logger = logging.getLogger("langchain_sync_monitors.middleware")


def render_sample_suspicions(samples: Sequence[SampleRecord]) -> str:
    """Render each judged sample's suspicion for a log line, as in `0.950, 0.100`."""
    return ", ".join(f"{sample['suspicion']:.3f}" for sample in samples)


@contextmanager
def report_failed_step(
    step: MonitoredStep,
    *,
    identity: StepIdentity,
    traced_step: TracedRun,
    middleware_name: str,
) -> Iterator[None]:
    """Report the samples judged in a step whose block raises, before the error propagates.

    The step is never committed, so this is the only trace of its samples:
    a `MonitorStepFailedEvent` on `stream_mode="custom"`, which holds each
    sample and the error, and the step span's inputs, which name the first
    sample judged. When the monitor had judged anything, a warning also
    gives the error's type and each sample's suspicion, and says the event
    holds the samples. It quotes no proposal, reason or error message,
    since any of them can hold the transcript. `identity` names the step as
    its span does. LangGraph's own control flow, such as an interrupt, is
    not a failed step, and passes through unreported.
    """
    try:
        yield
    except GraphBubbleUp:
        raise
    except BaseException as error:
        step_number = identity.step_number
        samples = [build_sample_record(sample, executed=False) for sample in step.judged_samples]
        traced_step.inputs_at_end = build_step_span_inputs(step_number=step_number, samples=samples)
        event = MonitorStepFailedEvent(
            type="monitor_step_failed",
            agent=identity.agent,
            monitor=identity.monitor,
            step_number=step_number,
            error=f"{type(error).__name__}: {error}",
            samples=samples,
        )
        if identity.delegation_id is not None:
            event["delegation_id"] = identity.delegation_id
        write_stream_event(step.request, event=event)
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


def check_step_decision(decision: object, *, protocol: ControlProtocol) -> StepDecision:
    """Return a protocol's decision once it has the shape a commit reads, else raise `MonitorError`.

    A decision must be a `StepDecision` whose `response` is a `ModelResponse`
    that holds a list of messages. A custom protocol could return a bare
    `AIMessage` there, which `wrap_model_call` itself accepts from a
    middleware [@langchain2026]; the commit would then fail after streaming
    the step's record, and report none of its judged samples. Checked inside
    the step instead, a malformed decision fails it as any error does. The
    error names the value by its type alone, since a protocol could return
    the model's text, and no error message may quote the transcript.
    """
    protocol_name = type(protocol).__name__
    if not isinstance(decision, StepDecision):
        message = f"{protocol_name}.decide returned {render_value(decision)}, not a StepDecision"
        raise MonitorError(message)
    response = decision.response
    is_model_response = isinstance(response, ModelResponse) and isinstance(response.result, list)
    if not is_model_response or not all(isinstance(item, BaseMessage) for item in response.result):
        message = (
            f"{protocol_name} decided with a response that is {render_value(response)}, "
            "not a ModelResponse holding a list of messages: wrap the messages in "
            "ModelResponse(result=[...])"
        )
        raise MonitorError(message)
    return decision


def commit_step(
    request: AgentModelRequest,
    *,
    decision: StepDecision,
    record: StepRecord,
    middleware_name: str,
    feedback_visibility: FeedbackVisibility,
) -> ExtendedModelResponse[StructuredOutput]:
    """Commit the decided messages and append the step's record to `monitor_log`.

    The record is also written to `stream_mode="custom"`, once the response
    and the update are built, so a commit that fails streams no record.
    With `FeedbackVisibility.IN_TRANSCRIPT`, each blocked attempt and its
    feedback come before the step's own messages. The untagged human
    messages in the state that the monitor had not seen are recorded as
    seen, so the next run does not take them for its input, and the
    subagent halts and blocks the step answered are removed.
    """
    messages = list(decision.response.result)
    if feedback_visibility is FeedbackVisibility.IN_TRANSCRIPT:
        messages = [*build_blocked_attempt_messages(decision.blocked_attempts), *messages]
    response = ModelResponse(
        result=messages,
        structured_response=decision.response.structured_response,
    )
    update = {
        MONITOR_LOG_KEY: [record],
        # A middleware listed after the monitor runs its `before_model` hook after the
        # monitor's own, so a human message it wrote is first recorded here.
        **build_note_update(request.state),
        # A middleware listed after the monitor may have rewritten a kept input since.
        **build_refresh_update(request.state),
        **build_halt_inputs_update(record, state=request.state, middleware_name=middleware_name),
        **build_answered_update(request.state),
    }
    write_stream_event(request, event=MonitorStepEvent(type="monitor_step", record=record))
    logger.debug(
        "%s committed step %d: %s", middleware_name, record["step_number"], record["outcome"]
    )
    return ExtendedModelResponse(model_response=response, command=Command(update=update))
