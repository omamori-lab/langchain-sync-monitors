"""The step an agent is about to take, as a control protocol reaches it.

A protocol is written once, as a coroutine that talks to a `PendingStep`.
`AsyncPendingStep` serves `ainvoke()` and awaits the model and the monitor.
`SyncPendingStep` serves `invoke()`: its methods are coroutines too, but they
call the synchronous model and monitor without awaiting anything, so
`run_synchronously` can finish the protocol with a single `send(None)`.
"""

from __future__ import annotations

from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from typing import override

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors._langchain import (
    AgentModelRequest,
    AgentModelResponse,
    AsyncModelCallHandler,
    ModelCallHandler,
    build_request_with_messages,
)
from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Monitor,
    MonitorInput,
    PendingStep,
    Sample,
    StepRecord,
    TaskAuthor,
)
from langchain_sync_monitors.errors import MonitorError, SynchronousRunError
from langchain_sync_monitors.feedback import build_feedback_messages

SYNCHRONOUS_RUN_MESSAGE = (
    "A control protocol awaited real asynchronous work during a synchronous invoke(). "
    "Under invoke() a protocol may await only the pending step's own methods; "
    "run the agent with ainvoke() to use anything else."
)
NO_RUNNING_LOOP_MESSAGE = "no running event loop"


def run_synchronously[ResultT](coroutine: Coroutine[object, object, ResultT]) -> ResultT:
    """Finish a coroutine that never waits on real asynchronous work, and return its result.

    A protocol driven by a `SyncPendingStep` completes on its first `send`. If
    it suspends instead, it awaited something only an event loop can finish,
    so it is closed and `SynchronousRunError` is raised rather than hanging.
    asyncio's own "no running event loop" error, raised when such work starts
    outside a loop, becomes the same error.
    """
    try:
        coroutine.send(None)
    except StopIteration as finished:
        result: ResultT = finished.value
        return result
    except RuntimeError as error:
        if NO_RUNNING_LOOP_MESSAGE in str(error):
            raise SynchronousRunError(SYNCHRONOUS_RUN_MESSAGE) from error
        raise
    coroutine.close()
    raise SynchronousRunError(SYNCHRONOUS_RUN_MESSAGE)


def find_proposal(response: AgentModelResponse) -> AIMessage:
    """Return the step a model call proposes: the first AI message of its response."""
    for message in response.result:
        if isinstance(message, AIMessage):
            return message
    error_message = "The model call returned no AI message, so the monitor has no step to judge."
    raise MonitorError(error_message)


def build_sampling_conversation(
    messages: Sequence[BaseMessage],
    *,
    blocked_attempts: tuple[BlockedAttempt, ...],
) -> tuple[BaseMessage, ...]:
    """Return the conversation a sample is drawn on: the request, then each blocked attempt.

    Each blocked attempt adds its proposal and the feedback on it, so the agent
    sees why its earlier try was blocked.
    """
    feedback = [
        message
        for attempt in blocked_attempts
        for message in build_feedback_messages(attempt=attempt)
    ]
    return (*messages, *feedback)


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitoredStep(PendingStep):
    """The parts of a pending step that do not depend on whether the run awaits.

    `request` is the step's model request, `monitor` judges each sample, and
    `previous_records` holds this monitor's records for this agent from
    earlier steps of the thread. `judged_samples` collects every sample the
    monitor has judged so far, in the order the verdicts arrived, so the
    middleware can report them if the step fails before it is committed. A
    pending step lives for one step only, so this is not state shared between
    runs.
    """

    request: AgentModelRequest
    monitor: Monitor
    task_author: TaskAuthor
    previous_records: tuple[StepRecord, ...] = ()
    judged_samples: list[Sample] = field(default_factory=list)

    def build_conversation(
        self,
        blocked_attempts: tuple[BlockedAttempt, ...],
    ) -> tuple[BaseMessage, ...]:
        """Return this step's conversation, followed by any blocked attempts and their feedback."""
        return build_sampling_conversation(self.request.messages, blocked_attempts=blocked_attempts)

    def build_sample_request(self, conversation: tuple[BaseMessage, ...]) -> AgentModelRequest:
        """Return this step's request with the conversation a sample is drawn on."""
        return build_request_with_messages(self.request, messages=conversation)

    def build_trusted_request(self, trusted_model: BaseChatModel) -> AgentModelRequest:
        """Return this step's request, with the same conversation and tools, for another model."""
        return self.request.override(model=trusted_model)

    def build_monitor_input(
        self,
        *,
        conversation: tuple[BaseMessage, ...],
        proposal: AIMessage,
    ) -> MonitorInput:
        """Return what the monitor judges: the conversation the sample saw and its proposal."""
        return MonitorInput(history=conversation, proposal=proposal, task_author=self.task_author)

    def keep_judged_sample(self, sample: Sample) -> Sample:
        """Remember a judged sample as evidence, and return it."""
        self.judged_samples.append(sample)
        return sample


@dataclass(frozen=True, kw_only=True, eq=False)
class AsyncPendingStep(MonitoredStep):
    """A pending step under `ainvoke()`, which awaits the model and the monitor.

    `handler` runs the rest of the middleware stack and the model.
    """

    handler: AsyncModelCallHandler

    @override
    async def sample(
        self,
        *,
        count: int,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
        concurrently: bool = False,
    ) -> tuple[Sample, ...]:
        """Draw `count` samples through the rest of the stack and have the monitor judge each.

        With `concurrently`, the samples are drawn at once in a task group, so
        one failed draw cancels the others.
        """
        conversation = self.build_conversation(blocked_attempts)
        if concurrently:
            draws = (self.draw_sample(conversation) for _ in range(count))
            return tuple(await run_concurrently(draws))
        return tuple([await self.draw_sample(conversation) for _ in range(count)])

    async def draw_sample(self, conversation: tuple[BaseMessage, ...]) -> Sample:
        """Call the untrusted model once on the conversation and judge its proposal."""
        response = await self.handler(self.build_sample_request(conversation))
        proposal = find_proposal(response)
        monitor_input = self.build_monitor_input(conversation=conversation, proposal=proposal)
        verdict = await self.monitor.evaluate(monitor_input)
        return self.keep_judged_sample(
            Sample(response=response, proposal=proposal, verdict=verdict),
        )

    @override
    async def request_trusted_step(self, *, trusted_model: BaseChatModel) -> AgentModelResponse:
        """Have the trusted model write the step through the rest of the stack."""
        return await self.handler(self.build_trusted_request(trusted_model))


@dataclass(frozen=True, kw_only=True, eq=False)
class SyncPendingStep(MonitoredStep):
    """A pending step under `invoke()`, whose coroutines never suspend.

    Its methods call the synchronous `handler` and `Monitor.evaluate_sync`
    directly, so `run_synchronously` can drive any protocol written against
    `PendingStep`.
    """

    handler: ModelCallHandler

    @override
    async def sample(
        self,
        *,
        count: int,
        blocked_attempts: tuple[BlockedAttempt, ...] = (),
        concurrently: bool = False,
    ) -> tuple[Sample, ...]:
        """Draw `count` samples one after another and have the monitor judge each.

        Without an event loop nothing can run at once, so `concurrently` draws
        the same samples in sequence.
        """
        conversation = self.build_conversation(blocked_attempts)
        return tuple(self.draw_sample(conversation) for _ in range(count))

    def draw_sample(self, conversation: tuple[BaseMessage, ...]) -> Sample:
        """Call the untrusted model once on the conversation and judge its proposal."""
        response = self.handler(self.build_sample_request(conversation))
        proposal = find_proposal(response)
        monitor_input = self.build_monitor_input(conversation=conversation, proposal=proposal)
        verdict = self.monitor.evaluate_sync(monitor_input)
        return self.keep_judged_sample(
            Sample(response=response, proposal=proposal, verdict=verdict),
        )

    @override
    async def request_trusted_step(self, *, trusted_model: BaseChatModel) -> AgentModelResponse:
        """Have the trusted model write the step through the rest of the stack."""
        return self.handler(self.build_trusted_request(trusted_model))
