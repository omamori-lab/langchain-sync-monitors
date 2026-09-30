"""The step an agent is about to take, as a control protocol reaches it.

A protocol is written once, as a coroutine that talks to a `PendingStep`.
`AsyncPendingStep` serves `ainvoke()` and awaits the model and the monitor.
`SyncPendingStep` serves `invoke()`: its methods are coroutines too, but they
call the synchronous model and monitor without awaiting anything, so
`run_synchronously` can finish the protocol with a single `send(None)`.
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import threading
import warnings
from collections.abc import Coroutine, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypedDict, override

from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors._langchain import (
    AgentModelRequest,
    AgentModelResponse,
    AsyncModelCallHandler,
    ModelCallHandler,
    build_request_with_messages,
    open_traced_run,
    open_traced_run_sync,
)
from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.contracts import (
    BlockedAttempt,
    Monitor,
    MonitorInput,
    PendingStep,
    Sample,
    StepDecision,
    StepRecord,
    TaskAuthor,
    Verdict,
)
from langchain_sync_monitors.errors import MonitorError, SynchronousRunError
from langchain_sync_monitors.feedback import build_feedback_messages
from langchain_sync_monitors.spans import (
    StepIdentity,
    build_judgement_span,
    build_verdict_outputs,
)
from langchain_sync_monitors.task_authorship import mark_context_notes
from langchain_sync_monitors.thresholds import LIBRARY_DIRECTORY

SYNCHRONOUS_RUN_MESSAGE = (
    "A control protocol awaited real asynchronous work during a synchronous invoke(). "
    "Under invoke() a protocol may await only the pending step's own methods; "
    "run the agent with ainvoke() to use anything else."
)
MONITOR_EVENT_LOOP_MESSAGE = (
    "The monitor {monitor_name} started asynchronous work in evaluate_sync, where no event "
    "loop can run it, during a synchronous invoke(). A monitor's evaluate_sync must finish "
    "without an event loop; run the agent with ainvoke() to use asyncio."
)
MISSING_EVENT_LOOP_MESSAGES = ("no running event loop", "no current event loop")
"""Parts of asyncio's errors for work started where no event loop can run it."""

CLOSED_STEP_MESSAGE = (
    "A pending step was used after its synchronous invoke() step was over, from a task a "
    "control protocol scheduled on an event loop. Run the agent with ainvoke() to use asyncio."
)


class CachedResampleWarning(UserWarning):
    """A step is sampled more than once on the same request while a response cache is active.

    LangChain answers an identical request from its cache [@langchaincore2026],
    so every resample is a copy of the first sample and resampling can never
    find a safer one: resampling helps less the more deterministic the model
    is [@bhatt2025ctrlzpost], and a cached model is fully deterministic.
    """


def read_running_loop() -> asyncio.AbstractEventLoop | None:
    """Return the event loop running in this thread, or None when there is none."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def is_missing_event_loop_error(error: RuntimeError) -> bool:
    """Tell whether asyncio refused to start work because no event loop could run it."""
    return any(fragment in str(error) for fragment in MISSING_EVENT_LOOP_MESSAGES)


def cancel_tasks_started_since(
    loop: asyncio.AbstractEventLoop | None,
    *,
    earlier_tasks: set[asyncio.Task[object]],
) -> None:
    """Cancel every task scheduled on `loop` since `earlier_tasks` were read.

    The loop is blocked while a synchronous step runs, so those tasks have not
    started yet, and cancelling them now keeps them from ever calling a model.
    """
    if loop is None:
        return
    for task in asyncio.all_tasks(loop) - earlier_tasks:
        task.cancel()


def run_synchronously[ResultT](coroutine: Coroutine[object, object, ResultT]) -> ResultT:
    """Finish a coroutine that never waits on real asynchronous work, and return its result.

    A protocol driven by a `SyncPendingStep` completes on its first `send`. If
    it suspends instead, it awaited something only an event loop can finish,
    so it is closed and `SynchronousRunError` is raised rather than hanging.
    asyncio's own errors for work started where no loop can run it, such as
    `gather` or `ensure_future` outside a running loop, become the same error;
    the pending step names the monitor instead when the error came from the
    monitor's `evaluate_sync`. Inside a running loop, as in a notebook, any task the
    protocol scheduled is cancelled before it starts.
    """
    loop = read_running_loop()
    # Read before the protocol runs, so only the tasks it schedules are cancelled, never
    # those already on the loop.
    earlier_tasks = asyncio.all_tasks(loop) if loop is not None else set()
    try:
        coroutine.send(None)
    except StopIteration as finished:
        result: ResultT = finished.value
        return result
    except SynchronousRunError:
        cancel_tasks_started_since(loop, earlier_tasks=earlier_tasks)
        raise
    except RuntimeError as error:
        if not is_missing_event_loop_error(error):
            raise
        cancel_tasks_started_since(loop, earlier_tasks=earlier_tasks)
        raise SynchronousRunError(SYNCHRONOUS_RUN_MESSAGE) from error
    # The protocol suspended on work only an event loop can finish. Closing it runs its
    # cleanup now, rather than whenever the coroutine is collected.
    coroutine.close()
    cancel_tasks_started_since(loop, earlier_tasks=earlier_tasks)
    raise SynchronousRunError(SYNCHRONOUS_RUN_MESSAGE)


def find_proposal(response: AgentModelResponse) -> AIMessage:
    """Return the step a model call proposes: the first AI message of its response."""
    # With structured output through a tool, the result also holds tool messages, so the loop
    # skips anything that is not an AI message.
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


def is_response_cache_active(model: BaseChatModel) -> bool:
    """Tell whether LangChain answers this model's calls from a response cache.

    A model's own `cache` wins: an instance is used, `False` turns caching
    off, and `True` or `None` use the global cache when one is set.
    """
    if isinstance(model.cache, BaseCache):
        return True
    return model.cache is not False and get_llm_cache() is not None


@functools.cache
def warn_about_cached_resamples() -> None:
    """Emit the `CachedResampleWarning`, on the first call in the process only.

    The cache on this function is what keeps the warning to one per process:
    Python's own once-per-location filter forgets what it has shown whenever
    any code changes the warning filters, as libraries often do.
    """
    warnings.warn(
        "A LangChain response cache is active for the agent's model, so each resample of a "
        "step returns a copy of the first sample and resampling cannot find a safer one. "
        "Build the untrusted model with cache=False, or unset the global cache with "
        "set_llm_cache(None).",
        CachedResampleWarning,
        stacklevel=2,
        skip_file_prefixes=(LIBRARY_DIRECTORY,),
    )


class PendingStepOptions(TypedDict):
    """The keywords a pending step is built with, whether the run awaits or not."""

    request: AgentModelRequest
    monitor: Monitor
    task_author: TaskAuthor
    task_message_ids: frozenset[str]
    previous_records: tuple[StepRecord, ...]
    blocks_in_thread: int
    new_subagent_blocks: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedStep:
    """A step as the middleware reads it before drawing any sample.

    `identity` names the step, `halt` is the halt it gets without a sample, if
    any, and `options` are the keywords of its pending step.
    """

    identity: StepIdentity
    halt: StepDecision | None
    options: PendingStepOptions


@dataclass(frozen=True, kw_only=True, eq=False)
class MonitoredStep(PendingStep):
    """The parts of a pending step that do not depend on whether the run awaits.

    `request` is the step's model request, `monitor` judges each sample, and
    `previous_records` holds this monitor's records for this agent from
    earlier steps of the thread, every turn of a checkpointed thread included,
    or, inside a subagent, from earlier steps of its current delegation.
    `task_message_ids` holds the ids of the human messages that arrived as a
    run's input; the monitor reads every other human message as a context
    note. The middleware counts `blocks_in_thread` and `new_subagent_blocks`
    from the log and from the agent's `Delegation`.
    `judged_samples` collects every sample the monitor has judged so far, in
    the order the verdicts arrived, so the middleware can report them if the
    step fails before it is committed. `sample_numbers` numbers the samples
    in the order their draws start, for the judgement spans. A pending step
    lives for one step only, so this is not state shared between runs.
    """

    request: AgentModelRequest
    monitor: Monitor
    task_author: TaskAuthor
    task_message_ids: frozenset[str] = frozenset()
    previous_records: tuple[StepRecord, ...] = ()
    blocks_in_thread: int = 0
    new_subagent_blocks: int = 0
    judged_samples: list[Sample] = field(default_factory=list)
    # The blocked attempts each draw of this step was shown, so a repeated request is spotted.
    sampled_attempts: list[tuple[BlockedAttempt, ...]] = field(default_factory=list)
    sample_numbers: Iterator[int] = field(default_factory=lambda: itertools.count(1))

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
        """Return what the monitor judges: the conversation the sample saw and its proposal.

        In the monitor's copy of the conversation, every untagged human
        message that did not arrive as a run's input is tagged as a context
        note, so it cannot speak as the task author.
        """
        history = mark_context_notes(conversation, task_message_ids=self.task_message_ids)
        return MonitorInput(history=history, proposal=proposal, task_author=self.task_author)

    def keep_judged_sample(self, sample: Sample) -> Sample:
        """Remember a judged sample as evidence, and return it."""
        self.judged_samples.append(sample)
        return sample

    def warn_when_resamples_are_cached(
        self,
        *,
        count: int,
        blocked_attempts: tuple[BlockedAttempt, ...],
    ) -> None:
        """Warn, once per process, when this draw repeats a request under a response cache.

        A request repeats when several samples are drawn at once, or when an
        earlier draw of this step saw the same blocked attempts.
        """
        is_repeat = count > 1 or blocked_attempts in self.sampled_attempts
        self.sampled_attempts.append(blocked_attempts)
        if is_repeat and is_response_cache_active(self.request.model):
            warn_about_cached_resamples()


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
        self.warn_when_resamples_are_cached(count=count, blocked_attempts=blocked_attempts)
        conversation = self.build_conversation(blocked_attempts)
        if concurrently:
            draws = (self.draw_sample(conversation) for _ in range(count))
            return tuple(await run_concurrently(draws))
        return tuple([await self.draw_sample(conversation) for _ in range(count)])

    async def draw_sample(self, conversation: tuple[BaseMessage, ...]) -> Sample:
        """Call the untrusted model once on the conversation and judge its proposal.

        The sample's number is taken before the first await, so samples drawn
        at once are numbered in the order their draws were started.
        """
        sample_number = next(self.sample_numbers)
        response = await self.handler(self.build_sample_request(conversation))
        proposal = find_proposal(response)
        monitor_input = self.build_monitor_input(conversation=conversation, proposal=proposal)
        judgement_span = build_judgement_span(sample_number=sample_number, monitor=self.monitor)
        async with open_traced_run(judgement_span) as traced_judgement:
            verdict = await self.monitor.evaluate(monitor_input)
            traced_judgement.outputs = build_verdict_outputs(verdict)
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
    `PendingStep`. Once the middleware has finished the step it calls
    `close`, and from then on the step refuses to call a model: a task that a
    protocol scheduled on an event loop, and that runs after the step, cannot
    reach the model.
    """

    handler: ModelCallHandler
    closed: threading.Event = field(default_factory=threading.Event)

    def close(self) -> None:
        """Refuse every later call to the model through this step."""
        self.closed.set()

    def check_open(self) -> None:
        """Raise `SynchronousRunError` when the step is already over."""
        if self.closed.is_set():
            raise SynchronousRunError(CLOSED_STEP_MESSAGE)

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
        self.check_open()
        self.warn_when_resamples_are_cached(count=count, blocked_attempts=blocked_attempts)
        conversation = self.build_conversation(blocked_attempts)
        return tuple(self.draw_sample(conversation) for _ in range(count))

    def draw_sample(self, conversation: tuple[BaseMessage, ...]) -> Sample:
        """Call the untrusted model once on the conversation and judge its proposal."""
        sample_number = next(self.sample_numbers)
        response = self.handler(self.build_sample_request(conversation))
        proposal = find_proposal(response)
        monitor_input = self.build_monitor_input(conversation=conversation, proposal=proposal)
        judgement_span = build_judgement_span(sample_number=sample_number, monitor=self.monitor)
        with open_traced_run_sync(judgement_span) as traced_judgement:
            verdict = self.evaluate_without_event_loop(monitor_input)
            traced_judgement.outputs = build_verdict_outputs(verdict)
        return self.keep_judged_sample(
            Sample(response=response, proposal=proposal, verdict=verdict),
        )

    def evaluate_without_event_loop(self, monitor_input: MonitorInput) -> Verdict:
        """Have the monitor judge a proposal through `evaluate_sync`.

        asyncio's error for work the monitor started where no event loop can
        run it becomes `SynchronousRunError` naming the monitor, so that
        `run_synchronously` does not blame the protocol for it.
        """
        try:
            return self.monitor.evaluate_sync(monitor_input)
        except RuntimeError as error:
            if not is_missing_event_loop_error(error):
                raise
            monitor_name = type(self.monitor).__name__
            message = MONITOR_EVENT_LOOP_MESSAGE.format(monitor_name=monitor_name)
            raise SynchronousRunError(message) from error

    @override
    async def request_trusted_step(self, *, trusted_model: BaseChatModel) -> AgentModelResponse:
        """Have the trusted model write the step through the rest of the stack."""
        self.check_open()
        return self.handler(self.build_trusted_request(trusted_model))
