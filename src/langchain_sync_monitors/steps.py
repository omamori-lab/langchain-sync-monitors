"""How the middleware drives a control protocol through one step.

A step is one run of the agent's model node. `plan_step` holds its logic
once, as a plan that does no I/O: it asks for a reply of the agent's model and
the protocol's decision on it, then for the replies each of the protocol's
requests asks for and the decision on them, and records each decision in the
monitor's trajectory [@hyoshigi2026]. After a block, the agent reads its
blocked reply and the feedback on it, and replies again. An audit, which no
person can answer inside the model node, halts the step. `run_step_sync`
answers the plan under `invoke()`, one call after another, and `run_step`
under `ainvoke()`, drawing the replies of one request at once.

`read_draw` reads a reply as the action the monitor rates: the last AI
message of the model call's response, so the trailing tool message of
tool-strategy structured output, and the blocked attempts a monitor inside
this one commits ahead of its step, are left aside. A reply whose id, or a
call's id, the run already holds, as a cached or scripted model's, or a retry
after a block, can give, is read under fresh `uuid4` ids, and so is each call
after the first with an id another call of the reply has. The new ids are set
by position wherever the reply carries the old one, and in the tool messages
of its response, which keep answering the call they answered first. The
trajectory refuses an id it holds, and LangGraph merges messages that share
one.
"""

from __future__ import annotations

import functools
import warnings
from collections import Counter
from collections.abc import Generator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import NamedTuple, assert_never, cast
from uuid import uuid4

import hyoshigi
from hyoshigi import (
    Action,
    Audit,
    Block,
    Flag,
    Halt,
    Policy,
    ProtocolDecision,
    ProtocolRequest,
    Replace,
    Resample,
    Run,
    TrustedStep,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langchain_sync_monitors._langchain import (
    AgentModelRequest,
    AgentModelResponse,
    AsyncModelCallHandler,
    ModelCallHandler,
    build_request_with_messages,
)
from langchain_sync_monitors.concurrency import run_concurrently
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.feedback import build_feedback_messages
from langchain_sync_monitors.model_calls import CachedResampleWarning, is_response_cache_active
from langchain_sync_monitors.options import LIBRARY_DIRECTORY
from langchain_sync_monitors.protocols import NO_PERSON_REASON, FeedbackTemplate
from langchain_sync_monitors.trajectories import LoadedTrajectory
from langchain_sync_monitors.trajectory_events import read_proposed_action

CALL_ID_KEYS = frozenset({"id", "call_id", "tool_call_id", "tool_use_id"})
"""The keys under which a content block carries a call's id, in the formats LangChain reads."""

type Renamed = dict[str, tuple[str, ...]]
"""Each id a reply carries that is renamed, with the id each of its occurrences takes, in order."""

type ProtocolOutcome = ProtocolDecision | ProtocolRequest
"""What a protocol returns: a decision, or a request for more proposals."""

type SyncProtocol = hyoshigi.DeferToResample | hyoshigi.BlockAndFeedBack
type AsyncProtocol = hyoshigi.AsyncDeferToResample | hyoshigi.AsyncBlockAndFeedBack


class Draw(NamedTuple):
    """A reply drawn in the step: the action read from it, and its model call's response."""

    action: Action
    response: AgentModelResponse


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """One step as the middleware drives it.

    `request` is the model call's request, `trajectory` the monitor's
    trajectory as the step loaded it, `trusted_model` the model a trusted
    step is drawn from, if any, and `feedback` the template of the message a
    block sends the agent. `drawn` keeps each reply drawn, by its action's id,
    and `outcomes` each outcome the protocol returned, for the report of a
    step that fails. A step lives for one model call, so this is no state
    shared between runs.
    """

    request: AgentModelRequest
    trajectory: LoadedTrajectory
    trusted_model: BaseChatModel | None
    feedback: FeedbackTemplate
    drawn: dict[str, Draw] = field(default_factory=dict)
    outcomes: list[ProtocolOutcome] = field(default_factory=list)

    @property
    def decisions(self) -> tuple[ProtocolDecision, ...]:
        """The decisions the trajectory recorded in this step, in order."""
        return self.trajectory.trajectory.records[self.trajectory.base["records"] :]


@dataclass(frozen=True, slots=True, kw_only=True)
class Decide:
    """Draw one reply of the agent's model on the conversation, and have the protocol decide it."""

    conversation: tuple[BaseMessage, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Resume:
    """Draw the replies a protocol's request asks for, and have the protocol resume with them.

    `count` replies are drawn on the conversation, from the agent's model for
    `Policy.UNTRUSTED` and from the trusted model for `Policy.TRUSTED`.
    """

    request: ProtocolRequest
    conversation: tuple[BaseMessage, ...]
    policy: Policy
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class StepResult:
    """How a step ended: each block's reply and the feedback on it, then the last decision."""

    blocked: tuple[BaseMessage, ...]
    final: Run | Flag | Replace | Halt


type StepPlan = Generator[Decide | Resume, ProtocolOutcome, StepResult]


def plan_step(step: Step) -> StepPlan:
    """Plan a step: a reply and its decision, each request answered, each block fed back."""
    blocked: tuple[BaseMessage, ...] = ()
    outcome = yield Decide(conversation=(*step.request.messages, *blocked))
    while True:
        match outcome:
            case Resample() | TrustedStep():
                outcome = yield build_resume(outcome, step=step, blocked=blocked)
            case Block():
                blocked = (*blocked, *record_block(outcome, step=step))
                outcome = yield Decide(conversation=(*step.request.messages, *blocked))
            case Run() | Flag() | Replace() | Audit() | Halt():
                return StepResult(blocked=blocked, final=finish_step(outcome, step=step))
            case unreachable:
                assert_never(unreachable)


def build_resume(
    request: ProtocolRequest,
    *,
    step: Step,
    blocked: tuple[BaseMessage, ...],
) -> Resume:
    """Ask for a request's replies: its resamples, or a trusted step, given a trusted model."""
    conversation = (*step.request.messages, *blocked)
    match request:
        case Resample(count=count):
            policy = Policy.UNTRUSTED
        case TrustedStep():
            policy, count = Policy.TRUSTED, int(step.trusted_model is not None)
        case unreachable:
            assert_never(unreachable)
    return Resume(request=request, conversation=conversation, policy=policy, count=count)


def record_block(block: Block, *, step: Step) -> tuple[BaseMessage, ...]:
    """Record a block, with the feedback the agent reads as its reason; return its messages.

    They are the blocked reply, under its own id, and the feedback on it.
    """
    rendered = replace(block, reason=step.feedback.render(reason=block.reason))
    step.trajectory.trajectory.record(rendered)
    reply = find_reply(step.drawn[block.action.id].response)
    return (reply, *build_feedback_messages(reply, feedback=rendered.reason))


def finish_step(
    final: Run | Flag | Replace | Audit | Halt,
    *,
    step: Step,
) -> Run | Flag | Replace | Halt:
    """Record the decision that ends the step; an audit is followed by a halt.

    No person can answer an audit inside the model node, so the halt
    follows it, at its position, with the proposals it held.
    """
    trajectory = step.trajectory.trajectory
    match final:
        case Audit(proposals=proposals, position=position):
            trajectory.record(final)
            ended: Run | Flag | Replace | Halt = Halt(
                reason=NO_PERSON_REASON,
                proposals=proposals,
                position=position,
            )
        case Run() | Flag() | Replace() | Halt():
            ended = final
        case unreachable:
            assert_never(unreachable)
    trajectory.record(ended)
    return ended


def find_reply_position(response: AgentModelResponse) -> int:
    """Return the position of the reply in a model call's response: its last AI message.

    A response without an AI message raises `MonitorError`.
    """
    for position in reversed(range(len(response.result))):
        if isinstance(response.result[position], AIMessage):
            return position
    message = "The model call returned no AI message, so the monitor has no step to judge."
    raise MonitorError(message)


def find_reply(response: AgentModelResponse) -> AIMessage:
    """Return the reply in a model call's response, its last AI message."""
    return cast("AIMessage", response.result[find_reply_position(response)])


def read_draw(response: AgentModelResponse, *, step: Step) -> Action:
    """Read a draw's reply as the action the monitor rates, under fresh ids where the run has its.

    The step keeps the response, with the reply, and the tool messages after
    it that answer its calls, as read.
    """
    position = find_reply_position(response)
    renamed = find_repeated_ids(cast("AIMessage", response.result[position]), step=step)
    messages = [
        rename_message_ids(message, renamed=renamed) if index >= position else message
        for index, message in enumerate(response.result)
    ]
    action = read_proposed_action(cast("AIMessage", messages[position]))
    step.drawn[action.id] = Draw(action=action, response=replace(response, result=messages))
    return action


def find_repeated_ids(reply: AIMessage, *, step: Step) -> Renamed:
    """Map each id of the reply that the run holds, or the reply repeats, to the ids it takes.

    The reply's id takes a fresh `uuid4` when the monitor's trajectory read or
    wrote a message under it, or holds an action or a draw of this step with
    it. A call's first occurrence keeps its id unless an action of the
    trajectory or a draw of this step made a call with it, and then takes a
    fresh one; each later occurrence in the reply takes a fresh one.
    """
    actions = [
        *(event for event in step.trajectory.trajectory.history if isinstance(event, Action)),
        *(draw.action for draw in step.drawn.values()),
    ]
    taken_ids = step.trajectory.seen | {action.id for action in actions}
    taken_call_ids = {call.id for action in actions for call in action.tool_calls}
    proposed = read_proposed_action(reply)
    renamed: Renamed = {proposed.id: (str(uuid4()),)} if proposed.id in taken_ids else {}
    call_ids = [call.id for call in proposed.tool_calls]
    return renamed | plan_call_ids(call_ids, taken=taken_call_ids)


def plan_call_ids(call_ids: Sequence[str], *, taken: AbstractSet[str]) -> Renamed:
    """Map each call id that is taken or repeated to the ids its occurrences take, in order.

    The first occurrence keeps its id unless it is taken; each later one takes a fresh `uuid4`.
    """
    renamed: Renamed = {}
    for call_id, count in Counter(call_ids).items():
        first = str(uuid4()) if call_id in taken else call_id
        occurrences = (first, *(str(uuid4()) for _ in range(count - 1)))
        if occurrences != (call_id,):
            renamed[call_id] = occurrences
    return renamed


@dataclass(slots=True)
class Renaming:
    """Hands out a message's new ids: the next occurrence of an id in one place takes the next."""

    renamed: Renamed
    taken: Counter[tuple[str, str]] = field(default_factory=Counter)

    def take(self, value: str | None, *, place: str) -> str | None:
        """Return the id the next occurrence of `value` in `place` takes, or `value` itself.

        Occurrences past the reply's count take its last id, and None stays None.
        """
        if value is None or value not in self.renamed:
            return value
        occurrences = self.renamed[value]
        index = min(self.taken[place, value], len(occurrences) - 1)
        self.taken[place, value] += 1
        return occurrences[index]


def rename_message_ids(message: BaseMessage, *, renamed: Renamed) -> BaseMessage:
    """Return a reply, or a tool message, with each renamed id it carries replaced.

    A reply's calls and malformed calls are renamed in order, and its content
    blocks by the order of each key that carries a call's id. A tool message
    answers the first occurrence of its call's id.
    """
    if not renamed:
        return message
    renaming = Renaming(renamed=renamed)
    if isinstance(message, ToolMessage):
        call_id = renaming.take(message.tool_call_id, place="answer")
        return message.model_copy(update={"tool_call_id": call_id})
    if not isinstance(message, AIMessage):
        return message
    content = message.content
    blocks = (
        content
        if isinstance(content, str)
        else [rename_block_ids(block, renaming=renaming) for block in content]
    )
    calls = [
        {**call, "id": renaming.take(call["id"], place="call")}
        for call in (*message.tool_calls, *message.invalid_tool_calls)
    ]
    return message.model_copy(
        update={
            "id": renaming.take(message.id, place="message"),
            "content": blocks,
            "tool_calls": calls[: len(message.tool_calls)],
            "invalid_tool_calls": calls[len(message.tool_calls) :],
        },
    )


def rename_block_ids(block: str | dict[str, object], *, renaming: Renaming) -> object:
    """Return a content block with each renamed call id it carries replaced.

    The id sits under one of `CALL_ID_KEYS`, at the top of the block, or in
    the provider's payload of a block LangChain could not map, under `value`.
    """
    if not isinstance(block, dict):
        return block
    renamed_block = {
        key: renaming.take(value, place=key)
        if key in CALL_ID_KEYS and isinstance(value, str)
        else value
        for key, value in block.items()
    }
    payload = block.get("value")
    if isinstance(payload, dict):
        renamed_block["value"] = rename_block_ids(payload, renaming=renaming)
    return renamed_block


@functools.cache
def warn_about_cached_resamples() -> None:
    """Emit the `CachedResampleWarning` for the agent's model, on the first call in the process.

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


def build_draw_requests(resume: Resume, *, step: Step) -> list[AgentModelRequest]:
    """Return the model requests a `Resume` draws: `count` of them, on its conversation.

    A trusted step is drawn from the trusted model, with the same tools. A
    resample redraws a conversation the step already drew, so under a
    response cache on the agent's model it warns `CachedResampleWarning`.
    """
    request = build_request_with_messages(step.request, messages=resume.conversation)
    if resume.policy is Policy.TRUSTED and step.trusted_model is not None:
        request = request.override(model=step.trusted_model)
    if isinstance(resume.request, Resample) and is_response_cache_active(request.model):
        warn_about_cached_resamples()
    return [request] * resume.count


def decide_sync(
    request: Decide | Resume,
    *,
    step: Step,
    handler: ModelCallHandler,
    protocol: SyncProtocol,
) -> ProtocolOutcome:
    """Draw what the plan asks for, one reply after another, and return the protocol's answer."""
    trajectory = step.trajectory.trajectory
    match request:
        case Decide(conversation=conversation):
            response = handler(build_request_with_messages(step.request, messages=conversation))
            outcome = protocol.decide(read_draw(response, step=step), trajectory=trajectory)
        case Resume():
            responses = [handler(each) for each in build_draw_requests(request, step=step)]
            # Only DeferToResample asks for proposals, so only it is resumed.
            outcome = cast("hyoshigi.DeferToResample", protocol).resume(
                request.request,
                proposals=tuple(read_draw(response, step=step) for response in responses),
                trajectory=trajectory,
            )
        case unreachable:
            assert_never(unreachable)
    step.outcomes.append(outcome)
    return outcome


async def decide(
    request: Decide | Resume,
    *,
    step: Step,
    handler: AsyncModelCallHandler,
    protocol: AsyncProtocol,
) -> ProtocolOutcome:
    """Draw what the plan asks for, a request's replies at once; return the protocol's answer."""
    trajectory = step.trajectory.trajectory
    match request:
        case Decide(conversation=conversation):
            response = await handler(
                build_request_with_messages(step.request, messages=conversation)
            )
            outcome = await protocol.decide(read_draw(response, step=step), trajectory=trajectory)
        case Resume():
            draws = (
                draw_reply(each, handler=handler)
                for each in build_draw_requests(request, step=step)
            )
            responses = await run_concurrently(draws)
            # Only DeferToResample asks for proposals, so only it is resumed.
            outcome = await cast("hyoshigi.AsyncDeferToResample", protocol).resume(
                request.request,
                proposals=tuple(read_draw(response, step=step) for response in responses),
                trajectory=trajectory,
            )
        case unreachable:
            assert_never(unreachable)
    step.outcomes.append(outcome)
    return outcome


async def draw_reply(
    request: AgentModelRequest, *, handler: AsyncModelCallHandler
) -> AgentModelResponse:
    """Draw one reply through the rest of the stack, as a coroutine a task group can run."""
    return await handler(request)


def run_step_sync(step: Step, *, handler: ModelCallHandler, protocol: SyncProtocol) -> StepResult:
    """Drive a step's plan under `invoke()`, and return how the step ended."""
    plan = plan_step(step)
    try:
        request = next(plan)
        while True:
            request = plan.send(decide_sync(request, step=step, handler=handler, protocol=protocol))
    except StopIteration as finished:
        return finished.value


async def run_step(
    step: Step,
    *,
    handler: AsyncModelCallHandler,
    protocol: AsyncProtocol,
) -> StepResult:
    """Drive a step's plan under `ainvoke()`, and return how the step ended."""
    plan = plan_step(step)
    try:
        request = next(plan)
        while True:
            request = plan.send(
                await decide(request, step=step, handler=handler, protocol=protocol)
            )
    except StopIteration as finished:
        return finished.value
