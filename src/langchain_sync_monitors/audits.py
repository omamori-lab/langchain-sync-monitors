"""A step held for a person to audit, and the person's answer.

When a protocol decides `Audit` [@hyoshigi2026], the model node commits the
held reply alone, and the monitor's `after_model` hook pauses the run with
LangGraph's `interrupt` before any tool runs, as LangChain's
`HumanInTheLoopMiddleware` does [@langchain2026]. The model node never
pauses: LangGraph runs a paused node again on resume [@langgraph2026]. The
interrupt's value is an `AuditRequest`. An `AuditAnswer` is checked against
a schema built for its audit, so a malformed one raises pydantic's
`ValidationError` and LangGraph keeps nothing of it [@langgraph2026]. The
answer is recorded at the audit's position: an approval as a `Run`, an edit
as a `Replace` whose action is a trusted proposal, a rejection as a `Block`
fed back to the agent, and a stop as a `Halt`; the step's record goes to
`monitor_log` with it. The how-to Answer an audit gives what each commits.

A step is held when its monitor's last record is an audit and the state
holds the held reply. Its decisions are the audit and the blocks before it,
back to the first record that is not a block, or that is a block right after
an audit at the same position: a person's rejection, which ends an earlier
step.

New input sent to a paused thread stops the held step, at the start of the
run it begins: the halt is recorded first, the held reply and every tool
message answering its calls, such as those Deep Agents' patch middleware
adds [@deepagents2026], are removed, and the paused run counts as ended at
the audit, so the new input is the run's input and lifts the halt.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial, reduce
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    NotRequired,
    TypedDict,
    assert_never,
    cast,
)
from uuid import uuid4

from hyoshigi import JSON, Audit, Block, Halt, Policy, Proposal, ProtocolDecision, Replace, Run
from langchain.agents.middleware.types import JumpTo
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolCall, ToolMessage
from langgraph.types import Overwrite, interrupt
from pydantic import AfterValidator, Discriminator

from langchain_sync_monitors._langchain import (
    MONITOR_LOG_KEY,
    AgentStateUpdate,
    MonitorStepEvent,
    write_stream_event,
)
from langchain_sync_monitors.contracts import SampleRecord, StepRecord
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.feedback import build_feedback_messages, build_halt_message
from langchain_sync_monitors.given_ids import build_run_start_trajectories_update
from langchain_sync_monitors.halts import build_halt_inputs_update
from langchain_sync_monitors.message_ids import build_run_start_update
from langchain_sync_monitors.protocols import DEFAULT_HALT_MESSAGE, FeedbackTemplate
from langchain_sync_monitors.records import build_sample_records
from langchain_sync_monitors.spans import StepIdentity
from langchain_sync_monitors.state_keys import RUN_OPEN_KEY
from langchain_sync_monitors.step_preparation import build_step_identity
from langchain_sync_monitors.task_authorship import read_state_messages
from langchain_sync_monitors.trajectories import (
    LoadedTrajectory,
    add_reply_results,
    build_trajectory_update,
    load_trajectory,
    mark_committed,
)
from langchain_sync_monitors.trajectory_channel import read_list, read_trajectory_state
from langchain_sync_monitors.trajectory_events import read_proposed_action

if TYPE_CHECKING:
    # The middleware module imports this one.
    from langchain_sync_monitors.middleware import MonitorMiddleware

type AnswerType = Literal["approve", "edit", "reject", "stop"]
"""The kinds of answer a person can give to an audit."""

ANSWERS_TO_CALLS: Final[list[AnswerType]] = ["approve", "edit", "reject", "stop"]
"""The answers offered for a held reply with tool calls."""

ANSWERS_TO_A_FINAL_ANSWER: Final[list[AnswerType]] = ["approve", "reject", "stop"]
"""The answers offered for a held final answer, which has no call to edit."""

NEW_INPUT_STOP_REASON: Final = (
    "New input arrived while the step was held for a person to audit, so the step was stopped."
)
"""The reason the trajectory records for the halt of a held step that new input stopped."""


class AuditRequest(TypedDict):
    """What a person sees of a held step: the value of the run's interrupt.

    `tool_calls` are the held reply's calls as the model made them, so the
    person approves what will run. `samples` are the step's rated
    proposals, as `monitor_log` stores them, none executed yet. `answers`
    are the answers this audit takes. A subagent's request also names its
    delegation, the tool call that started it.
    """

    type: Literal["monitor_audit"]
    agent: str
    monitor: str
    step_number: int
    tool_calls: list[ToolCall]
    samples: list[SampleRecord]
    answers: list[AnswerType]
    delegation_id: NotRequired[str]


class EditedToolCall(TypedDict):
    """A tool call as a person edits it: the tool's name and its arguments."""

    name: str
    args: dict[str, JSON]


class ApproveAnswer(TypedDict):
    """Run the held reply as the model proposed it."""

    type: Literal["approve"]


class EditAnswer(TypedDict):
    """Run the held reply's calls as the person edits them: one for each held call, in order."""

    type: Literal["edit"]
    edited_tool_calls: list[EditedToolCall]


class RejectAnswer(TypedDict):
    """Run nothing, and tell the agent why: `message`, or else the monitor's explanation."""

    type: Literal["reject"]
    message: NotRequired[str]


class StopAnswer(TypedDict):
    """Stop the run; `message`, if given, is the reason the halt records."""

    type: Literal["stop"]
    message: NotRequired[str]


type AuditAnswer = ApproveAnswer | EditAnswer | RejectAnswer | StopAnswer
"""A person's answer to an audit, sent with `Command(resume=...)`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HeldStep:
    """A step held for a person: the monitor's trajectory, the audit, the reply and the decisions.

    `decisions` are the step's own, the audit last, and `identity` names the
    step, whose number is one past the agent's records.
    """

    loaded: LoadedTrajectory
    audit: Audit
    reply: AIMessage
    decisions: tuple[ProtocolDecision, ...]
    identity: StepIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerCommit:
    """What an answer commits: its decision, its messages, the reply that runs, and the route."""

    decision: Run | Replace | Block | Halt
    messages: tuple[BaseMessage, ...] = ()
    ran: AIMessage | None = None
    route: JumpTo | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AnsweredAudit:
    """The update an answer returns, and the step's record in it."""

    update: AgentStateUpdate
    record: StepRecord


def is_audit_last(stored: Mapping[str, JSON]) -> bool:
    """Tell whether a stored trajectory's last record is an audit, without loading it."""
    match read_list(stored, key="records"):
        case [*_, {"kind": "audit"}]:
            return True
        case _:
            return False


def find_message(state: Mapping[str, object], *, message_id: str) -> AIMessage | None:
    """Return the AI message with this id in the state, if any."""
    for message in read_state_messages(state):
        if isinstance(message, AIMessage) and message.id == message_id:
            return message
    return None


def is_rejection(records: Sequence[ProtocolDecision], *, index: int) -> bool:
    """Tell whether the record at `index` is a person's rejection: a block right after an audit
    at the same position.
    """
    previous = records[index - 1] if index > 0 else None
    return isinstance(previous, Audit) and previous.position == records[index].position


def find_step_decisions(records: Sequence[ProtocolDecision]) -> tuple[ProtocolDecision, ...]:
    """Return the held step's decisions: its audit, last, and the blocks the step made before it."""
    start = len(records) - 1
    while (
        start > 0
        and isinstance(records[start - 1], Block)
        and not is_rejection(records, index=start - 1)
    ):
        start -= 1
    return tuple(records[start:])


def find_held_step(
    state: Mapping[str, object],
    *,
    middleware: MonitorMiddleware,
) -> HeldStep | None:
    """Return this monitor's held step, or None when its last record is no audit.

    An audit whose held reply is gone from the messages, as `update_state`
    can remove it, can be neither answered nor stopped, so it raises
    `MonitorError`, which names the monitor and quotes nothing of the reply.
    """
    stored = read_trajectory_state(state, label=middleware.label)
    if stored is None or not is_audit_last(stored["trajectory"]):
        return None
    loaded = load_trajectory(state, label=middleware.label, task_author=middleware.task_author)
    records = loaded.trajectory.records
    # read_draw gave the held reply its action's id.
    match records:
        case [*_, Audit() as audit]:
            reply = find_message(state, message_id=audit.action.id)
        case _:
            return None
    if reply is None:
        message = (
            f"The monitor {middleware.label!r} holds a step for a person to audit, but the "
            "held reply is no longer in messages, so the audit can be neither answered nor "
            "stopped. Put the held reply back with update_state, or start a new thread."
        )
        raise MonitorError(message)
    return HeldStep(
        loaded=loaded,
        audit=audit,
        reply=reply,
        decisions=find_step_decisions(records),
        identity=build_step_identity(state, middleware=middleware),
    )


def build_audit_request(held: HeldStep) -> AuditRequest:
    """Return what the person sees of a held step, its calls as the model made them."""
    identity = held.identity
    request = AuditRequest(
        type="monitor_audit",
        agent=identity.agent,
        monitor=identity.monitor,
        step_number=identity.step_number,
        tool_calls=list(held.reply.tool_calls),
        samples=build_sample_records(held.decisions, executed_id=None),
        answers=list(ANSWERS_TO_CALLS if held.reply.tool_calls else ANSWERS_TO_A_FINAL_ANSWER),
    )
    if identity.delegation_id is not None:
        request["delegation_id"] = identity.delegation_id
    return request


def check_edit_length(answer: EditAnswer, *, count: int) -> EditAnswer:
    """Return an edit that holds one edited call for each held call, else raise `ValueError`."""
    if len(answer["edited_tool_calls"]) != count:
        message = f"an edit holds one edited call for each of the {count} held call(s), in order"
        raise ValueError(message)
    return answer


def build_answer_schema(request: AuditRequest) -> object:
    """Return the schema an answer to this audit must fit: one of its answers, and for an edit,
    one edited call for each held call.

    It is a union discriminated on `type`, as LangChain's per-call review
    schema is [@langchain2026]; `interrupt` checks the answer against it, so
    the length check's `ValueError` reaches the caller as pydantic's
    `ValidationError`.
    """
    count = len(request["tool_calls"])
    edit = Annotated[EditAnswer, AfterValidator(partial(check_edit_length, count=count))]
    members = {"approve": ApproveAnswer, "edit": edit, "reject": RejectAnswer, "stop": StopAnswer}
    union = reduce(operator.or_, (members[answer] for answer in request["answers"]))
    return Annotated[union, Discriminator("type")]


def read_answer_message(answer: AuditAnswer) -> str | None:
    """Return the message a rejection or a stop carries, if any."""
    match answer:
        case {"message": str(message)}:
            return message
        case _:
            return None


def read_held_explanation(audit: Audit) -> str:
    """Return the monitor's explanation of the held action's score."""
    scores = [
        proposal.score
        for proposal in audit.proposals
        if proposal.action == audit.action and proposal.score is not None
    ]
    return scores[0].explanation if scores else ""


def build_edit(held: HeldStep, *, edited: Sequence[EditedToolCall]) -> AnswerCommit:
    """Return the commit of an edit: the held reply with the edited calls, as a trusted proposal.

    The reply keeps its id and each call its id, so the reply is replaced in
    place and each call answered under the id the model gave it. The action
    takes a fresh id, so the held action keeps its own, and its sample.
    """
    calls = [
        ToolCall(name=call["name"], args=call["args"], id=held_call["id"], type="tool_call")
        for call, held_call in zip(edited, held.reply.tool_calls, strict=True)
    ]
    message = held.reply.model_copy(update={"tool_calls": calls})
    action = replace(read_proposed_action(message), id=str(uuid4()))
    proposal = Proposal(action=action, score=None, policy=Policy.TRUSTED)
    audit = held.audit
    decision = Replace(
        action=action,
        proposals=(*audit.proposals, proposal),
        position=audit.position,
    )
    return AnswerCommit(decision=decision, messages=(message,), ran=message)


def build_rejection(
    held: HeldStep,
    *,
    message: str | None,
    feedback: FeedbackTemplate,
) -> AnswerCommit:
    """Return the commit of a rejection: a block, whose feedback answers the held reply.

    The reason is the person's message, or else the monitor's explanation, in
    the feedback template.
    """
    audit = held.audit
    reason = feedback.render(reason=message or read_held_explanation(audit))
    block = Block(
        action=audit.action,
        reason=reason,
        proposals=audit.proposals,
        position=audit.position,
    )
    answers = build_feedback_messages(held.reply, feedback=reason)
    return AnswerCommit(decision=block, messages=tuple(answers), route="model")


def build_stop(held: HeldStep, *, message: str | None) -> AnswerCommit:
    """Return the commit of a stop: a halt, the held reply removed, and the halt message.

    The halt's reason is the person's message, or else `DEFAULT_HALT_MESSAGE`.
    """
    audit = held.audit
    halt = Halt(
        reason=message or DEFAULT_HALT_MESSAGE,
        proposals=audit.proposals,
        position=audit.position,
    )
    removal = RemoveMessage(id=held.reply.id or "")
    return AnswerCommit(
        decision=halt,
        messages=(removal, build_halt_message(DEFAULT_HALT_MESSAGE)),
        route="end",
    )


def decide_answer(
    held: HeldStep,
    *,
    answer: AuditAnswer,
    feedback: FeedbackTemplate,
) -> AnswerCommit:
    """Return what a person's answer commits, at the audit's position."""
    audit = held.audit
    match answer["type"]:
        case "approve":
            run = Run(action=audit.action, proposals=audit.proposals, position=audit.position)
            return AnswerCommit(decision=run, ran=held.reply)
        case "edit":
            return build_edit(held, edited=answer["edited_tool_calls"])
        case "reject":
            message = read_answer_message(answer)
            return build_rejection(held, message=message, feedback=feedback)
        case "stop":
            return build_stop(held, message=read_answer_message(answer))
        case unreachable:
            assert_never(unreachable)


def commit_answer(
    held: HeldStep,
    *,
    commit: AnswerCommit,
    state: Mapping[str, object],
    middleware: MonitorMiddleware,
) -> AnsweredAudit:
    """Record the answer's decision, and return the update with the step's record.

    What follows a reply that runs, its server tools' results, follows its
    action in the trajectory, and every message committed is marked seen.
    """
    loaded = held.loaded
    loaded.trajectory.record(commit.decision)
    if commit.ran is not None:
        add_reply_results(loaded, messages=[commit.ran])
    mark_committed(loaded, committed=commit.messages)
    record = held.identity.build_record(
        (*held.decisions, commit.decision),
        flag_threshold=middleware.flag_threshold,
    )
    update: AgentStateUpdate = {
        MONITOR_LOG_KEY: [record],
        **build_halt_inputs_update(record, state=state, middleware_name=middleware.name),
        **build_trajectory_update(loaded, answered_lost_writes=0),
    }
    if commit.messages:
        update["messages"] = list(commit.messages)
    if commit.route is not None:
        update["jump_to"] = commit.route
    return AnsweredAudit(update=update, record=record)


def resolve_held_step(
    held: HeldStep,
    *,
    state: Mapping[str, object],
    runtime: object,
    middleware: MonitorMiddleware,
) -> AnsweredAudit:
    """Pause the run for a person's answer to the held step, then apply it.

    Nothing after `interrupt` returns raises on an answer, since LangGraph
    keeps an answer it returned, and would hand it back on every resume.
    """
    request = build_audit_request(held)
    schema = cast("type[AuditAnswer]", build_answer_schema(request))
    answer = interrupt(request, response_schema=schema)
    commit = decide_answer(held, answer=answer, feedback=middleware.feedback)
    answered = commit_answer(held, commit=commit, state=state, middleware=middleware)
    write_stream_event(runtime, event=MonitorStepEvent(type="monitor_step", record=answered.record))
    return answered


def find_reply_answers(reply: AIMessage, *, messages: Sequence[BaseMessage]) -> list[str]:
    """Return the ids of the tool messages that answer one of the reply's calls."""
    call_ids = {call["id"] for call in (*reply.tool_calls, *reply.invalid_tool_calls)}
    return [
        message.id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id in call_ids and message.id
    ]


def stop_held_step(
    held: HeldStep,
    *,
    state: Mapping[str, object],
    runtime: object,
    middleware: MonitorMiddleware,
) -> AgentStateUpdate:
    """Return the update a run starts with when its input stops a held step.

    The halt is recorded before the input is read, so its position is the
    audit's. The held reply and the tool messages answering its calls are
    left out of what the run reads and removed, and the paused run counts as
    ended at the audit, so the new input is recorded as the run's input,
    after the run inputs the halt counts.
    """
    messages = read_state_messages(state)
    removed = {held.reply.id, *find_reply_answers(held.reply, messages=messages)}
    kept = [message for message in messages if message.id not in removed]
    stopped = {**state, "messages": kept, RUN_OPEN_KEY: False}
    audit = held.audit
    halt = Halt(reason=NEW_INPUT_STOP_REASON, proposals=audit.proposals, position=audit.position)
    record = held.identity.build_record(
        (*held.decisions, halt),
        flag_threshold=middleware.flag_threshold,
    )
    update = build_run_start_update(stopped)
    trajectories = build_run_start_trajectories_update(
        stopped,
        update=update,
        label=middleware.label,
        task_author=middleware.task_author,
        stop=halt,
    )
    written = update.get("messages")
    # An `Overwrite` writes the history back without them already.
    if not isinstance(written, Overwrite):
        removals = [RemoveMessage(id=message_id) for message_id in removed if message_id]
        update["messages"] = [*removals, *(written or [])]
    write_stream_event(runtime, event=MonitorStepEvent(type="monitor_step", record=record))
    return {
        **update,
        **trajectories,
        MONITOR_LOG_KEY: [record],
        **build_halt_inputs_update(record, state=stopped, middleware_name=middleware.name),
    }
