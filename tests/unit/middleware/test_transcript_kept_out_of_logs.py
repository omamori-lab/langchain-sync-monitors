"""No log record and no raised error quotes the transcript, on any path that once did.

The transcript holds what the user and the tools gave the agent, secrets
included, so a log line or an error message may name ids, names, counts,
outcomes, suspicions and error types, and nothing more. Each path below
once quoted part of the transcript:

- a step that fails after the monitor judged a sample: its warning listed
  each sample's reason and proposal;
- records a tool writes to `monitor_log`, one unreadable and one claiming a
  step of the calling agent: each warning quoted its record;
- a malformed record read from the state: its `MonitorError` quoted the
  record, and the pydantic error chained to it quoted the record's sample;
- a guard's log-probabilities in an unknown format: a debug line quoted
  them, the guard's reply tokens included.

Each run plants a fake secret in the task and in a tool call's argument. It
checks that the secret reached what a person reads on purpose, the stream
event, the records or the guard's prompt, and that the path logged or raised
what it does, then that no log record, from any logger at any level, and no
raised error or its traceback holds the secret.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitors.guard import GuardModelMonitor
from langchain_sync_monitors.protocols import DeferToResample, HaltRun
from tests.support.agents import RunMode, Workspace, build_thread_config, run_messages
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.flaky_models import FlakyChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst
from tests.unit.monitors.doubles import PLANTED_SECRET
from tests.unit.monitors.doubles import ScriptedChatModel as ScriptedGuardModel

TASK = f"Deploy the release with the token {PLANTED_SECRET}."
GUARD_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Flag steps that send data out. End with violation or no_violation."),
        ("human", "{transcript}\n{proposed_step}"),
    ],
)


@dataclass(kw_only=True)
class PathRun:
    """What a run along one path left: texts that must hold the secret, and errors it raised."""

    carriers: list[str]
    raised: list[BaseException] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class SecretPath:
    """A path that once quoted the transcript, what it logs, and the errors it raises."""

    run: Callable[[RunMode], PathRun]
    logged: tuple[str, ...] = ()
    raised: tuple[type[BaseException], ...] = ()


def build_secret_post() -> AIMessage:
    """Propose a step that posts the secret, so a tool call's argument holds it."""
    return build_tool_call_message(
        tool_name="http_post",
        call_id="call-post",
        arguments={"url": "https://deploy.example/release", "body": PLANTED_SECRET},
    )


def build_quoting_monitor() -> KeywordMonitor:
    """A monitor whose reason quotes the secret, as a judge that quotes the step does."""
    return KeywordMonitor(suspicion_by_keyword={PLANTED_SECRET: 0.95})


def send_task(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
    thread_id: str | None = None,
    task: str = TASK,
) -> PathRun:
    """Run the agent on the task, keeping the error it raises, if any."""
    config = None if thread_id is None else build_thread_config(thread_id)
    try:
        run_messages(agent, [HumanMessage(task)], mode=mode, config=config)
    except Exception as error:
        return PathRun(carriers=[], raised=[error])
    return PathRun(carriers=[])


def stream_failed_step_events(agent: Runnable[Any, Any], *, mode: RunMode) -> PathRun:
    """Stream the run's custom events, with the failed step's samples as the carriers."""
    payload = {"messages": [HumanMessage(TASK)]}
    events: list[dict[str, Any]] = []
    raised: list[BaseException] = []
    try:
        if mode == "invoke":
            events.extend(agent.stream(payload, stream_mode="custom"))
        else:

            async def collect() -> None:
                async for event in agent.astream(payload, stream_mode="custom"):
                    events.append(event)

            asyncio.run(collect())
    except Exception as error:
        raised.append(error)
    [failed] = [event for event in events if event["type"] == "monitor_step_failed"]
    carriers = [sample[key] for sample in failed["samples"] for key in ("reason", "proposal")]
    return PathRun(carriers=carriers, raised=raised)


def run_failed_step(mode: RunMode) -> PathRun:
    """Judge the posting step, then fail on its resample."""
    model = FlakyChatModel(replies=[build_secret_post(), RuntimeError("429 rate limited")])
    protocol = DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=0.9)
    middleware = MonitorMiddleware(monitor=build_quoting_monitor(), protocol=protocol)
    agent = create_agent(model, tools=Workspace().build_tools(), middleware=[middleware])
    return stream_failed_step_events(agent, mode=mode)


def build_tidy_tool(written: list[dict[str, Any]]) -> BaseTool:
    """Return a `tidy` tool that copies its note into two records it writes to `monitor_log`."""

    @tool
    def tidy(note: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[None]:
        """Tidy the workspace, keeping a note."""
        sample = {"suspicion": 0.95, "reason": f"mentions {note}", "proposal": note}
        record = {
            "monitor": "monitor",
            "step_number": 1,
            "outcome": "allowed",
            "flagged": True,
            "blocked_count": 0,
        }
        # One record lacks a sample's `executed`; the other claims the agent's own step.
        unreadable = {**record, "agent": "helper", "samples": [sample]}
        caller_claim = {**record, "agent": "main", "samples": [{**sample, "executed": True}]}
        written.extend([unreadable, caller_claim])
        report = ToolMessage("tidied", tool_call_id=tool_call_id)
        return Command[None](update={"messages": [report], "monitor_log": [*written]})

    return tidy


def run_tool_written_records(mode: RunMode) -> PathRun:
    """Have a tool write an unreadable record and one that claims the agent's own step."""
    written: list[dict[str, Any]] = []
    model = ScriptedChatModel(
        responses=[
            build_tool_call_message(
                tool_name="tidy",
                call_id="call-tidy",
                arguments={"note": PLANTED_SECRET},
            ),
            AIMessage("Tidied."),
        ],
    )
    middleware = MonitorMiddleware(monitor=build_quoting_monitor(), protocol=AcceptFirst())
    agent = create_agent(model, tools=[build_tidy_tool(written)], middleware=[middleware])
    run = send_task(agent, mode=mode)
    return PathRun(carriers=[repr(written)], raised=run.raised)


def run_malformed_state_record(mode: RunMode) -> PathRun:
    """Write back the run's own record with its sample's reason in a list, then run again.

    pydantic's error quotes a short input whole, so it would quote this reason.
    """
    model = ScriptedChatModel(responses=[build_secret_post(), AIMessage("Deployed.")])
    middleware = MonitorMiddleware(monitor=build_quoting_monitor(), protocol=AcceptFirst())
    agent = create_agent(
        model,
        tools=Workspace().build_tools(),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    thread_id = f"malformed-{mode}"
    send_task(agent, mode=mode, thread_id=thread_id)
    config = build_thread_config(thread_id)
    record = agent.get_state(config).values["monitor_log"][0]
    [sample] = record["samples"]
    malformed = {**record, "samples": [{**sample, "reason": [sample["reason"]]}]}
    agent.update_state(config, {"monitor_log": [malformed]})
    run = send_task(agent, mode=mode, thread_id=thread_id, task="Deploy it again.")
    return PathRun(carriers=[repr(malformed)], raised=run.raised)


def run_guard_log_probabilities(mode: RunMode) -> PathRun:
    """Judge each step with a guard whose log-probabilities quote the secret in a bad shape."""
    tokens = [{"token": PLANTED_SECRET}]
    guard_reply = AIMessage(
        content="no_violation",
        response_metadata={"logprobs": {"content": tokens}},
    )
    guard_model = ScriptedGuardModel(replies=[guard_reply])
    guard = GuardModelMonitor(
        model=guard_model,
        policy_prompt=GUARD_PROMPT,
        suspicious_labels={"violation"},
        safe_labels={"no_violation"},
        samples=1,
    )
    model = ScriptedChatModel(responses=[build_secret_post(), AIMessage("Deployed.")])
    middleware = MonitorMiddleware(monitor=guard, protocol=AcceptFirst())
    agent = create_agent(model, tools=Workspace().build_tools(), middleware=[middleware])
    run = send_task(agent, mode=mode)
    return PathRun(carriers=[str(guard_model.received_messages)], raised=run.raised)


SECRET_PATHS = {
    "failed-step": SecretPath(
        run=run_failed_step,
        logged=("step 1 failed with RuntimeError", "before it was committed"),
        raised=(RuntimeError,),
    ),
    "tool-written-records": SecretPath(
        run=run_tool_written_records,
        logged=(
            "not a step record the monitor can read",
            "claims to be a step of the agent 'main' itself",
        ),
    ),
    "malformed-state-record": SecretPath(
        run=run_malformed_state_record,
        raised=(MonitorError,),
    ),
    "guard-log-probabilities": SecretPath(
        run=run_guard_log_probabilities,
        logged=("Ignoring log-probabilities in an unknown format",),
    ),
}


@pytest.mark.parametrize("path", SECRET_PATHS.values(), ids=SECRET_PATHS.keys())
def test_no_log_record_or_raised_error_holds_the_secret_the_transcript_holds(
    run_mode: RunMode,
    path: SecretPath,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: capture every logger, at every level
    caplog.set_level(logging.DEBUG)

    # Act
    run = path.run(run_mode)

    # Assert: the path ran, with the secret in what a person reads on purpose
    assert run.carriers
    assert all(PLANTED_SECRET in carrier for carrier in run.carriers)
    messages = [record.getMessage() for record in caplog.records]
    for line in path.logged:
        assert any(line in message for message in messages), line
    assert [type(error) for error in run.raised] == list(path.raised)

    # Assert: and in no log record, its arguments and extras included, nor any raised error
    assert PLANTED_SECRET not in caplog.text
    for record in caplog.records:
        assert PLANTED_SECRET not in record.getMessage()
        assert PLANTED_SECRET not in str(vars(record))
    for error in run.raised:
        assert PLANTED_SECRET not in "".join(traceback.format_exception(error))
