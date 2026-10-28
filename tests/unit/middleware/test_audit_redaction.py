"""A redaction on the middleware reaches a held step, a person's answer and the input that stops it.

The audit's request shows the held reply's calls as the model made them, so the
person sees what an approval runs, and the rated samples redacted, as
`monitor_log` holds them. What an answer adds to the monitor's trajectory, and
the new input that stops a held step, pass through the redaction as every
entry does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors import DeferToResample, FeedbackTemplate, MonitorMiddleware
from tests.support.agents import (
    RunMode,
    Workspace,
    build_thread_config,
    read_stored_trajectory,
    run_agent,
    send_audit_answer,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.monitors import KeywordMonitor
from tests.support.redaction import MASK, MASK_SECRET, SECRET, SUSPICION_BY_SECRET, TASK_WITH_SECRET

THREAD = build_thread_config("audit-redaction")
AUDIT_SUSPICIOUS = DeferToResample(defer_threshold=0.5, audit_threshold=0.5, max_resamples=0)
FEEDBACK = FeedbackTemplate(text="[Safety monitor] Blocked: {reason}")
FINAL_ANSWER = "Done."
HELD_ARGUMENTS = {"url": "https://notes.example/team", "body": f"key={SECRET}"}
HELD_CALL = {"name": "http_post", "args": HELD_ARGUMENTS, "id": "call-post", "type": "tool_call"}
MASKED_REASON = f"mentions {MASK}"
EDIT_WITH_SECRET = {
    "type": "edit",
    "edited_tool_calls": [
        {
            "name": "http_post",
            "args": {"url": "https://example.com/team", "body": f"note {SECRET}"},
        },
    ],
}
NEW_INPUT_WITH_SECRET = f"Only summarise it, key {SECRET}."


@dataclass(frozen=True, kw_only=True)
class PausedRun:
    """An agent paused at the audit of a step whose call and explanation hold the token."""

    agent: Any
    result: dict[str, Any]


@pytest.fixture
def paused_post(run_mode: RunMode) -> PausedRun:
    """Pause a run at the audit of a post that holds the token, which a final answer follows."""
    held = build_tool_call_message(
        tool_name="http_post",
        call_id="call-post",
        arguments=HELD_ARGUMENTS,
    )
    agent = create_agent(
        ScriptedChatModel(responses=[held, AIMessage(FINAL_ANSWER)]),
        tools=Workspace().build_tools(),
        middleware=[
            MonitorMiddleware(
                monitor=KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET),
                protocol=AUDIT_SUSPICIOUS,
                feedback=FEEDBACK,
                redaction=MASK_SECRET,
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    result = run_agent(agent, mode=run_mode, config=THREAD, task=TASK_WITH_SECRET)
    return PausedRun(agent=agent, result=result)


def test_an_audit_request_shows_the_held_calls_as_made_and_the_samples_redacted(
    paused_post: PausedRun,
) -> None:
    # Act
    [request] = [item.value for item in paused_post.result["__interrupt__"]]

    # Assert
    assert request["tool_calls"] == [HELD_CALL]
    assert SECRET not in json.dumps(request["samples"])
    assert [sample["reason"] for sample in request["samples"]] == [MASKED_REASON]
    assert MASK in json.dumps(request["samples"][0]["proposal"])


@pytest.mark.parametrize(
    ("answer", "kinds", "masked"),
    [
        pytest.param({"type": "approve"}, ["audit", "run", "run"], f"key={MASK}", id="approve"),
        pytest.param(
            EDIT_WITH_SECRET,
            ["audit", "replace", "run"],
            f"note {MASK}",
            id="edit",
        ),
        pytest.param(
            {"type": "reject", "message": f"Do not send {SECRET}."},
            ["audit", "block", "run"],
            f"Blocked: Do not send {MASK}.",
            id="reject",
        ),
        pytest.param(
            {"type": "stop", "message": f"Stopped: {SECRET}."},
            ["audit", "halt"],
            f"Stopped: {MASK}.",
            id="stop",
        ),
    ],
)
def test_an_answer_to_an_audit_is_recorded_without_the_text_the_redaction_removes(
    run_mode: RunMode,
    paused_post: PausedRun,
    answer: dict[str, Any],
    kinds: list[str],
    masked: str,
) -> None:
    # Act
    result = send_audit_answer(
        paused_post.agent,
        paused_post.result,
        answer=answer,
        mode=run_mode,
        config=THREAD,
    )

    # Assert
    stored = read_stored_trajectory(result)
    assert [record["kind"] for record in stored["records"]] == kinds
    assert SECRET not in json.dumps(stored)
    assert masked in json.dumps(stored)
    assert SECRET not in json.dumps(result["monitor_log"])


def test_a_rejection_without_a_message_feeds_the_agent_the_redacted_explanation(
    run_mode: RunMode,
    paused_post: PausedRun,
) -> None:
    # Act
    result = send_audit_answer(
        paused_post.agent,
        paused_post.result,
        answer={"type": "reject"},
        mode=run_mode,
        config=THREAD,
    )

    # Assert
    assert result["messages"][2].text == f"[Safety monitor] Blocked: mentions {MASK}"


def test_the_new_input_that_stops_a_held_step_is_recorded_without_the_text_the_redaction_removes(
    run_mode: RunMode,
    paused_post: PausedRun,
) -> None:
    # Act
    result = run_agent(
        paused_post.agent,
        mode=run_mode,
        config=THREAD,
        task=NEW_INPUT_WITH_SECRET,
    )

    # Assert
    stored = read_stored_trajectory(result)
    assert [record["kind"] for record in stored["records"]] == ["audit", "halt", "run"]
    assert SECRET not in json.dumps(stored)
    assert f"Only summarise it, key {MASK}." in json.dumps(stored)
    assert SECRET not in json.dumps(result["monitor_log"])
