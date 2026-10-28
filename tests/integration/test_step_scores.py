"""A monitored agent's highest and lowest suspicion reach LangSmith and Langfuse, per step.

The run is traced to LangSmith through its tracer on a client that reaches a
stand-in server, and to Langfuse through the SDK's own handler on a client
with a stand-in wire. Each is run under `invoke` and `ainvoke`. The keyword
monitor scores a step that posts to attacker.example 0.9, "mentions
attacker.example", and any other 0.1, "nothing suspicious".
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import tracing_context

from langchain_sync_monitors import AuditResponse, middleware, score_export
from langchain_sync_monitors.errors import MonitorError
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import DeferToResample
from langchain_sync_monitors.score_export import Tracer, read_project_id
from tests.support.agents import (
    STOP_ANSWER,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_task_input,
    run_agent,
    run_stopping_audits,
    send_audit_answer,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.score_backends import (
    NOT_A_STEP,
    NUMERIC_METADATA,
    PROJECT_ID,
    PROJECT_NAME,
    STEP_SPAN_NAME,
    Backend,
    FakeLangfuseHandler,
    LangfuseStandIn,
    LangSmithBackend,
    LangSmithServer,
    OtelModeClient,
    Sent,
    build_langfuse,
    build_langsmith_tracer,
)
from tests.support.tracing import RecordingTracer
from tests.unit.middleware.test_standing_halts import ReturningToModelMiddleware

EXFILTRATION_REASON = "mentions attacker.example"
BENIGN_REASON = "nothing suspicious"
RESAMPLING = DeferToResample(defer_threshold=0.6, audit_threshold=0.95, max_resamples=2)
"""A step at 0.6 or more is drawn again, so the exfiltration is followed by the read."""
FLAGGING = DeferToResample(
    defer_threshold=0.8,
    audit_threshold=0.8,
    max_resamples=0,
    on_audit=AuditResponse.FLAG,
)
"""Trusted monitoring at 0.8: every step runs, and one at 0.8 or above is flagged."""
AUDITING = DeferToResample(defer_threshold=0.5, audit_threshold=1.0, max_resamples=0)
"""A step at 0.5 or more with no resample left is audited, which pauses the run for a person."""
HELD_SENT = {
    (1, "monitor_suspicion"): [(0.9, EXFILTRATION_REASON)],
    (1, "monitor_lowest_suspicion"): [(0.9, EXFILTRATION_REASON)],
}
"""What a step held for a person has sent when the run pauses: its two scores, no flag."""
SECOND_STEP_SENT = {
    (2, "monitor_suspicion"): [(0.1, BENIGN_REASON)],
    (2, "monitor_lowest_suspicion"): [(0.1, BENIGN_REASON)],
}
"""What the step that answers "Done." sends."""
STOP_FLAG_SENT = {(NOT_A_STEP, "flagged"): [(True, None)]}
"""The flag a person's stop sends, on the answer's decision span, which is no step span."""
AFTER_MODEL_HOOK = "monitor[main].after_model"
"""The run of the hook that applies a person's answer."""


def build_monitor(
    backend: Backend,
    *,
    protocol: DeferToResample,
    flag_threshold: float | None = None,
) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=protocol,
        flag_threshold=flag_threshold,
        export_scores=frozenset({backend.tracer_member}),
    )


def build_agent(
    *middleware: AgentMiddleware[Any, Any, Any],
    steps: list[AIMessage],
    checkpointer: InMemorySaver | None = None,
) -> Runnable[Any, Any]:
    return create_agent(
        model=ScriptedChatModel(responses=steps),
        tools=Workspace().build_tools(),
        middleware=list(middleware),
        checkpointer=checkpointer,
    )


def run_traced(
    agent: Runnable[Any, Any],
    *,
    mode: RunMode,
    handlers: list[BaseCallbackHandler],
) -> dict[str, Any]:
    return run_agent(agent, mode=mode, config=RunnableConfig(callbacks=handlers))


def build_held_agent(
    backend: Backend,
    *extra: AgentMiddleware[Any, Any, Any],
    flag_threshold: float | None = None,
) -> Runnable[Any, Any]:
    """Build an agent that pauses at its first step, the exfiltration, then answers "Done."."""
    return build_agent(
        *extra,
        build_monitor(backend, protocol=AUDITING, flag_threshold=flag_threshold),
        steps=[build_exfiltration_step(), AIMessage("Done.")],
        checkpointer=InMemorySaver(),
    )


def build_held_config(backend: Backend) -> RunnableConfig:
    return RunnableConfig(
        callbacks=backend.build_handlers(),
        configurable={"thread_id": "held"},
    )


def test_a_step_sends_its_highest_and_lowest_suspicion_on_its_step_run(
    run_mode: RunMode,
    backend: Backend,
) -> None:
    # Arrange: the first step is drawn twice, 0.9 and then 0.1, and the second once, 0.1
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: the explanation is the comment, and the metadata says nothing failed closed
    assert backend.read_sent() == {
        (1, "monitor_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (1, "monitor_lowest_suspicion"): [(0.1, BENIGN_REASON)],
        (2, "monitor_suspicion"): [(0.1, BENIGN_REASON)],
        (2, "monitor_lowest_suspicion"): [(0.1, BENIGN_REASON)],
    }
    assert backend.read_metadata() == [NUMERIC_METADATA] * 4


@pytest.mark.parametrize(
    "flag_threshold",
    [None, 0.95, 0.9, 0.5],
    ids=["no-threshold", "above-the-lowest", "at-the-lowest", "below-the-lowest"],
)
def test_a_flagged_step_is_marked_once_and_an_unflagged_step_not_at_all(
    run_mode: RunMode,
    backend: Backend,
    flag_threshold: float | None,
) -> None:
    # Arrange: the exfiltration scores 0.9 and is flagged by the protocol; the answer scores 0.1
    agent = build_agent(
        build_monitor(backend, protocol=FLAGGING, flag_threshold=flag_threshold),
        steps=[build_exfiltration_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: a threshold of 0.9 or 0.5 flags by the report's own rule, and 0.95 or none by the
    # call to flag, and either way once
    assert backend.read_sent() == {
        (1, "monitor_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (1, "monitor_lowest_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (1, "flagged"): [(True, None)],
        (2, "monitor_suspicion"): [(0.1, BENIGN_REASON)],
        (2, "monitor_lowest_suspicion"): [(0.1, BENIGN_REASON)],
    }


def test_a_step_held_for_a_person_sends_its_scores_before_the_person_answers(
    run_mode: RunMode,
    backend: Backend,
) -> None:
    # Arrange
    agent = build_held_agent(backend)

    # Act: the run pauses at the audit
    paused = run_agent(agent, mode=run_mode, config=build_held_config(backend))

    # Assert: the person has not answered
    assert len(paused["__interrupt__"]) == 1
    assert backend.read_sent() == HELD_SENT


@pytest.mark.parametrize(
    ("answer", "sent_after"),
    [
        ({"type": "approve"}, HELD_SENT | SECOND_STEP_SENT),
        (STOP_ANSWER, HELD_SENT | STOP_FLAG_SENT),
    ],
    ids=["approved", "stopped"],
)
def test_the_answer_to_a_held_step_sends_no_second_set_of_its_scores(
    run_mode: RunMode,
    backend: Backend,
    answer: dict[str, str],
    sent_after: Sent,
) -> None:
    # Arrange
    agent = build_held_agent(backend)
    config = build_held_config(backend)
    paused = run_agent(agent, mode=run_mode, config=config)

    # Act
    send_audit_answer(agent, paused, answer=answer, mode=run_mode, config=config)

    # Assert: an approved run goes on to its next step, whose scores are its own, and a stop is
    # flagged without a score
    assert backend.read_sent() == sent_after


@pytest.mark.parametrize(
    ("answer", "flag_threshold", "places"),
    [
        pytest.param(
            {"type": "approve"},
            None,
            [],
            id="approved",
        ),
        pytest.param(
            STOP_ANSWER,
            None,
            [("monitor decision", AFTER_MODEL_HOOK)],
            id="stopped",
        ),
        pytest.param(
            STOP_ANSWER,
            0.9,
            [(STEP_SPAN_NAME, "model")],
            id="stopped-after-a-flag-at-the-pause",
        ),
    ],
)
def test_a_stopped_step_is_flagged_once_and_an_approved_one_not_at_all(
    run_mode: RunMode,
    backend: Backend,
    answer: dict[str, str],
    flag_threshold: float | None,
    places: list[tuple[str, str]],
) -> None:
    # Arrange: a threshold of 0.9 flags the held step at the pause, by its lowest score
    agent = build_held_agent(backend, flag_threshold=flag_threshold)
    config = build_held_config(backend)
    paused = run_agent(agent, mode=run_mode, config=config)

    # Act
    send_audit_answer(agent, paused, answer=answer, mode=run_mode, config=config)

    # Assert: a stop is flagged on the decision span of its answer, unless the pause flagged it
    assert backend.read_flag_places() == places


def test_a_step_halted_without_a_sample_is_marked_and_has_no_suspicion_score(
    run_mode: RunMode,
    backend: Backend,
) -> None:
    # Arrange: the first step is held and stopped, and the two steps a returning hook asks for
    # halt without a sample
    agent = build_held_agent(backend, ReturningToModelMiddleware(returns=2))
    config = build_held_config(backend)

    # Act
    run_stopping_audits(agent, build_task_input(), mode=run_mode, config=config)

    # Assert: the held step sent its scores at the pause, and its stop is flagged on the answer
    assert backend.read_sent() == HELD_SENT | STOP_FLAG_SENT | {
        (2, "flagged"): [(True, None)],
        (3, "flagged"): [(True, None)],
    }


def test_langsmith_in_opentelemetry_mode_gets_nothing_and_a_warning_once(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    inline_pool: object,
    fresh_notices: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    tracer = build_langsmith_tracer(langsmith, client_class=OtelModeClient)
    backend = LangSmithBackend(
        tracer_member=Tracer.LANGSMITH,
        server=langsmith,
        tracer=tracer,
        recorder=RecordingTracer(),
    )
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    with caplog.at_level(logging.WARNING):
        run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert
    assert langsmith.read_requests("GET", "/sessions") == []
    assert langsmith.read_feedback() == []
    assert sum("OpenTelemetry" in record.getMessage() for record in caplog.records) == 1


def test_a_send_that_fails_is_logged_by_type_and_never_reaches_the_run(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    inline_pool: object,
    fresh_notices: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: LangSmith refuses every feedback
    langsmith.feedback_status = 400
    backend = LangSmithBackend(
        tracer_member=Tracer.LANGSMITH,
        server=langsmith,
        tracer=build_langsmith_tracer(langsmith),
        recorder=RecordingTracer(),
    )
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    with caplog.at_level(logging.WARNING):
        result = run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: one failed send per step, named by its type, and no text of the transcript
    assert result["messages"][-1].text == "Done."
    failures = [
        record.getMessage()
        for record in caplog.records
        if record.name == "langchain_sync_monitors.request_pool"
    ]
    assert failures == ["score export: a request failed with LangSmithError"] * 2
    assert EXFILTRATION_REASON not in caplog.text


def test_the_step_does_not_wait_for_langsmith_to_answer(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    fresh_notices: None,
) -> None:
    # Arrange: LangSmith holds every feedback answer until released
    langsmith.release.clear()
    backend = LangSmithBackend(
        tracer_member=Tracer.LANGSMITH,
        server=langsmith,
        tracer=build_langsmith_tracer(langsmith),
        recorder=RecordingTracer(),
    )
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=backend.build_handlers())
    answered_when_the_run_returned = list(langsmith.answered)
    langsmith.release.set()
    answered = langsmith.wait_for_answers(4)

    # Assert: the run returned before any answer, and all four scores were written after it
    assert answered_when_the_run_returned == []
    assert answered == [200] * 4


def test_langfuse_in_a_run_with_tracing_off_gets_nothing_and_no_warning(
    run_mode: RunMode,
    fresh_notices: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the client traces nothing, so its handler holds a span with no ids
    with build_langfuse(tracing_enabled=False) as stand_in:
        monitor = MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=RESAMPLING,
            export_scores=frozenset({Tracer.LANGFUSE}),
        )
        agent = build_agent(
            monitor,
            steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
        )

        # Act
        with caplog.at_level(logging.WARNING):
            run_traced(agent, mode=run_mode, handlers=[stand_in.build_handler()])

        # Assert
        assert stand_in.read_scores() == []
        assert [record for record in caplog.records if "score export" in record.getMessage()] == []


def test_a_langfuse_handler_of_another_shape_gets_nothing_and_a_warning_once(
    run_mode: RunMode,
    fresh_notices: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    pytest.importorskip("langfuse")
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=RESAMPLING,
        export_scores=frozenset({Tracer.LANGFUSE}),
    )
    agent = build_agent(
        monitor,
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    with caplog.at_level(logging.WARNING):
        result = run_traced(agent, mode=run_mode, handlers=[FakeLangfuseHandler()])

    # Assert: two steps, one warning
    assert len(result["monitor_log"]) == 2
    warnings = [record for record in caplog.records if "score export" in record.getMessage()]
    assert len(warnings) == 1


def test_only_the_tools_asked_for_receive_scores(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    langfuse: LangfuseStandIn,
    inline_pool: object,
    fresh_notices: None,
) -> None:
    # Arrange: the run is traced to both tools, and the monitor exports to Langfuse alone
    handlers = [build_langsmith_tracer(langsmith), langfuse.build_handler()]
    monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=RESAMPLING,
        export_scores=frozenset({Tracer.LANGFUSE}),
    )
    agent = build_agent(
        monitor,
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=handlers)

    # Assert
    assert langsmith.read_feedback() == []
    assert len(langfuse.read_scores()) == 4


def test_a_run_with_langsmith_tracing_turned_off_gets_no_langsmith_score(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    inline_pool: object,
    fresh_notices: None,
) -> None:
    # Arrange
    backend = LangSmithBackend(
        tracer_member=Tracer.LANGSMITH,
        server=langsmith,
        tracer=build_langsmith_tracer(langsmith),
        recorder=RecordingTracer(),
    )
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    with tracing_context(enabled=False):
        result = run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert
    assert len(result["monitor_log"]) == 2
    assert langsmith.read_requests("GET", "/sessions") == []
    assert langsmith.read_feedback() == []


def test_stacked_monitors_send_their_scores_under_their_own_labels(
    run_mode: RunMode,
    backend: Backend,
) -> None:
    # Arrange: the outer monitor's step span opens first and the inner one's inside it
    protocol = DeferToResample(defer_threshold=0.95, audit_threshold=0.95, max_resamples=0)
    agent = build_agent(
        MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=protocol,
            label="outer",
            export_scores=frozenset({backend.tracer_member}),
        ),
        MonitorMiddleware(
            monitor=build_keyword_monitor(),
            protocol=protocol,
            label="inner",
            export_scores=frozenset({backend.tracer_member}),
        ),
        steps=[build_exfiltration_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: the step spans in the order they start are outer 1, inner 1, outer 2, inner 2
    assert backend.read_sent() == {
        (1, "outer_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (1, "outer_lowest_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (2, "inner_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (2, "inner_lowest_suspicion"): [(0.9, EXFILTRATION_REASON)],
        (3, "outer_suspicion"): [(0.1, BENIGN_REASON)],
        (3, "outer_lowest_suspicion"): [(0.1, BENIGN_REASON)],
        (4, "inner_suspicion"): [(0.1, BENIGN_REASON)],
        (4, "inner_lowest_suspicion"): [(0.1, BENIGN_REASON)],
    }


def test_a_step_whose_commit_fails_sends_no_scores(
    run_mode: RunMode,
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the commit raises, as a state the update cannot read would make it
    def fail_to_commit(*args: object, **kwargs: object) -> object:
        message = "the commit failed"
        raise MonitorError(message)

    monkeypatch.setattr(middleware, "commit_step", fail_to_commit)
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[AIMessage("Done.")],
    )

    # Act
    with pytest.raises(MonitorError, match="the commit failed"):
        run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: a step that is not in monitor_log gets no score in the tracing tool
    assert backend.read_sent() == {}


def test_a_project_is_looked_up_once_for_all_the_steps_of_a_tracer(
    run_mode: RunMode,
    langsmith: LangSmithServer,
    inline_pool: object,
    fresh_notices: None,
) -> None:
    # Arrange
    backend = LangSmithBackend(
        tracer_member=Tracer.LANGSMITH,
        server=langsmith,
        tracer=build_langsmith_tracer(langsmith),
        recorder=RecordingTracer(),
    )
    agent = build_agent(
        build_monitor(backend, protocol=RESAMPLING),
        steps=[build_exfiltration_step(), build_read_step(), AIMessage("Done.")],
    )

    # Act
    run_traced(agent, mode=run_mode, handlers=backend.build_handlers())

    # Assert: two steps sent four scores, all on the project looked up, which was asked for once
    assert [item["session_id"] for item in langsmith.read_feedback()] == [PROJECT_ID] * 4
    assert len(langsmith.read_requests("GET", "/sessions")) == 1


def test_the_remembered_project_ids_are_bounded_and_the_oldest_goes_first(
    langsmith: LangSmithServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: 65 clients have each been asked for the project
    monkeypatch.setattr(score_export, "LANGSMITH_PROJECT_IDS", OrderedDict())
    clients = [build_langsmith_tracer(langsmith).client for _ in range(65)]
    for client in clients:
        read_project_id(client, project_name=PROJECT_NAME)
    asked = len(langsmith.read_requests("GET", "/sessions"))

    # Act
    read_project_id(clients[-1], project_name=PROJECT_NAME)
    asked_after_the_newest = len(langsmith.read_requests("GET", "/sessions"))
    read_project_id(clients[0], project_name=PROJECT_NAME)
    asked_after_the_oldest = len(langsmith.read_requests("GET", "/sessions"))

    # Assert: the newest was remembered, and the oldest had gone
    assert (asked, asked_after_the_newest, asked_after_the_oldest) == (65, 65, 66)
