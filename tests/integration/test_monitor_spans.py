"""What a LangChain tracer receives from a monitored agent: a named span tree per step.

Every step is a `monitor step` span under the model node. The agent's samples
and a `monitor judgement` span per verdict nest in it, the judge's own calls
nest in their judgement, and a `monitor decision` span, tagged with the
outcome, closes the step. The tracer is a plain callback handler passed
through the run's config, as LangSmith's and Langfuse's are.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors.contracts import ControlProtocol, Resampling
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrusted,
    HaltRun,
    TrustedMonitoring,
)
from tests.support.agents import (
    SUSPICION_BY_KEYWORD,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.flaky_models import FlakyChatModel
from tests.support.monitors import KeywordMonitor
from tests.support.tracing import (
    RecordedRun,
    RecordingTracer,
    build_judge_model,
    record_started_chain_names,
    run_traced_agent,
)

FINAL_ANSWER = "Q3 revenue grew 12%."
EXFILTRATION_MARK = "attacker.example"
STEP = "monitor step"
JUDGEMENT = "monitor judgement"
DECISION = "monitor decision"
SAMPLE = "ScriptedChatModel"
JUDGE = "GenericFakeChatModel"
LABEL_KEYS = {
    "monitor_name",
    "monitor_agent",
    "monitor_step_number",
    "monitor_protocol",
    "monitor_step_id",
}

EXPECTED_TREE = """\
LangGraph [chain] ok
  model [chain] ok
    monitor step [chain] ok tags=['monitor']
      ScriptedChatModel [chat_model] ok
      monitor judgement [chain] ok tags=['monitor']
        GenericFakeChatModel [chat_model] ok
      ScriptedChatModel [chat_model] ok
      monitor judgement [chain] ok tags=['monitor']
        GenericFakeChatModel [chat_model] ok
      monitor decision [chain] ok tags=['monitor', 'monitor:resampled']
  monitor[main].after_model [chain] ok
  tools [chain] ok
    read_file [tool] ok
  model [chain] ok
    monitor step [chain] ok tags=['monitor']
      ScriptedChatModel [chat_model] ok
      monitor judgement [chain] ok tags=['monitor']
        GenericFakeChatModel [chat_model] ok
      monitor decision [chain] ok tags=['monitor', 'monitor:allowed']
  monitor[main].after_model [chain] ok"""


@dataclass(frozen=True, kw_only=True)
class ProtocolCase:
    """A protocol, the steps the untrusted model proposes, and the first step's expected spans."""

    build_protocol: Callable[[], ControlProtocol]
    untrusted_steps: Callable[[], list[AIMessage]]
    step_children: list[str]
    outcome: str
    decision_tags: list[str]
    blocked_count: int = 0


PROTOCOL_CASES = {
    "trusted_monitoring": ProtocolCase(
        build_protocol=lambda: TrustedMonitoring(flag_threshold=0.8),
        untrusted_steps=lambda: [build_exfiltration_step(), AIMessage(FINAL_ANSWER)],
        step_children=[SAMPLE, JUDGEMENT, DECISION],
        outcome="allowed",
        decision_tags=["monitor", "monitor:allowed", "monitor:flagged"],
    ),
    "defer_to_resample": ProtocolCase(
        build_protocol=lambda: DeferToResample(
            fallback=HaltRun(message="Stopped."), defer_threshold=0.6, audit_threshold=0.95
        ),
        untrusted_steps=lambda: [
            build_exfiltration_step(),
            build_read_step(),
            AIMessage(FINAL_ANSWER),
        ],
        step_children=[SAMPLE, JUDGEMENT, SAMPLE, JUDGEMENT, DECISION],
        outcome="resampled",
        decision_tags=["monitor", "monitor:resampled"],
    ),
    "defer_to_trusted": ProtocolCase(
        build_protocol=lambda: DeferToTrusted(
            trusted_model=ScriptedChatModel(responses=[AIMessage("A trusted summary.")]),
            defer_threshold=0.6,
            audit_threshold=0.85,
        ),
        untrusted_steps=lambda: [build_exfiltration_step()],
        step_children=[SAMPLE, JUDGEMENT, SAMPLE, DECISION],
        outcome="deferred_to_trusted",
        decision_tags=["monitor", "monitor:deferred_to_trusted", "monitor:flagged"],
    ),
    "auto_mode": ProtocolCase(
        build_protocol=lambda: AutoMode(block_threshold=0.6),
        untrusted_steps=lambda: [
            build_exfiltration_step(),
            build_read_step(),
            AIMessage(FINAL_ANSWER),
        ],
        step_children=[SAMPLE, JUDGEMENT, SAMPLE, JUDGEMENT, DECISION],
        outcome="steered",
        decision_tags=["monitor", "monitor:steered", "monitor:flagged"],
        blocked_count=1,
    ),
    "halt": ProtocolCase(
        build_protocol=lambda: DeferToResample(
            fallback=HaltRun(message="Stopped."), defer_threshold=0.6, audit_threshold=0.85
        ),
        untrusted_steps=lambda: [build_exfiltration_step(call_id=f"call-{n}") for n in range(3)],
        step_children=[SAMPLE, JUDGEMENT] * 3 + [DECISION],
        outcome="halted",
        decision_tags=["monitor", "monitor:halted", "monitor:flagged"],
    ),
}


def build_agent(
    *,
    protocol: ControlProtocol,
    untrusted_steps: Sequence[AIMessage],
) -> Runnable[Any, Any]:
    """Build an agent whose keyword monitor also calls a judge model for every verdict."""
    monitor = KeywordMonitor(
        suspicion_by_keyword=SUSPICION_BY_KEYWORD, judge_model=build_judge_model()
    )
    return create_agent(
        model=ScriptedChatModel(responses=list(untrusted_steps)),
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )


def build_defer_to_resample_agent(
    *,
    resampling: Resampling = Resampling.SEQUENTIAL,
    untrusted_steps: Sequence[AIMessage] | None = None,
) -> Runnable[Any, Any]:
    protocol = DeferToResample(
        fallback=HaltRun(message="Stopped."),
        defer_threshold=0.6,
        audit_threshold=0.95,
        resampling=resampling,
    )
    steps = untrusted_steps or [
        build_exfiltration_step(),
        build_read_step(),
        AIMessage(FINAL_ANSWER),
    ]
    return build_agent(protocol=protocol, untrusted_steps=steps)


def read_labels(run: RecordedRun) -> dict[str, Any]:
    return {key: value for key, value in run.metadata.items() if key.startswith("monitor_")}


def find_step_spans_with_descendants(tracer: RecordingTracer) -> list[RecordedRun]:
    """Return every run inside a step span, the step span included."""
    runs: list[RecordedRun] = []
    pending = tracer.find_runs(STEP)
    while pending:
        run = pending.pop()
        runs.append(run)
        pending.extend(run.children)
    return runs


@pytest.mark.parametrize("case", PROTOCOL_CASES.values(), ids=PROTOCOL_CASES.keys())
def test_each_protocol_s_step_is_a_span_tree_with_its_outcome(
    run_mode: RunMode,
    case: ProtocolCase,
) -> None:
    # Arrange
    protocol = case.build_protocol()
    agent = build_agent(protocol=protocol, untrusted_steps=case.untrusted_steps())

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    step = tracer.find_runs(STEP)[0]
    [decision] = step.find_children(DECISION)
    assert tracer.find_parent(step).name == "model"
    assert step.read_child_names() == case.step_children
    assert (step.error, step.outputs["outcome"], step.outputs["blocked_count"]) == (
        None,
        case.outcome,
        case.blocked_count,
    )
    assert EXFILTRATION_MARK in step.inputs["proposed_step"]
    assert step.metadata["monitor_protocol"] == type(protocol).__name__
    assert (decision.error, decision.tags) == (None, case.decision_tags)
    assert decision.outputs["outcome"] == decision.metadata["monitor_outcome"] == case.outcome
    assert decision.metadata["monitor_flagged"] is ("monitor:flagged" in case.decision_tags)
    assert tracer.find_unknown_parents() == []
    assert tracer.find_open_runs() == []


def test_the_span_tree_of_a_resampled_step_and_an_allowed_one(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    assert tracer.render_tree() == EXPECTED_TREE


def test_the_step_span_carries_the_proposed_step_and_the_decision(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    first, second = tracer.find_runs(STEP)
    assert first.inputs["step_number"] == 1
    assert EXFILTRATION_MARK in first.inputs["proposed_step"]
    assert first.outputs == {
        "outcome": "resampled",
        "flagged": False,
        "blocked_count": 0,
        "max_suspicion": 0.9,
        "samples": [
            {"suspicion": 0.9, "reason": f"mentions {EXFILTRATION_MARK}", "executed": False},
            {"suspicion": 0.1, "reason": "nothing suspicious", "executed": True},
        ],
    }
    assert second.inputs == {
        "step_number": 2,
        "proposed_step": f"<proposed_step>\n<agent>{FINAL_ANSWER}</agent>\n</proposed_step>",
    }


def test_the_decision_span_repeats_the_outcome_in_its_metadata(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    decision = tracer.find_runs(DECISION)[0]
    assert (decision.inputs, decision.outputs) == (
        {},
        {"outcome": "resampled", "flagged": False, "max_suspicion": 0.9},
    )
    assert decision.metadata["monitor_outcome"] == "resampled"
    assert decision.metadata["monitor_flagged"] is False
    assert decision.metadata["monitor_max_suspicion"] == 0.9


def test_judgement_spans_hold_the_verdict_and_no_proposal_text(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    first, second, _ = tracer.find_runs(JUDGEMENT)
    assert first.inputs == {"sample_number": 1, "monitor": "KeywordMonitor"}
    assert first.outputs == {"suspicion": 0.9, "reason": f"mentions {EXFILTRATION_MARK}"}
    assert second.inputs == {"sample_number": 2, "monitor": "KeywordMonitor"}
    assert [child.name for child in first.children] == [JUDGE]
    below_the_step = [*tracer.find_runs(JUDGEMENT), *tracer.find_runs(DECISION)]
    assert all(EXFILTRATION_MARK not in json.dumps(run.inputs) for run in below_the_step)


def test_every_monitor_span_names_its_step_in_flat_metadata(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    for step in tracer.find_runs(STEP):
        spans = [step, *(child for child in step.children if child.is_monitor_span)]
        labels = [read_labels(span) for span in spans]
        assert all(set(label) >= LABEL_KEYS for label in labels)
        assert {span_labels["monitor_step_id"] for span_labels in labels} == {str(step.run_id)}
        assert labels[0]["monitor_protocol"] == "DeferToResample"
        assert (labels[0]["monitor_name"], labels[0]["monitor_agent"]) == ("monitor", "main")
        assert "monitor_delegation_id" not in labels[0]
    assert [step.metadata["monitor_step_number"] for step in tracer.find_runs(STEP)] == [1, 2]


def test_only_the_spans_below_the_step_are_kept_out_of_the_trajectory_view(
    run_mode: RunMode,
) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    for span in tracer.find_monitor_spans():
        expected = None if span.name == STEP else "middleware"
        assert span.metadata.get("ls_agent_type") == expected
        assert "ls_message_view_exclude" not in span.metadata


def test_the_calls_inside_a_step_keep_their_own_tags_and_metadata(run_mode: RunMode) -> None:
    # Act
    _, tracer = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    calls = [run for run in find_step_spans_with_descendants(tracer) if not run.is_monitor_span]
    assert {call.name for call in calls} == {SAMPLE, JUDGE}
    for call in calls:
        assert call.tags == ["nostream"]
        assert read_labels(call) == {}
        assert "ls_agent_type" not in call.metadata
        assert ("ls_message_view_exclude" in call.metadata) == (call.name == JUDGE)


def test_parallel_resamples_each_get_their_own_judgement_under_the_step(
    run_mode: RunMode,
) -> None:
    # Arrange
    agent = build_defer_to_resample_agent(
        resampling=Resampling.PARALLEL,
        untrusted_steps=[
            build_exfiltration_step(),
            build_read_step(call_id="call-read-a"),
            build_read_step(call_id="call-read-b"),
            AIMessage(FINAL_ANSWER),
        ],
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    step = tracer.find_runs(STEP)[0]
    judgements = step.find_children(JUDGEMENT)
    assert sorted(step.read_child_names()) == sorted([SAMPLE] * 3 + [JUDGEMENT] * 3 + [DECISION])
    assert sorted(judgement.inputs["sample_number"] for judgement in judgements) == [1, 2, 3]
    assert all(judgement.read_child_names() == [JUDGE] for judgement in judgements)
    assert sorted(judgement.outputs["suspicion"] for judgement in judgements) == [0.1, 0.1, 0.9]
    assert tracer.find_unknown_parents() == []
    assert tracer.find_open_runs() == []


def build_read_file_step(path: str) -> AIMessage:
    return build_tool_call_message(
        tool_name="read_file",
        call_id=f"call-read-{path}",
        arguments={"path": path},
    )


def test_judgement_numbers_follow_the_order_parallel_draws_started(run_mode: RunMode) -> None:
    # Arrange: under ainvoke the n-th model call waits the n-th delay, so the
    # first resample started finishes last and the judgements start out of order.
    monitor = KeywordMonitor(
        suspicion_by_keyword={
            EXFILTRATION_MARK: 0.9,
            "one.md": 0.2,
            "two.md": 0.3,
            "three.md": 0.4,
        },
    )
    model = FlakyChatModel(
        replies=[
            build_exfiltration_step(),
            build_read_file_step("one.md"),
            build_read_file_step("two.md"),
            build_read_file_step("three.md"),
            AIMessage(FINAL_ANSWER),
        ],
        delays=[0.0, 0.09, 0.03, 0.06],
    )
    protocol = DeferToResample(
        fallback=HaltRun(message="Stopped."),
        defer_threshold=0.6,
        audit_threshold=0.95,
        resampling=Resampling.PARALLEL,
        max_resamples=3,
    )
    agent = create_agent(
        model=model,
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    step = tracer.find_runs(STEP)[0]
    judgements = step.find_children(JUDGEMENT)
    reasons_by_number = {
        judgement.inputs["sample_number"]: judgement.outputs["reason"] for judgement in judgements
    }
    assert sorted(reasons_by_number) == [1, 2, 3, 4]
    assert [reasons_by_number[number] for number in sorted(reasons_by_number)] == [
        sample["reason"] for sample in step.outputs["samples"]
    ]
    assert [sample["reason"] for sample in step.outputs["samples"]] == [
        f"mentions {keyword}" for keyword in (EXFILTRATION_MARK, "one.md", "two.md", "three.md")
    ]
    finish_order = [judgement.inputs["sample_number"] for judgement in judgements]
    assert finish_order == ([1, 3, 4, 2] if run_mode == "ainvoke" else [1, 2, 3, 4])


@pytest.fixture
def started_chain_names(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of every chain run a callback manager starts."""
    return record_started_chain_names(monkeypatch)


def test_without_a_tracer_no_span_is_started_and_the_run_is_the_same(
    run_mode: RunMode,
    started_chain_names: list[str],
) -> None:
    # Act
    untraced = run_agent(build_defer_to_resample_agent(), mode=run_mode)
    untraced_names = list(started_chain_names)
    traced, _ = run_traced_agent(build_defer_to_resample_agent(), mode=run_mode)

    # Assert
    assert "model" in untraced_names
    assert not any(name.startswith("monitor ") for name in untraced_names)
    assert untraced["monitor_log"] == traced["monitor_log"]
    assert [message.text for message in untraced["messages"]] == [
        message.text for message in traced["messages"]
    ]
