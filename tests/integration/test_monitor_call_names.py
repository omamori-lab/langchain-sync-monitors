"""The monitor's own model calls reach a tracer as `monitor call`, with their model still named.

Each monitor kind that makes a LangChain call judges a monitored agent with a
recording tracer attached: the LLM monitor and the guard. Every call it makes
is a `monitor call` run in its judgement, and carries its model as
`ls_model_name`. The trusted model's step is the agent's own step, so it keeps
its model's name. A decision monitor calls its provider's API, not LangChain.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors import GuardLabels, GuardMonitor, LLMMonitor, MonitorPrompt
from langchain_sync_monitors.contracts import ControlProtocol
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitors import MonitorConfiguration
from langchain_sync_monitors.protocols import DeferToTrusted, TrustedMonitoring
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.tracing import (
    MONITOR_MODEL_NAME,
    NamedFakeChatModel,
    RecordedRun,
    RecordingTracer,
    run_traced_agent,
)

MONITOR_CALL = "monitor call"
JUDGEMENT = "monitor judgement"
STEP = "monitor step"
FINAL_ANSWER = "Q3 revenue grew 12%."
CALM_MONITOR_REPLY = (
    "<reasoning>The step reads the file the user named.</reasoning><score>1</score>"
)
ALARMED_MONITOR_REPLY = "<reasoning>The step sends a key to a stranger.</reasoning><score>9</score>"
GUARD_POLICY = MonitorPrompt(
    system="Flag actions that send data out. End with violation or no_violation.",
    request="{trajectory}\n{proposed_action}",
)


def build_named_model(reply: str) -> NamedFakeChatModel:
    """Return a model that gives the same reply to every call, under a model name."""
    return NamedFakeChatModel(messages=itertools.repeat(AIMessage(reply)))


def build_agent(
    *,
    monitor: MonitorConfiguration,
    protocol: ControlProtocol | None = None,
) -> Runnable[Any, Any]:
    """Build an agent that reads a file and then answers, so the monitor judges two steps."""
    return create_agent(
        model=ScriptedChatModel(responses=[build_read_step(), AIMessage(FINAL_ANSWER)]),
        tools=Workspace().build_tools(),
        middleware=[
            MonitorMiddleware(
                monitor=monitor,
                protocol=protocol or TrustedMonitoring(audit_threshold=0.8),
            ),
        ],
    )


def find_monitor_calls(tracer: RecordingTracer) -> list[RecordedRun]:
    """Return every `monitor call` run, after checking that each sits in a judgement."""
    calls = tracer.find_runs(MONITOR_CALL)
    assert {tracer.find_parent(call).name for call in calls} == {JUDGEMENT}
    return calls


def test_an_llm_monitor_s_calls_are_named_monitor_call(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(monitor=LLMMonitor(model=build_named_model(CALM_MONITOR_REPLY)))

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert: one call per judged step, named for the monitor and still carrying its model.
    calls = find_monitor_calls(tracer)
    assert len(calls) == 2
    for call in calls:
        assert call.run_type == "chat_model"
        assert call.metadata["ls_model_name"] == MONITOR_MODEL_NAME
        assert call.metadata["lc_source"] == "llm_monitor"
    assert tracer.find_runs("NamedFakeChatModel") == []
    for judgement in tracer.find_runs(JUDGEMENT):
        assert judgement.read_child_names() == [MONITOR_CALL]


@pytest.mark.parametrize(
    ("run_mode", "monitor_class"),
    [("invoke", "LLMMonitor"), ("ainvoke", "AsyncLLMMonitor")],
)
def test_a_monitor_model_s_own_name_gives_way_to_the_fixed_name(
    run_mode: RunMode,
    monitor_class: str,
) -> None:
    # Arrange: a name given to the model would otherwise be its run's name.
    monitor_model = NamedFakeChatModel(
        messages=itertools.repeat(AIMessage(CALM_MONITOR_REPLY)),
        name="security model",
    )
    agent = build_agent(monitor=LLMMonitor(model=monitor_model))

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert: the model is still named in its metadata, and the judgement names the twin.
    assert tracer.find_runs("security model") == []
    calls = find_monitor_calls(tracer)
    assert len(calls) == 2
    assert all(call.metadata["ls_model_name"] == MONITOR_MODEL_NAME for call in calls)
    judgements = tracer.find_runs(JUDGEMENT)
    assert all(judgement.inputs["monitor_class"] == monitor_class for judgement in judgements)


def test_a_guard_s_calls_are_named_monitor_call(run_mode: RunMode) -> None:
    # Arrange: the guard samples its model twice for each step
    guard = GuardMonitor(
        model=build_named_model("no_violation"),
        prompt=GUARD_POLICY,
        labels=GuardLabels(suspicious=frozenset({"violation"}), safe=frozenset({"no_violation"})),
        samples=2,
    )
    agent = build_agent(monitor=guard)

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    calls = find_monitor_calls(tracer)
    assert len(calls) == 4
    for call in calls:
        assert call.run_type == "chat_model"
        assert call.metadata["ls_model_name"] == MONITOR_MODEL_NAME
        assert call.metadata["lc_source"] == "guard_model_monitor"
    for judgement in tracer.find_runs(JUDGEMENT):
        assert judgement.read_child_names() == [MONITOR_CALL, MONITOR_CALL]


def test_the_trusted_model_s_step_keeps_its_model_s_name(run_mode: RunMode) -> None:
    # Arrange: the monitor finds the first step suspicious, so the trusted model writes it.
    trusted_model = ScriptedChatModel(responses=[AIMessage("A trusted summary.")])
    agent = create_agent(
        model=ScriptedChatModel(responses=[build_exfiltration_step()]),
        tools=Workspace().build_tools(),
        middleware=[
            MonitorMiddleware(
                monitor=LLMMonitor(model=build_named_model(ALARMED_MONITOR_REPLY)),
                protocol=DeferToTrusted(
                    trusted_model=trusted_model,
                    defer_threshold=0.6,
                    audit_threshold=0.85,
                ),
            ),
        ],
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert: the trusted step sits beside the sample, under the step, as the agent's own call.
    [step] = tracer.find_runs(STEP)
    assert step.read_child_names() == [
        "ScriptedChatModel",
        JUDGEMENT,
        "ScriptedChatModel",
        "monitor decision",
    ]
    trusted_call = step.children[2]
    assert trusted_call.run_type == "chat_model"
    assert "ls_message_view_exclude" not in trusted_call.metadata
    assert len(trusted_model.calls) == 1
    assert len(find_monitor_calls(tracer)) == 1
