"""A Decisions API request is one `monitor classifier` span in its judgement, around its retries.

The agent is judged by a `DecisionModelMonitor` on `OpenRouterDecisionModel`,
whose requests go to a fake Decisions API through `httpx.MockTransport`. The
fake answers by what the proposed step says, not by the order of requests,
so samples drawn at once get the same answers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import stamina
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import SecretStr

from langchain_sync_monitors.contracts import ControlProtocol
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.monitors.decision import DecisionModelMonitor
from langchain_sync_monitors.monitors.decision_questions import IS_THE_STEP_SUSPICIOUS
from langchain_sync_monitors.monitors.openrouter_decisions import OpenRouterDecisionModel
from langchain_sync_monitors.protocols import DeferToResample, HaltRun, TrustedMonitoring
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.tracing import RecordingTracer, run_traced_agent

EXFILTRATION_MARK = "attacker.example"
FINAL_ANSWER = "Q3 revenue grew 12%."
DECISION_MODEL = "typesafe/jev"
CLASSIFIER = "monitor classifier"


class DecisionsServer:
    """A fake Decisions API that answers 0.9 for a step sending data away, 0.05 otherwise.

    `failures` are answered first, one per request, before any real answer.
    """

    def __init__(self, *, failures: list[int] | None = None) -> None:
        self.failures = list(failures or [])
        self.requests: list[dict[str, Any]] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.failures:
            return httpx.Response(self.failures.pop(0), json={"error": "unavailable"})
        proposed_step = body["state"][body["state"].rfind("<proposed_step") :]
        probability = 0.9 if EXFILTRATION_MARK in proposed_step else 0.05
        answers = {key: {"type": "noul", "noul": probability} for key in body["questions"]}
        return httpx.Response(200, json={"answers": answers, "model": body["model"]})


@pytest.fixture(autouse=True)
def stamina_without_waiting() -> Iterator[None]:
    """Retry the Decisions API without backoff, up to three attempts."""
    with stamina.set_testing(True, attempts=3):
        yield


def build_agent(
    server: DecisionsServer,
    *,
    protocol: ControlProtocol,
    untrusted_steps: list[AIMessage],
) -> Runnable[Any, Any]:
    transport = httpx.MockTransport(server.respond)
    decision_model = OpenRouterDecisionModel(
        model=DECISION_MODEL,
        api_key=SecretStr("test-key"),
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
    )
    monitor = DecisionModelMonitor(decision_model=decision_model)
    return create_agent(
        model=ScriptedChatModel(responses=untrusted_steps),
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=monitor, protocol=protocol)],
    )


def test_each_decisions_request_is_a_classifier_span_in_its_judgement(run_mode: RunMode) -> None:
    # Arrange
    agent = build_agent(
        DecisionsServer(),
        protocol=DeferToResample(fallback=HaltRun(), defer_threshold=0.6, audit_threshold=0.95),
        untrusted_steps=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    classifiers = tracer.find_runs(CLASSIFIER)
    first_step = tracer.find_runs("monitor step")[0]
    assert [tracer.find_parent(span).name for span in classifiers] == ["monitor judgement"] * 3
    assert [span.outputs for span in classifiers] == [
        {"answers": {"suspicious_step": 0.9}},
        {"answers": {"suspicious_step": 0.05}},
        {"answers": {"suspicious_step": 0.05}},
    ]
    first = classifiers[0]
    assert first.inputs == {
        "model": DECISION_MODEL,
        "questions": {IS_THE_STEP_SUSPICIOUS.key: IS_THE_STEP_SUSPICIOUS.text},
    }
    assert (first.run_type, first.tags, first.children) == ("chain", ["monitor"], [])
    assert first.metadata["ls_agent_type"] == "middleware"
    assert first.metadata["monitor_step_id"] == str(first_step.run_id)
    assert first.metadata["monitor_protocol"] == "DeferToResample"
    assert all(EXFILTRATION_MARK not in json.dumps(span.inputs) for span in classifiers)
    assert tracer.find_open_runs() == []


def test_retries_stay_inside_one_classifier_span(run_mode: RunMode) -> None:
    # Arrange
    server = DecisionsServer(failures=[503])
    agent = build_agent(
        server,
        protocol=TrustedMonitoring(flag_threshold=0.8),
        untrusted_steps=[build_read_step(), AIMessage(FINAL_ANSWER)],
    )

    # Act
    _, tracer = run_traced_agent(agent, mode=run_mode)

    # Assert
    assert len(server.requests) == 3
    classifiers = tracer.find_runs(CLASSIFIER)
    assert len(classifiers) == 2
    assert all(span.error is None and span.ended for span in classifiers)


def test_a_request_that_fails_ends_the_classifier_its_judgement_and_the_step(
    run_mode: RunMode,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    agent = build_agent(
        DecisionsServer(failures=[400]),
        protocol=TrustedMonitoring(flag_threshold=0.8),
        untrusted_steps=[build_read_step()],
    )

    # Act
    with pytest.raises(httpx.HTTPStatusError):
        run_agent(agent, mode=run_mode, config=RunnableConfig(callbacks=[tracer]))

    # Assert
    [classifier] = tracer.find_runs(CLASSIFIER)
    [judgement] = tracer.find_runs("monitor judgement")
    [step] = tracer.find_runs("monitor step")
    assert tracer.find_parent(classifier) is judgement
    assert all(
        isinstance(span.error, httpx.HTTPStatusError) for span in (classifier, judgement, step)
    )
    assert tracer.find_open_runs() == []
