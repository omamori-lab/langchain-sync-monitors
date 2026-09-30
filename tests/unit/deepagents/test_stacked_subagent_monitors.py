"""monitor_subagents warns when the monitor it adds sits inside one a spec already has.

A subagent's monitor goes after the spec's own middleware, so a monitor the
spec already holds wraps it, with the same effect on the records as two
monitors in one `create_agent` list.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from deepagents import SubAgent, create_deep_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from langchain_sync_monitors.contracts import ControlProtocol
from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.placement import MonitorPlacementWarning, check_monitor_placement
from langchain_sync_monitors.protocols import AutoMode, DeferToResample, HaltRun, TrustedMonitoring
from tests.support.agents import RunMode, build_exfiltration_step, build_keyword_monitor, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step
from tests.support.monitors import KeywordMonitor

THRESHOLD = 0.5
WORKER_REPORT = "Three sources found."


def build_resampling(*, max_resamples: int) -> ControlProtocol:
    return DeferToResample(
        fallback=HaltRun(),
        defer_threshold=THRESHOLD,
        audit_threshold=THRESHOLD,
        max_resamples=max_resamples,
    )


def build_trusted_monitoring_monitor(monitor: KeywordMonitor | None = None) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=monitor or KeywordMonitor(),
        protocol=TrustedMonitoring(flag_threshold=THRESHOLD),
    )


def build_guarded_worker(
    guard_protocol: ControlProtocol,
    *,
    model: ScriptedChatModel | None = None,
) -> SubAgent:
    """Build a worker spec that already holds a monitor of its own, labelled `guard`."""
    guard = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=guard_protocol,
        label="guard",
        agent_name="worker",
    )
    spec_middleware: list[AgentMiddleware[Any, Any, Any]] = [guard]
    worker = SubAgent(name="worker", description="Finds sources.", middleware=spec_middleware)
    if model is not None:
        worker["model"] = model
    return worker


def test_a_monitor_added_inside_a_spec_s_resampling_monitor_is_named(
    run_mode: RunMode,
) -> None:
    # Arrange: the guard rejects the worker's first draft, which the added monitor allows
    inner_monitor = KeywordMonitor()
    worker_model = ScriptedChatModel(
        responses=[build_exfiltration_step(), AIMessage(WORKER_REPORT)],
    )
    worker = build_guarded_worker(build_resampling(max_resamples=2), model=worker_model)
    with pytest.warns(MonitorPlacementWarning, match="more than once in a step") as caught:
        subagents = monitor_subagents(
            middleware=build_trusted_monitoring_monitor(inner_monitor),
            subagents=[worker],
        )
    agent = create_deep_agent(
        model=ScriptedChatModel(responses=[build_delegation_step(), AIMessage("Done.")]),
        middleware=[build_trusted_monitoring_monitor()],
        subagents=subagents,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the warning is right, since the added monitor keeps one of its two judgements
    [message] = [str(warning.message) for warning in caught]
    assert message.startswith("monitor[worker] sits inside guard[worker] (DeferToResample)")
    assert caught[0].filename == __file__
    [inner_record] = [
        record
        for record in result["monitor_log"]
        if (record["agent"], record["monitor"]) == ("worker", "monitor")
    ]
    assert len(inner_monitor.inputs) == 2
    assert len(inner_record["samples"]) == 1


@pytest.mark.parametrize("given_as", ["middleware", "override"])
def test_a_monitor_given_either_way_is_named_inside_a_spec_s_resampling_monitor(
    given_as: str,
) -> None:
    # Arrange
    worker = build_guarded_worker(build_resampling(max_resamples=1))
    added = build_trusted_monitoring_monitor()
    overrides = {"worker": added} if given_as == "override" else None
    middleware = build_trusted_monitoring_monitor() if given_as == "override" else added

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        monitor_subagents(middleware=middleware, subagents=[worker], overrides=overrides)

    # Assert
    assert [str(warning.message).split(" ")[0] for warning in caught] == ["monitor[worker]"]


def test_monitoring_its_own_output_again_names_each_monitor_added_inside() -> None:
    # Arrange: the first call gives every spec a resampling monitor of its own
    guard = MonitorMiddleware(
        monitor=build_keyword_monitor(), protocol=build_resampling(max_resamples=2), label="guard"
    )
    worker = SubAgent(name="worker", description="Finds sources.")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monitored = monitor_subagents(middleware=guard, subagents=[worker])

    # Act
    with pytest.warns(MonitorPlacementWarning) as caught:
        monitor_subagents(middleware=build_trusted_monitoring_monitor(), subagents=monitored)

    # Assert
    assert [str(warning.message).split(" ")[0] for warning in caught] == [
        "monitor[worker]",
        "monitor[general-purpose]",
    ]


@pytest.mark.parametrize(
    "guard_protocol",
    [TrustedMonitoring(flag_threshold=THRESHOLD), build_resampling(max_resamples=0)],
    ids=["trusted-monitoring", "resample-none-then-halt"],
)
def test_a_monitor_added_inside_a_spec_s_single_call_monitor_is_not_named(
    guard_protocol: ControlProtocol,
) -> None:
    # Arrange
    worker = build_guarded_worker(guard_protocol)

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        specs = monitor_subagents(middleware=build_trusted_monitoring_monitor(), subagents=[worker])

    # Assert
    worker_middleware = specs[0].get("middleware", [])
    assert [item.name for item in worker_middleware] == ["guard[worker]", "monitor[worker]"]


def test_subagent_blocks_at_the_total_leave_a_monitor_inside_auto_mode_sound(
    run_mode: RunMode,
) -> None:
    # Arrange: Auto Mode allows one block in the thread; the worker's block reaches it, so
    # the main agent's next step halts without a sample
    inner_monitor = KeywordMonitor()
    outer = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=AutoMode(block_threshold=THRESHOLD, max_consecutive_blocks=3, max_total_blocks=1),
        label="outer",
    )
    inner = MonitorMiddleware(
        monitor=inner_monitor,
        protocol=TrustedMonitoring(flag_threshold=THRESHOLD),
        label="inner",
    )
    worker = SubAgent(
        name="worker",
        description="Finds sources.",
        model=ScriptedChatModel(responses=[build_exfiltration_step()]),
    )
    main_model = ScriptedChatModel(responses=[build_delegation_step()])
    agent = create_deep_agent(
        model=main_model,
        middleware=[outer, inner],
        subagents=monitor_subagents(middleware=outer, subagents=[worker]),
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert: the main agent drew one sample, and the inner monitor recorded it
    main_records = [record for record in result["monitor_log"] if record["agent"] == "main"]
    outcomes = [(record["monitor"], record["outcome"]) for record in main_records]
    assert outcomes == [("inner", "allowed"), ("outer", "allowed"), ("outer", "halted")]
    assert len(main_model.calls) == 1
    inner_records = [record for record in main_records if record["monitor"] == "inner"]
    assert sum(len(record["samples"]) for record in inner_records) == len(inner_monitor.inputs)
    assert check_monitor_placement(middleware=[outer, inner]) == []


def test_a_spec_s_other_middleware_is_left_to_check_monitor_placement() -> None:
    # Arrange: a tool retry around the added monitor, which the full check warns about
    worker = SubAgent(
        name="worker", description="Finds sources.", middleware=[ToolRetryMiddleware()]
    )
    monitor = build_trusted_monitoring_monitor()

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        specs = monitor_subagents(middleware=monitor, subagents=[worker])

    # Assert
    with pytest.warns(MonitorPlacementWarning, match="failed tool calls"):
        check_monitor_placement(middleware=specs[0].get("middleware", []))
