"""A monitor that scores with numpy's numbers still leaves records of plain values.

numpy is not a dependency, so small stand-ins play its scalars, as elsewhere
in the suite: `float64` is a `float` whose comparisons give numpy's `bool`,
which is not Python's, and `float32` is a real number that is not a `float`.
"""

from __future__ import annotations

import numbers
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_sync_monitors.contracts import ControlProtocol, Monitor, MonitorInput, Verdict
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.protocols import AutoMode, TrustedMonitoring
from tests.support.agents import RunMode, Workspace, build_read_step, build_thread_config, run_agent
from tests.support.chat_models import ScriptedChatModel

SCORE = 0.2


class ArrayBool:
    """Stands in for numpy's `bool`: it has a truth value, but is not Python's `bool`."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value


class ArrayFloat64(float):
    """Stands in for numpy's `float64`: a `float` whose comparisons give numpy's `bool`."""

    def __ge__(self, other: object) -> Any:
        return ArrayBool(float(self) >= cast("float", other))

    def __gt__(self, other: object) -> Any:
        return ArrayBool(float(self) > cast("float", other))

    def __le__(self, other: object) -> Any:
        return ArrayBool(float(self) <= cast("float", other))

    def __lt__(self, other: object) -> Any:
        return ArrayBool(float(self) < cast("float", other))


class ArrayFloat32:
    """Stands in for numpy's `float32`: a real number that is not a `float`."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    def __ge__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value >= cast("float", other))

    def __gt__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value > cast("float", other))

    def __le__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value <= cast("float", other))

    def __lt__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value < cast("float", other))


numbers.Real.register(ArrayFloat32)


class ArrayScoreMonitor(Monitor):
    """Scores every step 0.2, as a number of the given array type."""

    def __init__(self, score_type: type) -> None:
        self.score_type = score_type

    def build_verdict(self) -> Verdict:
        return Verdict(suspicion=cast("float", self.score_type(SCORE)), reason="classifier score")

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        return self.build_verdict()

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        return self.build_verdict()


SCORE_TYPES = {"float64": ArrayFloat64, "float32": ArrayFloat32}
PROTOCOLS = {
    "trusted-monitoring": lambda: TrustedMonitoring(flag_threshold=0.1),
    "auto-mode": lambda: AutoMode(block_threshold=0.5),
}


@pytest.mark.parametrize("checkpointed", [False, True], ids=["no-checkpointer", "checkpointer"])
@pytest.mark.parametrize("build_protocol", PROTOCOLS.values(), ids=PROTOCOLS.keys())
@pytest.mark.parametrize("score_type", SCORE_TYPES.values(), ids=SCORE_TYPES.keys())
def test_a_monitor_scoring_with_numpy_numbers_leaves_plain_records(
    run_mode: RunMode,
    score_type: type,
    build_protocol: Any,
    checkpointed: bool,
) -> None:
    # Arrange: the second step reads the first step's record back
    model = ScriptedChatModel(responses=[build_read_step(), AIMessage("Done.")])
    protocol: ControlProtocol = build_protocol()
    agent = create_agent(
        model,
        tools=Workspace().build_tools(),
        middleware=[MonitorMiddleware(monitor=ArrayScoreMonitor(score_type), protocol=protocol)],
        checkpointer=InMemorySaver() if checkpointed else None,
    )
    config = build_thread_config(f"array-{run_mode}") if checkpointed else None

    # Act
    result = run_agent(agent, mode=run_mode, config=config)

    # Assert
    assert result["messages"][-1].text == "Done."
    records = result["monitor_log"]
    assert [type(record["flagged"]) for record in records] == [bool, bool]
    assert [type(record["samples"][0]["suspicion"]) for record in records] == [float, float]
    assert records[0]["samples"][0]["suspicion"] == pytest.approx(SCORE)
    if config is not None:
        assert agent.get_state(config).values["monitor_log"] == records
