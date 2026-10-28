"""At a block limit the next proposal is drawn and audited, and the audit halts the run.

Block and feed back counts the blocks in a row in the agent's own records,
and the blocks in total in its run's. Once either reaches its
limit, the next proposal is still drawn and rated, then held for a person,
and while no person can answer an audit, the step halts.
"""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from langchain_sync_monitors import DEFAULT_HALT_MESSAGE, BlockAndFeedBack, MonitorMiddleware
from tests.support.agents import (
    TASK,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    read_stored_trajectory,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel


@pytest.mark.xfail(strict=True, reason="PR 5 audits the proposal after the third block")
@pytest.mark.parametrize(
    "protocol",
    [
        BlockAndFeedBack(block_threshold=0.5, blocks_in_a_row=3, blocks_in_total=20),
        BlockAndFeedBack(block_threshold=0.5, blocks_in_a_row=20, blocks_in_total=3),
    ],
    ids=["three-in-a-row", "three-in-total"],
)
def test_after_three_blocks_the_next_proposal_is_drawn_audited_and_halted(
    run_mode: RunMode,
    protocol: BlockAndFeedBack,
) -> None:
    # Arrange: every proposal is suspicious; the answer after them is never drawn
    workspace = Workspace()
    model = ScriptedChatModel(
        responses=[
            build_exfiltration_step(call_id="call-post-1"),
            build_exfiltration_step(call_id="call-post-2"),
            build_exfiltration_step(call_id="call-post-3"),
            build_exfiltration_step(call_id="call-post-4"),
            AIMessage("Done."),
        ],
    )
    middleware = MonitorMiddleware(monitor=build_keyword_monitor(), protocol=protocol)
    agent = create_agent(model, tools=workspace.build_tools(), middleware=[middleware])

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    assert len(model.calls) == 4
    assert workspace.executed == []
    assert result["messages"][0].text == TASK
    assert result["messages"][-1].text == DEFAULT_HALT_MESSAGE
    [record] = result["monitor_log"]
    assert (record["outcome"], record["blocked_count"], record["flagged"]) == ("halted", 3, True)
    assert [sample["suspicion"] for sample in record["samples"]] == [0.9, 0.9, 0.9, 0.9]
    assert [entry["kind"] for entry in read_stored_trajectory(result)["records"]] == [
        "block",
        "block",
        "block",
        "audit",
        "halt",
    ]
