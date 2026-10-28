"""A score's comment is the explanation the monitor's trajectory recorded, redacted.

The run is traced to LangSmith through its tracer on a client that reaches a
stand-in server, and to Langfuse through the SDK's own handler on a client
with a stand-in wire, each under `invoke` and `ainvoke`. The keyword monitor
scores the step whose post holds the planted token 0.9, "mentions SECRET-123",
and any other 0.1, "nothing suspicious".
"""

from __future__ import annotations

import pytest
from hyoshigi import Redaction
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, Workspace, run_agent
from tests.support.chat_models import ScriptedChatModel, build_tool_call_message
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import RUN_EVERY_STEP
from tests.support.redaction import MASK, MASK_SECRET, SECRET, SUSPICION_BY_SECRET
from tests.support.score_backends import Backend

BENIGN_REASON = "nothing suspicious"


def build_sent(reason: str) -> dict[tuple[int, str], list[tuple[float, str]]]:
    """Return what two steps send: the post rated 0.9 with this reason, then the answer."""
    return {
        (1, "monitor_suspicion"): [(0.9, reason)],
        (1, "monitor_lowest_suspicion"): [(0.9, reason)],
        (2, "monitor_suspicion"): [(0.1, BENIGN_REASON)],
        (2, "monitor_lowest_suspicion"): [(0.1, BENIGN_REASON)],
    }


@pytest.mark.parametrize(
    ("redaction", "reason"),
    [
        pytest.param(MASK_SECRET, f"mentions {MASK}", id="with-a-redaction"),
        pytest.param(None, f"mentions {SECRET}", id="without-one"),
    ],
)
def test_a_score_s_comment_is_the_explanation_the_redaction_leaves(
    run_mode: RunMode,
    backend: Backend,
    redaction: Redaction | None,
    reason: str,
) -> None:
    # Arrange
    post = build_tool_call_message(
        tool_name="http_post",
        call_id="call-post",
        arguments={"url": "https://notes.example/team", "body": f"key={SECRET}"},
    )
    agent = create_agent(
        model=ScriptedChatModel(responses=[post, AIMessage("Done.")]),
        tools=Workspace().build_tools(),
        middleware=[
            MonitorMiddleware(
                monitor=KeywordMonitor(suspicion_by_keyword=SUSPICION_BY_SECRET),
                protocol=RUN_EVERY_STEP,
                export_scores=frozenset({backend.tracer_member}),
                redaction=redaction,
            ),
        ],
    )

    # Act
    run_agent(
        agent,
        mode=run_mode,
        config=RunnableConfig(callbacks=backend.build_handlers()),
    )

    # Assert
    assert backend.read_sent() == build_sent(reason)
