"""Inside a subagent only the delegated task speaks as the delegator, and nothing flows back."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_reading_monitor,
    build_thread_config,
    run_agent,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_deep_agent, build_delegation_step
from tests.support.monitors import RenderingMonitor, read_tagged_entries
from tests.support.written_human_messages import FRAMES_TEXT, attach_video, build_attach_step

TASK = "Summarise q3.md for the team."
DESCRIPTION = "Find the sources for q3.md."


def test_a_human_message_a_subagent_tool_writes_is_a_note_not_the_delegator(
    run_mode: RunMode,
) -> None:
    # Arrange
    worker_reader = RenderingMonitor()
    main_model = ScriptedChatModel(
        responses=[build_delegation_step(description=DESCRIPTION), AIMessage("Done.")],
    )
    worker_model = ScriptedChatModel(
        responses=[build_attach_step(), build_exfiltration_step(), AIMessage("Found them.")],
    )
    agent = build_deep_agent(
        main_model=main_model,
        worker_model=worker_model,
        main_monitor=build_reading_monitor(RenderingMonitor()),
        worker_monitor=build_reading_monitor(worker_reader),
        checkpointer=InMemorySaver(),
        tools=[attach_video, *Workspace().build_tools()],
    )
    config = build_thread_config(f"subagent-authorship-{run_mode}")

    # Act
    run_agent(agent, mode=run_mode, config=config, task=TASK)

    # Assert
    transcript = worker_reader.find_reading(tool_name="http_post").transcript
    assert read_tagged_entries(transcript, tag="delegator") == [DESCRIPTION]
    assert read_tagged_entries(transcript, tag="context_note") == [FRAMES_TEXT]
    state = agent.get_state(config).values
    task_message_id = state["messages"][0].id
    assert state["monitor_task_messages"] == [task_message_id]
    assert state["monitor_seen_human_messages"] == [task_message_id]
