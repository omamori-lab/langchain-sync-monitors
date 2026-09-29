"""The general-purpose subagent monitor_subagents adds keeps what Deep Agents' own would have."""

from __future__ import annotations

from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_sync_monitors.deepagents import monitor_subagents
from langchain_sync_monitors.middleware import MonitorMiddleware
from tests.support.agents import RunMode, run_agent
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import build_delegation_step
from tests.support.monitors import KeywordMonitor
from tests.support.protocols import AcceptFirst

SKILLS = ["/skills/"]
SKILL_NAME = "quarterly-report"
SKILL_FILE = f"""---
name: {SKILL_NAME}
description: Summarise a quarterly report for the team.
---
# Quarterly report

List the report's figures, then summarise them in one sentence.
"""
DELEGATED_TASK = "Summarise q3.md with the quarterly-report skill."


@pytest.fixture
def backend(tmp_path: Path) -> FilesystemBackend:
    skill_directory = tmp_path / "skills" / SKILL_NAME
    skill_directory.mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text(SKILL_FILE)
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def read_system_prompt(messages: list[BaseMessage]) -> str:
    [system_message] = [message for message in messages if isinstance(message, SystemMessage)]
    return system_message.text


def read_first_human_text(messages: list[BaseMessage]) -> str:
    return next(message.text for message in messages if isinstance(message, HumanMessage))


def test_the_general_purpose_subagent_keeps_the_main_agents_skills(
    run_mode: RunMode,
    backend: FilesystemBackend,
) -> None:
    # Arrange
    model = ScriptedChatModel(
        responses=[
            build_delegation_step(description=DELEGATED_TASK, subagent_type="general-purpose"),
            AIMessage("Revenue grew 12%."),
            AIMessage("Here is the summary."),
        ]
    )
    main_monitor = MonitorMiddleware(monitor=KeywordMonitor(), protocol=AcceptFirst())
    agent = create_deep_agent(
        model=model,
        middleware=[main_monitor],
        subagents=monitor_subagents(middleware=main_monitor, skills=SKILLS),
        skills=SKILLS,
        backend=backend,
    )

    # Act
    result = run_agent(agent, mode=run_mode)

    # Assert
    log = result["monitor_log"]
    assert [(record["agent"], record["step_number"]) for record in log] == [
        ("main", 1),
        ("general-purpose", 1),
        ("main", 2),
    ]
    [main_call, general_purpose_call, _final_call] = model.calls
    assert read_first_human_text(general_purpose_call) == DELEGATED_TASK
    assert SKILL_NAME in read_system_prompt(main_call)
    assert SKILL_NAME in read_system_prompt(general_purpose_call)
