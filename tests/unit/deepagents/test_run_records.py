"""Each decision in a run takes the run's next sequence number, and subagents' blocks count.

A subagent records into its own trajectory, joined into the run its parent
handed it, so its records take the numbers that follow the run's, and keep
them when its trajectory joins the parent's under the call that started it.
Subagents run at once each count from the run as it was handed to them, and
the parent's join numbers the second after the first. A middleware listed
before the monitor reads each subagent's trajectory as its call returns it,
before the parent's join renumbers it. Block and feed back
counts the blocks of the whole run, so a subagent's blocks bring its parent
nearer the total.
"""

from __future__ import annotations

from typing import Any

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_sync_monitors import (
    DEFAULT_HALT_MESSAGE,
    BlockAndFeedBack,
    MonitorMiddleware,
    TaskAuthor,
)
from tests.support.agents import (
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_keyword_monitor,
    build_read_step,
    build_task_input,
    build_thread_config,
    read_texts,
    run_agent,
    run_stopping_audits,
    summarise_records,
)
from tests.support.chat_models import ScriptedChatModel
from tests.support.deep_agents import (
    build_deep_agent,
    build_delegation_step,
    build_parallel_delegation,
)
from tests.support.protocols import RUN_EVERY_STEP

type Agent = CompiledStateGraph[Any, Any, Any, Any]
type StoredTrajectory = dict[str, Any]


class ReturnedTrajectories(AgentMiddleware[Any, Any, Any]):
    """Keeps the trajectory each tool call returns under the monitor's label, by call id."""

    def __init__(self, *, agent_name: str) -> None:
        self.agent_name = agent_name
        self.returned: dict[str, StoredTrajectory] = {}

    @property
    def name(self) -> str:
        return f"returned[{self.agent_name}]"

    def keep(self, request: ToolCallRequest, *, result: Any) -> Any:
        if isinstance(result, Command) and isinstance(result.update, dict):
            state = result.update.get("monitor_trajectory", {}).get("monitor")
            if state is not None:
                self.returned[str(request.tool_call["id"])] = state["trajectory"]
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        return self.keep(request, result=handler(request))

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        return self.keep(request, result=await handler(request))


def build_subagent_monitor(agent_name: str) -> MonitorMiddleware:
    return MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=RUN_EVERY_STEP,
        agent_name=agent_name,
        task_author=TaskAuthor.PARENT_AGENT,
    )


def build_reader(name: str, *, answer: str) -> CompiledSubAgent:
    """Build a compiled subagent that reads q3.md, then answers: two steps."""
    model = ScriptedChatModel(responses=[build_read_step(), AIMessage(answer)])
    runnable = create_agent(
        model,
        tools=Workspace().build_tools(),
        middleware=[build_subagent_monitor(name)],
    )
    return CompiledSubAgent(name=name, description=f"The {name}.", runnable=runnable)


def read_root(agent: Agent, *, thread_id: str) -> StoredTrajectory:
    """Return the main agent's trajectory as the checkpoint stores it."""
    values = agent.get_state(build_thread_config(thread_id)).values
    return values["monitor_trajectory"]["monitor"]["trajectory"]


def read_sequence(stored: StoredTrajectory) -> list[int]:
    """Return the sequence number of each of a stored trajectory's records, in order."""
    return [record["sequence"] for record in stored["records"]]


def test_a_nested_subagent_s_two_steps_keep_the_run_s_sequence_numbers(
    run_mode: RunMode,
) -> None:
    # Arrange: main delegates to middle, which delegates to inner; each takes two steps
    returned_to_main = ReturnedTrajectories(agent_name="main")
    returned_to_middle = ReturnedTrajectories(agent_name="middle")
    middle_stack: list[AgentMiddleware[Any, Any, Any]] = [
        returned_to_middle,
        build_subagent_monitor("middle"),
    ]
    main_stack: list[AgentMiddleware[Any, Any, Any]] = [
        returned_to_main,
        MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP),
    ]
    middle = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(call_id="call-task-middle", subagent_type="inner"),
                AIMessage("Middle done."),
            ],
        ),
        middleware=middle_stack,
        subagents=[build_reader("inner", answer="Inner done.")],
    )
    agent = create_deep_agent(
        model=ScriptedChatModel(
            responses=[
                build_delegation_step(call_id="call-task-main", subagent_type="middle"),
                AIMessage("Done."),
            ],
        ),
        middleware=main_stack,
        subagents=[CompiledSubAgent(name="middle", description="Delegates.", runnable=middle)],
        checkpointer=InMemorySaver(),
    )
    thread_id = f"nested-sequence-{run_mode}"

    # Act
    run_agent(agent, mode=run_mode, config=build_thread_config(thread_id))

    # Assert: each subagent returns the numbers it recorded, which the joins keep
    root = read_root(agent, thread_id=thread_id)
    middle_run = root["subagents"]["call-task-main"]
    assert read_sequence(returned_to_middle.returned["call-task-middle"]) == [2, 3]
    assert read_sequence(returned_to_main.returned["call-task-main"]) == [1, 4]
    assert read_sequence(root) == [0, 5]
    assert read_sequence(middle_run) == [1, 4]
    assert read_sequence(middle_run["subagents"]["call-task-middle"]) == [2, 3]


def test_subagents_run_at_once_take_numbers_that_do_not_collide(run_mode: RunMode) -> None:
    # Arrange
    returned = ReturnedTrajectories(agent_name="main")
    stack: list[AgentMiddleware[Any, Any, Any]] = [
        returned,
        MonitorMiddleware(monitor=build_keyword_monitor(), protocol=RUN_EVERY_STEP),
    ]
    agent = create_deep_agent(
        model=ScriptedChatModel(responses=[build_parallel_delegation(), AIMessage("Done.")]),
        middleware=stack,
        subagents=[
            build_reader("worker", answer="Three sources."),
            build_reader("reviewer", answer="Sources check out."),
        ],
        checkpointer=InMemorySaver(),
    )
    thread_id = f"parallel-sequence-{run_mode}"

    # Act
    run_agent(agent, mode=run_mode, config=build_thread_config(thread_id))

    # Assert: both record 1 and 2 after main's 0; whichever joins first keeps them, the other
    # takes 3 and 4
    assert read_sequence(returned.returned["call-task-worker"]) == [1, 2]
    assert read_sequence(returned.returned["call-task-reviewer"]) == [1, 2]
    root = read_root(agent, thread_id=thread_id)
    worker = read_sequence(root["subagents"]["call-task-worker"])
    reviewer = read_sequence(root["subagents"]["call-task-reviewer"])
    assert read_sequence(root) == [0, 5]
    assert {tuple(worker), tuple(reviewer)} == {(1, 2), (3, 4)}


def test_a_subagent_s_block_counts_towards_its_parent_s_total(run_mode: RunMode) -> None:
    # Arrange: two blocks in total; the worker blocks once, then the main agent once
    main_monitor = MonitorMiddleware(
        monitor=build_keyword_monitor(),
        protocol=BlockAndFeedBack(block_threshold=0.5, blocks_in_total=2),
    )
    agent = build_deep_agent(
        main_model=ScriptedChatModel(
            responses=[build_delegation_step(), build_exfiltration_step(), AIMessage("Done.")],
        ),
        worker_model=ScriptedChatModel(
            responses=[build_exfiltration_step(), AIMessage("Three sources.")],
        ),
        main_monitor=main_monitor,
        checkpointer=InMemorySaver(),
    )

    # Act
    result = run_stopping_audits(
        agent,
        build_task_input(),
        mode=run_mode,
        config=build_thread_config("subagent-block"),
    )

    # Assert: counting its own block alone, the main agent would run "Done."; at the total
    # its proposal is audited, and stopped
    assert read_texts(result["messages"])[-1] == DEFAULT_HALT_MESSAGE
    assert summarise_records(result["monitor_log"]) == [
        ("main", "allowed", 0, None),
        ("worker", "steered", 1, "call-task"),
        ("main", "halted", 1, None),
    ]
