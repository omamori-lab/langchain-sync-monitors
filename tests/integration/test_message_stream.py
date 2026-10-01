"""What a `stream_mode="messages"` consumer receives from a monitored agent.

Only committed steps stream. A sample the protocol rejects, a step it blocks
and the monitor's own model calls never reach the stream, and each committed
step still arrives, whole, once the model node returns it. The untrusted model
and the monitor's judge both stream token by token, as real providers do.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langchain_sync_monitors.contracts import (
    ControlProtocol,
    FeedbackVisibility,
    MonitorInput,
    Resampling,
    Verdict,
)
from langchain_sync_monitors.middleware import MonitorMiddleware
from langchain_sync_monitors.model_calls import build_internal_call_config
from langchain_sync_monitors.protocols import (
    AutoMode,
    DeferToResample,
    DeferToTrustedModel,
    HaltRun,
)
from tests.support.agents import (
    SUSPICION_BY_KEYWORD,
    RunMode,
    Workspace,
    build_exfiltration_step,
    build_read_step,
    stream_messages,
    stream_v3_messages,
)
from tests.support.chat_models import StreamingScriptedChatModel
from tests.support.monitors import KeywordMonitor

FINAL_ANSWER = "Q3 revenue grew 12%."
TRUSTED_ANSWER = "A careful trusted summary."
JUDGE_MARK = "Judge verdict"
JUDGE_REPLY = f"{JUDGE_MARK}: the step sends the key away."
REJECTED_STEP_MARK = "attacker.example"
FEEDBACK_MARK = "[Safety monitor]"


def build_judge() -> GenericFakeChatModel:
    """Return a judge that streams its reply token by token, as a real provider does."""
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(JUDGE_REPLY)))


@dataclass(kw_only=True)
class OwnTagsJudgeMonitor(KeywordMonitor):
    """A keyword monitor whose judge call sets tags of its own, as a user's monitor may.

    The call's own tags replace the `nostream` tag it would inherit, so only
    its internal-call metadata can keep the judge's reply out of the v3 event
    stream.
    """

    def build_judge_config(self) -> RunnableConfig:
        return RunnableConfig(
            **build_internal_call_config(source="own_tags_judge"),
            tags=["security-judge"],
        )

    async def evaluate(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            await self.judge_model.ainvoke("Judge this step.", config=self.build_judge_config())
        return self.score(monitor_input)

    def evaluate_sync(self, monitor_input: MonitorInput) -> Verdict:
        if self.judge_model is not None:
            self.judge_model.invoke("Judge this step.", config=self.build_judge_config())
        return self.score(monitor_input)


def build_monitor_middleware(
    *,
    protocol: ControlProtocol,
    feedback_visibility: FeedbackVisibility = FeedbackVisibility.HIDDEN,
    monitor_type: type[KeywordMonitor] = KeywordMonitor,
) -> MonitorMiddleware:
    """Build a keyword monitor whose every verdict also streams a judge reply, like a real one."""
    monitor = monitor_type(suspicion_by_keyword=SUSPICION_BY_KEYWORD, judge_model=build_judge())
    return MonitorMiddleware(
        monitor=monitor,
        protocol=protocol,
        feedback_visibility=feedback_visibility,
    )


def stream_monitored_agent(
    *,
    untrusted_steps: Sequence[AIMessage],
    protocol: ControlProtocol,
    run_mode: RunMode,
    feedback_visibility: FeedbackVisibility = FeedbackVisibility.HIDDEN,
) -> list[BaseMessage]:
    """Stream a scripted agent under the protocol and return what the consumer received."""
    agent = create_agent(
        model=StreamingScriptedChatModel(responses=list(untrusted_steps)),
        tools=Workspace().build_tools(),
        middleware=[
            build_monitor_middleware(protocol=protocol, feedback_visibility=feedback_visibility)
        ],
    )
    return stream_messages(agent, mode=run_mode)


def find_mentions(messages: Sequence[BaseMessage], *, text: str) -> list[BaseMessage]:
    """Return the streamed messages that mention the text anywhere, tool call arguments included."""
    return [message for message in messages if text in repr(message)]


def read_tool_call_names(messages: Sequence[BaseMessage]) -> list[str]:
    return [
        call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def read_answers(messages: Sequence[BaseMessage]) -> list[str]:
    return [
        message.text
        for message in messages
        if isinstance(message, AIMessage) and not message.tool_calls and message.text
    ]


@pytest.mark.parametrize(
    ("resampling", "resampled_reads"),
    [(Resampling.SEQUENTIAL, 1), (Resampling.PARALLEL, 2)],
    ids=["sequential", "parallel"],
)
def test_defer_to_resample_streams_only_the_committed_sample(
    run_mode: RunMode,
    resampling: Resampling,
    resampled_reads: int,
) -> None:
    # Arrange: parallel resampling draws both resamples at once, and commits one.
    protocol = DeferToResample(
        fallback=HaltRun(message="Stopped."),
        defer_threshold=0.6,
        audit_threshold=0.95,
        resampling=resampling,
    )
    reads = [build_read_step(call_id=f"call-read-{index}") for index in range(resampled_reads)]

    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(), *reads, AIMessage(FINAL_ANSWER)],
        protocol=protocol,
        run_mode=run_mode,
    )

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert find_mentions(streamed, text=JUDGE_MARK) == []
    assert read_tool_call_names(streamed) == ["read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER]


def test_the_trusted_step_streams_when_every_sample_is_rejected(run_mode: RunMode) -> None:
    # Arrange
    trusted_model = StreamingScriptedChatModel(responses=[AIMessage(TRUSTED_ANSWER)])
    protocol = DeferToResample(
        fallback=DeferToTrustedModel(trusted_model=trusted_model),
        defer_threshold=0.6,
        audit_threshold=0.85,
    )

    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(call_id=f"call-{index}") for index in range(3)],
        protocol=protocol,
        run_mode=run_mode,
    )

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert read_answers(streamed) == [TRUSTED_ANSWER]


def test_auto_mode_streams_neither_the_blocked_step_nor_hidden_feedback(run_mode: RunMode) -> None:
    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        protocol=AutoMode(block_threshold=0.6),
        run_mode=run_mode,
    )

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert find_mentions(streamed, text=FEEDBACK_MARK) == []
    assert read_tool_call_names(streamed) == ["read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER]


def test_feedback_in_the_transcript_streams_the_blocked_step_once_as_committed(
    run_mode: RunMode,
) -> None:
    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        protocol=AutoMode(block_threshold=0.6),
        run_mode=run_mode,
        feedback_visibility=FeedbackVisibility.IN_TRANSCRIPT,
    )

    # Assert
    blocked_steps = [
        message
        for message in find_mentions(streamed, text=REJECTED_STEP_MARK)
        if isinstance(message, AIMessage)
    ]
    rejections = [
        message.tool_call_id
        for message in streamed
        if isinstance(message, ToolMessage) and message.status == "error"
    ]
    assert len(blocked_steps) == 1
    assert (blocked_steps[0].id or "").startswith("monitor-")
    assert rejections == ["call-post"]
    assert read_tool_call_names(streamed) == ["http_post", "read_file"]


def test_a_rejected_subagent_step_never_streams(run_mode: RunMode) -> None:
    # Arrange
    pytest.importorskip("deepagents")
    from tests.support.deep_agents import build_deep_agent, build_delegation_step

    agent = build_deep_agent(
        main_model=StreamingScriptedChatModel(
            responses=[build_delegation_step(), AIMessage("The worker found the figure.")],
        ),
        worker_model=StreamingScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        main_monitor=build_monitor_middleware(protocol=AutoMode(block_threshold=0.6)),
        tools=Workspace().build_tools(),
    )

    # Act
    streamed = stream_messages(agent, mode=run_mode, subgraphs=True)

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert read_tool_call_names(streamed) == ["task", "read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER, "The worker found the figure."]


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
@pytest.mark.parametrize(
    "monitor_type",
    [KeywordMonitor, OwnTagsJudgeMonitor],
    ids=["judge-inherits-tags", "judge-sets-its-own-tags"],
)
def test_the_v3_event_stream_carries_only_committed_steps(
    run_mode: RunMode,
    monitor_type: type[KeywordMonitor],
) -> None:
    # Arrange
    agent = create_agent(
        model=StreamingScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        tools=Workspace().build_tools(),
        middleware=[
            build_monitor_middleware(
                protocol=AutoMode(block_threshold=0.6), monitor_type=monitor_type
            )
        ],
    )

    # Act
    streamed = stream_v3_messages(agent, mode=run_mode)

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert find_mentions(streamed, text=JUDGE_MARK) == []
    assert read_tool_call_names(streamed) == ["read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER]
