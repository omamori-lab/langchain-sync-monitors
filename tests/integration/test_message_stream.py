"""What a `stream_mode="messages"` consumer receives from a monitored agent.

Only committed steps stream. A sample the protocol rejects and the monitor's
own model calls never reach the stream, a blocked step streams once, as the
step commits it with its feedback, and each committed step arrives, whole,
once the model node returns it. The untrusted model
and the monitor's model both stream token by token, as real providers do.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langchain_sync_monitors import BlockAndFeedBack, DeferToResample, Resampling
from langchain_sync_monitors.middleware import MonitorMiddleware
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
MONITOR_REPLY_MARK = "Judge verdict"
MONITOR_REPLY = f"{MONITOR_REPLY_MARK}: the step sends the key away."
REJECTED_STEP_MARK = "attacker.example"
RESAMPLE = DeferToResample(defer_threshold=0.6, audit_threshold=0.95, max_resamples=2)
"""Resamples a step at or above 0.6, so the exfiltration step is rejected unseen."""


def build_monitor_model() -> GenericFakeChatModel:
    """Return a monitor's model that streams its reply token by token, as a real provider does."""
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(MONITOR_REPLY)))


def build_monitor_middleware(
    *,
    protocol: DeferToResample | BlockAndFeedBack,
    trusted_model: BaseChatModel | None = None,
    call_tags: tuple[str, ...] = (),
) -> MonitorMiddleware:
    """Build a keyword monitor whose every rating also streams a model reply, like a real one.

    With `call_tags`, the monitor's call sets tags of its own, as a user's
    monitor may: they replace the `nostream` tag it would inherit, so only its
    internal-call metadata can keep the model's reply out of the v3 event stream.
    """
    monitor = KeywordMonitor(
        suspicion_by_keyword=SUSPICION_BY_KEYWORD,
        model=build_monitor_model(),
        call_tags=call_tags,
    )
    return MonitorMiddleware(monitor=monitor, protocol=protocol, trusted_model=trusted_model)


def stream_monitored_agent(
    *,
    untrusted_steps: Sequence[AIMessage],
    protocol: DeferToResample | BlockAndFeedBack,
    run_mode: RunMode,
    trusted_model: BaseChatModel | None = None,
) -> list[BaseMessage]:
    """Stream a scripted agent under the protocol and return what the consumer received."""
    agent = create_agent(
        model=StreamingScriptedChatModel(responses=list(untrusted_steps)),
        tools=Workspace().build_tools(),
        middleware=[build_monitor_middleware(protocol=protocol, trusted_model=trusted_model)],
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
        defer_threshold=0.6,
        audit_threshold=0.95,
        max_resamples=2,
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
    assert find_mentions(streamed, text=MONITOR_REPLY_MARK) == []
    assert read_tool_call_names(streamed) == ["read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER]


def test_the_trusted_step_streams_when_every_sample_is_rejected(run_mode: RunMode) -> None:
    # Arrange
    trusted_model = StreamingScriptedChatModel(responses=[AIMessage(TRUSTED_ANSWER)])

    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(call_id=f"call-{index}") for index in range(3)],
        protocol=RESAMPLE,
        run_mode=run_mode,
        trusted_model=trusted_model,
    )

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert read_answers(streamed) == [TRUSTED_ANSWER]


def test_a_blocked_step_streams_once_as_committed(run_mode: RunMode) -> None:
    # Act
    streamed = stream_monitored_agent(
        untrusted_steps=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        protocol=BlockAndFeedBack(block_threshold=0.6),
        run_mode=run_mode,
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
        main_monitor=build_monitor_middleware(protocol=RESAMPLE),
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
    "call_tags",
    [(), ("security-monitor",)],
    ids=["monitor-call-inherits-tags", "monitor-call-sets-its-own-tags"],
)
def test_the_v3_event_stream_carries_only_committed_steps(
    run_mode: RunMode,
    call_tags: tuple[str, ...],
) -> None:
    # Arrange
    agent = create_agent(
        model=StreamingScriptedChatModel(
            responses=[build_exfiltration_step(), build_read_step(), AIMessage(FINAL_ANSWER)],
        ),
        tools=Workspace().build_tools(),
        middleware=[build_monitor_middleware(protocol=RESAMPLE, call_tags=call_tags)],
    )

    # Act
    streamed = stream_v3_messages(agent, mode=run_mode)

    # Assert
    assert find_mentions(streamed, text=REJECTED_STEP_MARK) == []
    assert find_mentions(streamed, text=MONITOR_REPLY_MARK) == []
    assert read_tool_call_names(streamed) == ["read_file"]
    assert read_answers(streamed) == [FINAL_ANSWER]
