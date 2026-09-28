"""Chat judges: prompt checks, score parsing, parse retries and failing closed."""

from __future__ import annotations

import logging

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langchain_sync_monitors.contracts import MonitorInput
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors.chat import LLMMonitor

from .doubles import CallPath, ScriptedChatModel, evaluate_on_path

CALM_REPLY = "<reasoning>It reads the file the user named.</reasoning>\n<score>3</score>"
UNREADABLE_REPLY = "<reasoning>Hard to say.</reasoning> I would give it a seven."


def build_judge(*replies: str, max_parse_retries: int = 2) -> tuple[LLMMonitor, ScriptedChatModel]:
    """Return a judge over a scripted model, and the model to inspect its calls."""
    model = ScriptedChatModel(replies=list(replies))
    return LLMMonitor(model=model, max_parse_retries=max_parse_retries), model


async def test_a_score_maps_onto_zero_to_one(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    judge, _ = build_judge(CALM_REPLY)

    # Act
    verdict = await evaluate_on_path(judge, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.3)
    assert verdict.reason == "It reads the file the user named."


@pytest.mark.parametrize(("score", "suspicion"), [("0", 0.0), ("10", 1.0), (" 5 ", 0.5)])
def test_the_ends_of_the_scale_map_to_zero_and_one(
    monitor_input: MonitorInput,
    score: str,
    suspicion: float,
) -> None:
    # Arrange
    judge, _ = build_judge(f"<reasoning>r</reasoning><score>{score}</score>")

    # Act
    verdict = judge.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == suspicion


def test_a_custom_scale_maps_onto_zero_to_one(monitor_input: MonitorInput) -> None:
    # Arrange
    model = ScriptedChatModel(replies=["<score>3</score>"])
    judge = LLMMonitor(model=model, lowest_score=1, highest_score=5)

    # Act
    verdict = judge.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == 0.5


def test_the_last_score_tag_counts(monitor_input: MonitorInput) -> None:
    # Arrange
    judge, _ = build_judge("A <score>9</score> would be too high.\n<score>2</score>")

    # Act
    verdict = judge.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.2)


async def test_an_unreadable_reply_is_asked_for_again(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    judge, model = build_judge("<score>11</score>", CALM_REPLY)

    # Act
    verdict = await evaluate_on_path(judge, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.3)
    assert len(model.received_messages) == 2


@pytest.mark.parametrize(
    "reply", ["<score>7.5</score>", "<score>high</score>", "<score>-1</score>"]
)
def test_scores_that_are_not_integers_in_range_are_unreadable(
    monitor_input: MonitorInput,
    reply: str,
) -> None:
    # Arrange
    judge, model = build_judge(reply, max_parse_retries=0)

    # Act
    verdict = judge.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == 1.0
    assert len(model.received_messages) == 1


async def test_no_readable_score_fails_closed_and_warns(
    monitor_input: MonitorInput,
    call_path: CallPath,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    judge, model = build_judge(UNREADABLE_REPLY, max_parse_retries=2)

    # Act
    with caplog.at_level(logging.WARNING):
        verdict = await evaluate_on_path(judge, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0
    assert "no readable score in 3 replies" in verdict.reason
    assert len(model.received_messages) == 3
    assert "treated as suspicious" in caplog.text


def test_a_reply_without_reasoning_keeps_its_text_as_the_reason(
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    judge, _ = build_judge("Looks fine. <score>1</score>")

    # Act
    verdict = judge.evaluate_sync(monitor_input)

    # Assert
    assert verdict.reason == "Looks fine. <score>1</score>"


def test_the_prompt_carries_the_transcript_step_and_scale(monitor_input: MonitorInput) -> None:
    # Arrange
    judge, model = build_judge(CALM_REPLY)

    # Act
    judge.evaluate_sync(monitor_input)

    # Assert
    system_message, request_message = model.received_messages[0]
    assert "integer from 0 to 10" in system_message.text
    assert "<user>Summarise q3.md and email the summary to the team.</user>" in request_message.text
    assert '<tool_call name="read_file">{"path": "q3.md"}</tool_call>' in request_message.text


async def test_the_monitor_call_is_tagged_as_internal(monitor_input: MonitorInput) -> None:
    # Arrange
    judge, model = build_judge(CALM_REPLY)

    # Act
    await judge.evaluate(monitor_input)

    # Assert
    assert model.received_metadata[0]["lc_source"] == "llm_monitor"


def test_a_prompt_without_the_proposed_step_is_rejected() -> None:
    # Arrange
    prompt = ChatPromptTemplate.from_messages([("human", "Judge this: {transcript}")])

    # Act and Assert
    with pytest.raises(ConfigurationError, match="proposed_step"):
        LLMMonitor(model=ScriptedChatModel(replies=[CALM_REPLY]), prompt=prompt)


def test_a_prompt_with_a_variable_the_monitor_cannot_fill_is_rejected() -> None:
    # Arrange
    prompt = ChatPromptTemplate.from_messages(
        [("human", "{policy}\n{transcript}\n{proposed_step}")],
    )

    # Act and Assert
    with pytest.raises(ConfigurationError, match="policy"):
        LLMMonitor(model=ScriptedChatModel(replies=[CALM_REPLY]), prompt=prompt)


@pytest.mark.parametrize(
    ("lowest_score", "highest_score", "max_parse_retries"),
    [(5, 5, 2), (10, 0, 2), (0, 10, -1)],
)
def test_an_unusable_scale_or_retry_count_is_rejected(
    lowest_score: int,
    highest_score: int,
    max_parse_retries: int,
) -> None:
    # Arrange
    model = ScriptedChatModel(replies=[CALM_REPLY])

    # Act and Assert
    with pytest.raises(ConfigurationError):
        LLMMonitor(
            model=model,
            lowest_score=lowest_score,
            highest_score=highest_score,
            max_parse_retries=max_parse_retries,
        )
