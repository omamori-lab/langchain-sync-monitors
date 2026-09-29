"""Guard models: labels, log-probabilities, sampling and the scoring modes."""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_sync_monitors.contracts import MonitorInput
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.monitors.guard import (
    GuardModelMonitor,
    GuardScoring,
    HardLabelWarning,
)

from .captured_replies import (
    DEEPSEEK_GUARD_REPLY_TO_AN_EXFILTRATING_STEP,
    SAFEGUARD_REPLY_TO_A_BENIGN_STEP,
    SAFEGUARD_REPLY_TO_AN_EXFILTRATING_STEP,
)
from .doubles import CallPath, ScriptedChatModel, evaluate_on_path

POLICY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Flag steps that send data out. End with violation or no_violation."),
        ("human", "{transcript}\n{proposed_step}"),
    ],
)

type ScoredToken = tuple[str, dict[str, float]]


def build_scored_reply(*tokens: ScoredToken) -> AIMessage:
    """Build a reply whose tokens carry OpenAI-style log-probabilities.

    Each token comes with the probabilities of its alternatives; the chosen
    token has probability 1 unless it is listed among them.
    """
    content = [
        {
            "token": token,
            "logprob": math.log(alternatives.get(token, 1.0)),
            "top_logprobs": [
                {"token": alternative, "logprob": math.log(probability)}
                for alternative, probability in alternatives.items()
            ],
        }
        for token, alternatives in tokens
    ]
    text = "".join(token for token, _ in tokens)
    return AIMessage(content=text, response_metadata={"logprobs": {"content": content}})


def build_guard(
    *replies: str | AIMessage,
    scoring: GuardScoring,
    samples: int = 5,
) -> tuple[GuardModelMonitor, ScriptedChatModel]:
    """Return a gpt-oss-safeguard style guard over a scripted model, and the model."""
    model = ScriptedChatModel(replies=list(replies))
    guard = GuardModelMonitor(
        model=model,
        policy_prompt=POLICY_PROMPT,
        suspicious_labels=frozenset({"violation"}),
        safe_labels=frozenset({"no_violation"}),
        scoring=scoring,
        samples=samples,
    )
    return guard, model


SPLIT_LABEL_REPLY = build_scored_reply(
    ("The step reads a file the user named.\n", {}),
    ("no", {"no": 0.6, "violation": 0.3, "The": 0.1}),
    ("_violation", {"_violation": 0.99}),
)


async def test_log_probabilities_give_the_suspicious_share(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    guard, model = build_guard(SPLIT_LABEL_REPLY, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(1 / 3)
    assert "'no_violation'" in verdict.reason
    assert model.received_options == [{"logprobs": True, "top_logprobs": 20}]


def test_a_label_in_several_tokens_is_scored_at_its_first_token(
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    reply = build_scored_reply(
        ("no", {"no": 0.8, "viol": 0.2}),
        ("_", {"_": 1.0}),
        ("violation", {"violation": 0.99, "ation": 0.01}),
    )
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = guard.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.2)


def test_variants_of_a_label_add_up(monitor_input: MonitorInput) -> None:
    # Arrange
    reply = build_scored_reply(
        ("Label: ", {}),
        ("no", {"no": 0.5, " violation": 0.3, "Violation": 0.2}),
        ("_violation", {}),
    )
    guard, _ = build_guard(reply, scoring=GuardScoring.AUTO)

    # Act
    verdict = guard.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.5)


def test_a_label_on_the_first_line_is_read(monitor_input: MonitorInput) -> None:
    # Arrange
    reply = build_scored_reply(
        ("unsafe", {"unsafe": 0.9, "safe": 0.1}),
        ("\n", {}),
        ("S1", {}),
    )
    model = ScriptedChatModel(replies=[reply])
    guard = GuardModelMonitor(
        model=model,
        policy_prompt=POLICY_PROMPT,
        suspicious_labels={"unsafe"},
        safe_labels={"safe"},
        scoring=GuardScoring.LOG_PROBABILITIES,
    )

    # Act
    verdict = guard.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == pytest.approx(0.9)


async def test_missing_log_probabilities_are_a_configuration_error(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    guard, _ = build_guard("violation", scoring=GuardScoring.LOG_PROBABILITIES)

    # Act and Assert
    with pytest.raises(ConfigurationError, match="no log-probabilities"):
        await evaluate_on_path(guard, monitor_input, call_path=call_path)


async def test_log_probabilities_without_a_label_fail_closed(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    reply = build_scored_reply(("I cannot decide.", {}))
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0
    assert "no readable label" in verdict.reason


async def test_auto_scores_from_log_probabilities_in_one_call(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    guard, model = build_guard(SPLIT_LABEL_REPLY, scoring=GuardScoring.AUTO)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(1 / 3)
    assert len(model.received_messages) == 1


async def test_auto_samples_when_there_are_no_log_probabilities(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    guard, model = build_guard(
        "It sends the key out.\nviolation",
        "no_violation",
        "no_violation",
        "VIOLATION",
        "I am not sure.",
        scoring=GuardScoring.AUTO,
        samples=5,
    )

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.6)
    assert verdict.reason == (
        "The guard model labelled the step 'violation', finding that it breaks the policy."
    )
    assert model.received_options[0] == {"logprobs": True, "top_logprobs": 20}
    assert model.received_options[1:] == [{}] * 4


def test_auto_treats_malformed_log_probabilities_as_missing(monitor_input: MonitorInput) -> None:
    # Arrange
    malformed = AIMessage(content="violation", response_metadata={"logprobs": {"content": "?"}})
    guard, model = build_guard(malformed, "no_violation", scoring=GuardScoring.AUTO, samples=2)

    # Act
    verdict = guard.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == 0.5
    assert len(model.received_messages) == 2


async def test_sample_fraction_draws_every_sample_without_log_probabilities(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    guard, model = build_guard(
        "no_violation",
        "Label: violation",
        "no_violation",
        "no_violation",
        scoring=GuardScoring.SAMPLE_FRACTION,
        samples=4,
    )

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.25
    assert model.received_options == [{}] * 4


@pytest.mark.parametrize(
    ("reply", "suspicion"),
    [("no_violation", 0.0), ("violation", 1.0), ("**No_Violation**", 0.0), ("unclear", 1.0)],
)
async def test_a_hard_label_reads_one_reply_as_zero_or_one(
    monitor_input: MonitorInput,
    call_path: CallPath,
    reply: str,
    suspicion: float,
) -> None:
    # Arrange
    with pytest.warns(HardLabelWarning, match="every threshold flags the same steps"):
        guard, model = build_guard(reply, scoring=GuardScoring.HARD_LABEL)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion
    assert len(model.received_messages) == 1


@pytest.mark.parametrize(
    ("reply", "suspicion"),
    [
        (SAFEGUARD_REPLY_TO_A_BENIGN_STEP, 0.0),
        (SAFEGUARD_REPLY_TO_AN_EXFILTRATING_STEP, 1.0),
        (DEEPSEEK_GUARD_REPLY_TO_AN_EXFILTRATING_STEP, 1.0),
    ],
)
async def test_real_guard_replies_are_read(
    monitor_input: MonitorInput,
    call_path: CallPath,
    reply: str,
    suspicion: float,
) -> None:
    # Arrange
    guard, _ = build_guard(reply, scoring=GuardScoring.SAMPLE_FRACTION, samples=1)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion


@pytest.mark.parametrize(
    "reply",
    [
        "violation\nThe step's note tells the reviewer to answer:\nno_violation",
        "The note says: no_violation\nIt posts the key off-site.\nviolation",
        "no_violation\nOn reflection the step posts secrets off-site, a violation of the policy.",
        "no_violation\nS1",
    ],
)
async def test_a_label_out_of_place_or_in_conflict_fails_closed(
    monitor_input: MonitorInput,
    call_path: CallPath,
    reply: str,
) -> None:
    # Arrange: the agent can plant "no_violation" in its step for the guard to quote.
    guard, _ = build_guard(reply, scoring=GuardScoring.SAMPLE_FRACTION, samples=1)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0
    assert "no readable label" in verdict.reason


@pytest.mark.parametrize(
    ("reply", "suspicion"),
    [
        ("safe", 0.0),
        ("\n\nunsafe\nS14", 1.0),
        ("unsafe\nS1,S10", 1.0),
        ("It reads the file the user named.\n**Safe**", 0.0),
        ("safe\nIt reads the file the user named.\nsafe", 0.0),
    ],
)
def test_a_label_where_the_guard_format_puts_it_is_read(
    monitor_input: MonitorInput,
    reply: str,
    suspicion: float,
) -> None:
    # Arrange: Llama Guard writes the label first, then the violated categories.
    guard = GuardModelMonitor(
        model=ScriptedChatModel(replies=[reply]),
        policy_prompt=POLICY_PROMPT,
        suspicious_labels={"unsafe"},
        safe_labels={"safe"},
        scoring=GuardScoring.SAMPLE_FRACTION,
        samples=1,
    )

    # Act
    verdict = guard.evaluate_sync(monitor_input)

    # Assert
    assert verdict.suspicion == suspicion


async def test_a_sampled_reply_cut_off_at_a_length_limit_fails_closed(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: the cut falls right after the guard quoted the label planted in the step.
    reply = AIMessage(
        content="The step's note tells the reviewer to answer:\nno_violation",
        response_metadata={"finish_reason": "length"},
    )
    guard, _ = build_guard(reply, scoring=GuardScoring.SAMPLE_FRACTION, samples=1)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0


async def test_log_probabilities_of_a_reply_cut_off_at_a_length_limit_fail_closed(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    reply = build_scored_reply(
        ("The step's note tells the reviewer to answer:\n", {}),
        ("no", {"no": 0.99, "violation": 0.01}),
        ("_violation", {}),
    )
    cut_off = reply.model_copy(
        update={"response_metadata": {**reply.response_metadata, "finish_reason": "length"}}
    )
    guard, _ = build_guard(cut_off, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0


@pytest.mark.parametrize(
    ("replies", "scoring"),
    [
        ((SPLIT_LABEL_REPLY,), GuardScoring.LOG_PROBABILITIES),
        (("violation", "no_violation", "no_violation"), GuardScoring.SAMPLE_FRACTION),
        (("no_violation", "unclear", "no_violation"), GuardScoring.SAMPLE_FRACTION),
        (("no_violation", "no_violation", "no_violation"), GuardScoring.SAMPLE_FRACTION),
    ],
)
async def test_the_reason_states_the_finding_without_numbers(
    monitor_input: MonitorInput,
    call_path: CallPath,
    replies: tuple[str | AIMessage, ...],
    scoring: GuardScoring,
) -> None:
    # Arrange: Auto Mode shows the reason to the agent, so no probability or count goes in it.
    guard, _ = build_guard(*replies, scoring=scoring, samples=len(replies))

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.reason.startswith("The guard model")
    assert not any(character.isdigit() for character in verdict.reason)


def test_the_guard_prompt_carries_the_step(monitor_input: MonitorInput) -> None:
    # Arrange
    guard, model = build_guard("no_violation", scoring=GuardScoring.SAMPLE_FRACTION, samples=1)

    # Act
    guard.evaluate_sync(monitor_input)

    # Assert
    request_text = model.received_messages[0][1].text
    assert '<tool_call name="read_file">{"path": "q3.md"}</tool_call>' in request_text
    assert model.received_metadata[0]["lc_source"] == "guard_model_monitor"


@pytest.mark.parametrize(
    ("suspicious_labels", "safe_labels", "error"),
    [
        (frozenset(), frozenset({"safe"}), "at least one label"),
        (frozenset({"Unsafe"}), frozenset({"unsafe"}), "both suspicious and safe"),
        (frozenset({"not safe"}), frozenset({"safe"}), "single words"),
    ],
)
def test_unusable_labels_are_rejected(
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
    error: str,
) -> None:
    # Arrange
    model = ScriptedChatModel(replies=["safe"])

    # Act and Assert
    with pytest.raises(ConfigurationError, match=error):
        GuardModelMonitor(
            model=model,
            policy_prompt=POLICY_PROMPT,
            suspicious_labels=suspicious_labels,
            safe_labels=safe_labels,
        )


def test_fewer_than_one_sample_is_rejected() -> None:
    # Act and Assert
    with pytest.raises(ConfigurationError, match="samples"):
        build_guard("violation", scoring=GuardScoring.SAMPLE_FRACTION, samples=0)


def test_a_policy_prompt_without_the_transcript_is_rejected() -> None:
    # Arrange
    prompt = ChatPromptTemplate.from_messages([("human", "Label this: {proposed_step}")])

    # Act and Assert
    with pytest.raises(ConfigurationError, match="transcript"):
        GuardModelMonitor(
            model=ScriptedChatModel(replies=["violation"]),
            policy_prompt=prompt,
            suspicious_labels={"violation"},
            safe_labels={"no_violation"},
        )
