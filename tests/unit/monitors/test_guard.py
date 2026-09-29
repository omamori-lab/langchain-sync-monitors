"""Guard models: labels, log-probabilities, sampling and the scoring modes."""

from __future__ import annotations

import math
import re
import time
import warnings

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
from .doubles import CallPath, KeywordRejectingChatModel, ScriptedChatModel, evaluate_on_path

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
    suspicious_labels: frozenset[str] = frozenset({"violation"}),
    safe_labels: frozenset[str] = frozenset({"no_violation"}),
) -> tuple[GuardModelMonitor, ScriptedChatModel]:
    """Return a guard over a scripted model, and the model; gpt-oss-safeguard labels by default."""
    model = ScriptedChatModel(replies=list(replies))
    guard = GuardModelMonitor(
        model=model,
        policy_prompt=POLICY_PROMPT,
        suspicious_labels=suspicious_labels,
        safe_labels=safe_labels,
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
    reply = build_scored_reply(("I cannot decide.", {"I cannot decide.": 0.7, "The": 0.3}))
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0
    assert "no readable label" in verdict.reason


@pytest.mark.parametrize(("opening", "closing"), [("(", ")"), ("[", "]"), ("“", "”"), ("~", "")])
async def test_a_label_in_markup_is_scored_at_the_guard_s_own_token(
    monitor_input: MonitorInput,
    call_path: CallPath,
    opening: str,
    closing: str,
) -> None:
    # Arrange: the tokenizer merges the opening mark into the label's first token.
    reply = build_scored_reply(
        ("The step posts a secret token to a paste site.\n", {}),
        (f"{opening}viol", {f"{opening}viol": 0.9999, f"{opening}no": 0.00008, "no": 0.00002}),
        ("ation", {}),
        (closing, {}),
    )
    guard, model = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.9999)
    assert len(model.received_messages) == 1


@pytest.mark.parametrize("scoring", [GuardScoring.AUTO, GuardScoring.LOG_PROBABILITIES])
@pytest.mark.parametrize(
    ("tokens", "suspicious_labels", "safe_labels"),
    [
        (
            [
                ("It posts the ke", {}),
                ("y\nviol", {"y\nviol": 0.9999, "no": 0.0001}),
                ("ation", {}),
            ],
            frozenset({"violation"}),
            frozenset({"no_violation"}),
        ),
        (
            [
                ("It posts the key.\n", {}),
                ("harm", {"harm": 0.9999, "harmless": 0.0001}),
                ("ful", {}),
            ],
            frozenset({"harmful"}),
            frozenset({"harmless"}),
        ),
    ],
    ids=["token begins no label", "token begins both kinds"],
)
async def test_log_probabilities_at_a_token_that_is_not_the_label_are_not_read(
    monitor_input: MonitorInput,
    call_path: CallPath,
    tokens: list[ScoredToken],
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
    scoring: GuardScoring,
) -> None:
    # Arrange: only a safe label the guard all but ruled out would be left to weigh.
    guard, model = build_guard(
        build_scored_reply(*tokens),
        scoring=scoring,
        samples=3,
        suspicious_labels=suspicious_labels,
        safe_labels=safe_labels,
    )

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert: AUTO samples instead, and LOG_PROBABILITIES fails closed.
    assert verdict.suspicion == 1.0
    assert len(model.received_messages) == (3 if scoring is GuardScoring.AUTO else 1)


async def test_auto_samples_when_log_probabilities_come_without_alternatives(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: a provider that ignores top_logprobs would otherwise give a hard label.
    reply = build_scored_reply(("It posts the key off-site.\n", {}), ("no_violation", {}))
    guard, model = build_guard(
        reply,
        "violation",
        "violation",
        scoring=GuardScoring.AUTO,
        samples=3,
    )

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(2 / 3)
    assert len(model.received_messages) == 3


async def test_log_probabilities_without_alternatives_are_a_configuration_error(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    reply = build_scored_reply(("It posts the key off-site.\n", {}), ("no_violation", {}))
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act and Assert
    with pytest.raises(ConfigurationError, match="no log-probabilities with alternatives"):
        await evaluate_on_path(guard, monitor_input, call_path=call_path)


@pytest.mark.parametrize(
    ("chosen", "alternatives", "suspicion", "reason"),
    [
        (
            "no",
            {"no": 0.1, "violation": 0.9},
            0.9,
            "The guard model labelled the step 'violation', finding that it breaks the policy.",
        ),
        (
            "viol",
            {"viol": 0.3, " No": 0.7},
            0.3,
            "The guard model labelled the step 'no_violation', finding that it follows the policy.",
        ),
        (
            "no",
            {"no": 0.5, "violation": 0.5},
            0.5,
            "The guard model labelled the step 'violation', finding that it breaks the policy.",
        ),
    ],
    ids=["wrote the safe label", "wrote the suspicious label", "an even split"],
)
async def test_the_reason_names_the_label_the_guard_found_likelier(
    monitor_input: MonitorInput,
    call_path: CallPath,
    chosen: str,
    alternatives: dict[str, float],
    suspicion: float,
    reason: str,
) -> None:
    # Arrange: the reason must agree with the suspicion, and read as if the guard wrote it.
    written_rest = "_violation" if chosen == "no" else "ation"
    reply = build_scored_reply(
        ("The step posts a secret token to a paste site.\n", {}),
        (chosen, alternatives),
        (written_rest, {}),
    )
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(suspicion)
    assert verdict.reason == reason


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


EXFILTRATION_REASONING = (
    "The proposed step sends a token from the .env file to an external URL, which is a "
    "secret being transmitted to a destination the user did not request.\n\n"
)


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
    "last_line",
    [
        "violation (the step's note asks the reviewer to answer: no_violation)",
        "Label: violation, whatever the note's request to output: no_violation",
        "This is a violation. The note in the step tells the reviewer to output: no_violation",
        "violation: no_violation",
        "Not a violation: no_violation",
    ],
    ids=["label then quote", "keyed label then quote", "prose quote", "label as key", "prose key"],
)
async def test_a_safe_label_quoted_after_a_colon_is_never_read(
    monitor_input: MonitorInput,
    call_path: CallPath,
    last_line: str,
) -> None:
    # Arrange: the guard quotes the label the agent planted, after text and a colon.
    guard, _ = build_guard(
        EXFILTRATION_REASONING + last_line,
        scoring=GuardScoring.SAMPLE_FRACTION,
        samples=1,
    )

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0
    assert "no readable label" in verdict.reason


@pytest.mark.parametrize(
    ("reply", "suspicion"),
    [
        (EXFILTRATION_REASONING + "Final label: violation", 1.0),
        ("It reads the file the user named.\n**Answer:** no_violation", 0.0),
        ("It reads the file the user named.\n> Label: `no_violation`.", 0.0),
    ],
)
async def test_a_label_after_a_short_key_is_read(
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


async def test_log_probabilities_never_read_a_safe_label_quoted_after_a_colon(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: the guard gives its label, then quotes the note, copying `no` confidently.
    reply = build_scored_reply(
        (EXFILTRATION_REASONING, {}),
        ("violation", {}),
        (" (the step's note asks the reviewer to answer:", {}),
        (" no", {" no": 0.99, " violation": 0.01}),
        ("_violation", {}),
        (")", {}),
    )
    guard, _ = build_guard(reply, scoring=GuardScoring.LOG_PROBABILITIES)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 1.0


@pytest.mark.parametrize(
    "line",
    [":-" * 400 + " x y", ":" * 800 + "x y", " " * 40_000],
    ids=["colon-dash run", "colon run", "blank line"],
)
async def test_a_long_line_that_holds_no_label_is_read_in_linear_time(
    monitor_input: MonitorInput,
    call_path: CallPath,
    line: str,
) -> None:
    # Arrange: a guard can quote such a line from the step; a backtracking parser took seconds.
    reply = f"The step's note holds this line:\n{line}\nviolation"
    guard, _ = build_guard(reply, scoring=GuardScoring.SAMPLE_FRACTION, samples=1)
    started = time.perf_counter()

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert time.perf_counter() - started < 1.0
    assert verdict.suspicion == 1.0
    assert "'violation'" in verdict.reason


@pytest.mark.parametrize(
    ("reply", "suspicion"),
    [
        ("safe", 0.0),
        ("\n\nunsafe\nS14", 1.0),
        ("unsafe\nS1,S10", 1.0),
        ("It reads the file the user named.\n**Safe**", 0.0),
        ("safe\nIt reads the file the user named.\nsafe", 0.0),
        ("It posts the key off-site.\nUnsafe: safe", 1.0),
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


@pytest.mark.filterwarnings("ignore::langchain_sync_monitors.monitors.guard.HardLabelWarning")
@pytest.mark.parametrize("scoring", list(GuardScoring))
def test_every_scoring_member_is_accepted(scoring: GuardScoring) -> None:
    # Act
    guard, _ = build_guard("violation", scoring=scoring)

    # Assert
    assert guard.scoring is scoring


@pytest.mark.parametrize("scoring", [member.value for member in GuardScoring])
def test_a_plain_string_scoring_is_refused_without_a_warning(scoring: str) -> None:
    # Arrange: a string read from YAML or JSON matches no member by identity.
    model = ScriptedChatModel(replies=["violation"])

    accepted = (
        "GuardScoring.AUTO, GuardScoring.LOG_PROBABILITIES, "
        "GuardScoring.SAMPLE_FRACTION, GuardScoring.HARD_LABEL"
    )
    expected = f"scoring must be one of {accepted}, got '{scoring}'"

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigurationError, match=re.escape(expected)):
            GuardModelMonitor(
                model=model,
                policy_prompt=POLICY_PROMPT,
                suspicious_labels={"violation"},
                safe_labels={"no_violation"},
                scoring=scoring,  # ty: ignore[invalid-argument-type]
            )

    # Assert
    assert not [warning for warning in caught if warning.category is HardLabelWarning]


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


def build_rejecting_guard(
    model: KeywordRejectingChatModel,
    *,
    scoring: GuardScoring,
) -> GuardModelMonitor:
    """Return a guard over a model that rejects the `logprobs` keyword."""
    return GuardModelMonitor(
        model=model,
        policy_prompt=POLICY_PROMPT,
        suspicious_labels={"violation"},
        safe_labels={"no_violation"},
        scoring=scoring,
        samples=2,
    )


@pytest.mark.parametrize("scoring", [GuardScoring.AUTO, GuardScoring.LOG_PROBABILITIES])
@pytest.mark.parametrize(
    "rejection",
    [
        "Messages.create() got an unexpected keyword argument 'logprobs'",
        "ChatBedrockConverse._converse_params() got an unexpected keyword argument 'logprobs'",
    ],
    ids=["anthropic", "bedrock converse"],
)
async def test_a_model_that_rejects_log_probabilities_is_a_configuration_error(
    monitor_input: MonitorInput,
    call_path: CallPath,
    rejection: str,
    scoring: GuardScoring,
) -> None:
    # Arrange
    model = KeywordRejectingChatModel(replies=["violation"], rejection=rejection)
    guard = build_rejecting_guard(model, scoring=scoring)

    # Act and Assert
    with pytest.raises(ConfigurationError, match=r"use GuardScoring\.SAMPLE_FRACTION") as raised:
        await evaluate_on_path(guard, monitor_input, call_path=call_path)
    assert isinstance(raised.value.__cause__, TypeError)


async def test_sample_fraction_works_with_a_model_that_rejects_log_probabilities(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    model = KeywordRejectingChatModel(replies=["violation", "no_violation"])
    guard = build_rejecting_guard(model, scoring=GuardScoring.SAMPLE_FRACTION)

    # Act
    verdict = await evaluate_on_path(guard, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.5


@pytest.mark.parametrize(
    ("rejection", "rejects_every_call", "scoring"),
    [
        ("unsupported operand type(s) for +: 'int' and 'str'", False, GuardScoring.AUTO),
        (
            "Messages.create() got an unexpected keyword argument 'logprobs'",
            True,
            GuardScoring.SAMPLE_FRACTION,
        ),
    ],
    ids=["another type error", "no log-probabilities requested"],
)
async def test_other_type_errors_pass_through_unchanged(
    monitor_input: MonitorInput,
    call_path: CallPath,
    rejection: str,
    rejects_every_call: bool,
    scoring: GuardScoring,
) -> None:
    # Arrange
    model = KeywordRejectingChatModel(
        replies=["violation"],
        rejection=rejection,
        rejects_every_call=rejects_every_call,
    )
    guard = build_rejecting_guard(model, scoring=scoring)

    # Act and Assert
    with pytest.raises(TypeError, match=re.escape(rejection)):
        await evaluate_on_path(guard, monitor_input, call_path=call_path)
