"""OpenRouter's Decisions API as hyoshigi's decision models, over httpx.

Each statement is one request; a transient failure is sent again, and after
the last attempt it is an outcome, as is a reply without a probability.
These tests reach the client through a mock transport, since `DecisionMonitor`
builds it from the environment.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator

import httpx
import hyoshigi
import pytest
import stamina
from hyoshigi import (
    AsyncDecisionModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    Question,
    Refused,
    Rejected,
    Step,
    TextPart,
    Unavailable,
)
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors import ConfigurationError, MonitorTwins
from langchain_sync_monitors.openrouter_decisions import (
    DECISIONS_ENDPOINT,
    DecisionsClient,
    build_decision_models,
    read_openrouter_api_key,
)
from tests.support.log_records import find_frame_leaks, find_leaks, find_logged_leaks
from tests.support.malformed_replies import (
    PLANTED_REPLY_HEADER,
    build_malformed_reply_error,
    fail_with_a_malformed_reply,
    raise_on_send,
)

from .doubles import PLANTED_SECRET, CallPath, rate_on_path

type Responder = Callable[[httpx.Request], httpx.Response]

STATEMENT = "The proposed action leaks a secret."
QUESTION = Question(statement=STATEMENT, state=(TextPart(text="the step"),))
PLANTED_KEY = "sk-planted-key-4d2a"
UNAVAILABLE_REASON = "The monitor's model was unavailable, so the action is treated as suspicious."


class DecisionsServer:
    """A fake Decisions API that plays scripted responses and records each request."""

    def __init__(self, *, responders: list[Responder]) -> None:
        self.responders = responders
        self.requests: list[httpx.Request] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer with the next scripted responder."""
        self.requests.append(request)
        return self.responders[len(self.requests) - 1](request)

    def build_models(
        self, *, api_key: str = "unit-test-key"
    ) -> tuple[DecisionModel, AsyncDecisionModel]:
        """Return the decision models of a client whose requests reach this server."""
        transport = httpx.MockTransport(self.respond)
        client = DecisionsClient(
            model="typesafe/jev-1.14",
            api_key=api_key,
            client=httpx.Client(transport=transport),
            async_client=httpx.AsyncClient(transport=transport),
        )
        return build_decision_models(client)

    def build_monitor(self, *, api_key: str = "unit-test-key") -> MonitorTwins:
        """Return a decision monitor's twins on this server's models, asking the one statement."""
        model, async_model = self.build_models(api_key=api_key)
        return MonitorTwins(
            monitor=hyoshigi.DecisionMonitor(model=model, statements=(STATEMENT,)),
            async_monitor=hyoshigi.AsyncDecisionMonitor(model=async_model, statements=(STATEMENT,)),
        )


def answer_with(probability: float) -> Responder:
    """Build a responder that answers the statement with this probability."""
    body = {
        "model": "typesafe/jev-1.14-20260917",
        "answers": {"statement": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": 380, "output_tokens": 0, "cost": 0.000016},
        "id": "decision-1",
        "provider": "TypeSafe",
    }
    return lambda _request: httpx.Response(200, json=body)


def answer_with_raw(raw_answer: str) -> Responder:
    """Build a responder whose answer to the statement is `raw_answer`, sent as it is written."""
    body = f'{{"answers": {{"statement": {{"type": "noul", "noul": {raw_answer}}}}}}}'.encode()
    headers = {"content-type": "application/json"}
    return lambda _request: httpx.Response(200, content=body, headers=headers)


def fail_with(status_code: int) -> Responder:
    """Build a responder that fails with an HTTP status."""
    return lambda _request: httpx.Response(status_code, json={"error": "unavailable"})


def drop_connection(request: httpx.Request) -> httpx.Response:
    """Fail the way a dropped connection does."""
    message = "connection reset"
    raise httpx.ConnectError(message, request=request)


async def decide_on_path(
    models: tuple[DecisionModel, AsyncDecisionModel],
    *,
    call_path: CallPath,
) -> DecisionOutcome:
    """Ask the statement through the async or the sync model, as the test asks."""
    model, async_model = models
    if call_path == "async":
        return await async_model.decide(QUESTION)
    return model.decide(QUESTION)


@pytest.fixture
def three_attempts() -> Iterator[None]:
    with stamina.set_testing(True, attempts=3):
        yield


async def test_a_statement_goes_in_one_request_as_a_noul_question(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with(0.98)])

    # Act
    outcome = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    (request,) = server.requests
    assert str(request.url) == DECISIONS_ENDPOINT
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert json.loads(request.content) == {
        "model": "typesafe/jev-1.14",
        "state": ["the step"],
        "questions": {"statement": {"type": "noul", "instructions": STATEMENT}},
    }
    assert outcome == Decision(probability=0.98)


def test_both_models_are_named_by_the_provider_and_the_model() -> None:
    # Act
    model, async_model = DecisionsServer(responders=[]).build_models()

    # Assert
    assert model.name == async_model.name == "openrouter:typesafe/jev-1.14"


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize("first_failure", [fail_with(503), fail_with(429), drop_connection])
async def test_a_transient_failure_is_sent_again(
    call_path: CallPath,
    first_failure: Responder,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[first_failure, answer_with(0.1)])

    # Act
    outcome = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert outcome == Decision(probability=0.1)
    assert len(server.requests) == 2


RETRIED_FAILURES = [
    pytest.param(
        fail_with(503),
        "RetriedCallError(error_type='HTTPStatusError', http_status=503)",
        id="server-error",
    ),
    pytest.param(
        fail_with(429),
        "RetriedCallError(error_type='HTTPStatusError', http_status=429)",
        id="rate-limit",
    ),
    pytest.param(
        fail_with_a_malformed_reply,
        "RetriedCallError(error_type='RemoteProtocolError', http_status=None)",
        id="malformed-reply",
    ),
    pytest.param(
        drop_connection,
        "RetriedCallError(error_type='ConnectError', http_status=None)",
        id="dropped-connection",
    ),
]
"""Each failure the client sends again, and the repr stamina's retry log holds for it."""


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize(("first_failure", "logged_error"), RETRIED_FAILURES)
async def test_a_retried_request_hands_a_retry_hook_only_the_error_type_and_status(
    call_path: CallPath,
    step_holding_a_secret: Step,
    retry_details: list[RetryDetails],
    retry_frame_locals: list[dict[str, str]],
    every_log_record: list[logging.LogRecord],
    first_failure: Responder,
    logged_error: str,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[first_failure, answer_with(0.1)])
    monitor = server.build_monitor(api_key=PLANTED_KEY)

    # Act
    score = await rate_on_path(monitor, step_holding_a_secret, call_path=call_path)

    # Assert: the request was sent again, and stamina logged the error's type and status
    assert score.value == 0.1
    assert len(server.requests) == 2
    assert PLANTED_SECRET in server.requests[0].content.decode()
    (details,) = retry_details
    (record,) = [record for record in every_log_record if record.name == "stamina"]
    assert repr(details.caused_by) == vars(record)["stamina.caused_by"] == logged_error
    assert (details.caused_by.__cause__, details.caused_by.__context__) == (None, None)

    # Assert: no record on any logger, and nothing a retry hook is handed, holds the
    # trajectory, the key, any part of the reply or a live httpx object
    secrets = [PLANTED_SECRET, PLANTED_KEY, PLANTED_REPLY_HEADER]
    assert find_logged_leaks(every_log_record, secrets=secrets) == []
    assert find_leaks(details.caused_by, secrets=secrets) == []
    (frame_locals,) = retry_frame_locals
    assert any(name.startswith("run_attempt") for name in frame_locals)
    assert find_frame_leaks(retry_frame_locals, secrets=secrets) == []


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize(
    ("build_failure", "outcome"),
    [
        pytest.param(lambda: fail_with(503), Unavailable(status=503), id="server-error"),
        pytest.param(lambda: fail_with(429), Unavailable(status=429), id="rate-limit"),
        pytest.param(
            lambda: raise_on_send(build_malformed_reply_error()),
            Unavailable(),
            id="malformed-reply",
        ),
        pytest.param(lambda: drop_connection, Unavailable(), id="dropped-connection"),
    ],
)
async def test_a_failure_on_every_attempt_is_unavailable(
    call_path: CallPath,
    retry_details: list[RetryDetails],
    build_failure: Callable[[], Responder],
    outcome: DecisionOutcome,
) -> None:
    # Arrange: an answer waits behind the last attempt, so a further attempt would succeed
    responders = [build_failure(), build_failure(), build_failure(), answer_with(0.1)]
    server = DecisionsServer(responders=responders)

    # Act
    decided = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert decided == outcome
    assert len(server.requests) == 3
    assert len(retry_details) == 2


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize(
    ("status", "outcome"),
    [(403, Refused()), (413, Rejected(status=413)), (408, Unavailable(status=408))],
)
async def test_a_status_the_provider_gives_is_read_as_its_outcome_without_a_retry(
    call_path: CallPath,
    status: int,
    outcome: DecisionOutcome,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[fail_with(status), answer_with(0.1)])

    # Act
    decided = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert decided == outcome
    assert len(server.requests) == 1


@pytest.mark.usefixtures("three_attempts")
async def test_any_other_client_error_raises_naming_its_status_alone(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[fail_with(400), answer_with(0.1)])

    # Act
    with pytest.raises(ConfigurationError) as raised:
        await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert str(raised.value) == "OpenRouter answered HTTP 400"
    assert len(server.requests) == 1


REFUSED_RAW_ANSWERS = [
    pytest.param(answer_with_raw("false"), id="false"),
    pytest.param(answer_with_raw("true"), id="true"),
    pytest.param(answer_with_raw('"0"'), id="the string 0"),
    pytest.param(answer_with_raw('"0.5"'), id="the string 0.5"),
    pytest.param(answer_with_raw("1.5"), id="above one"),
    pytest.param(answer_with_raw("-0.1"), id="below zero"),
    pytest.param(answer_with_raw("null"), id="null"),
    pytest.param(answer_with_raw("NaN"), id="NaN"),
]
"""Replies whose answer is no JSON number from 0 to 1; read leniently, `false` would be a 0."""


@pytest.mark.parametrize(
    "respond",
    [
        *REFUSED_RAW_ANSWERS,
        pytest.param(lambda _request: httpx.Response(200, json={"a": 1}), id="no-answers"),
        pytest.param(
            lambda _request: httpx.Response(
                200,
                json={"answers": {"other": {"type": "noul", "noul": 0.5}}},
            ),
            id="another-question",
        ),
    ],
)
async def test_a_reply_without_a_probability_for_the_statement_is_unavailable(
    call_path: CallPath,
    respond: Responder,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[respond])

    # Act
    decided = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert decided == Unavailable()


@pytest.mark.parametrize(
    ("raw_answer", "probability"),
    [
        pytest.param("0", 0.0, id="the integer 0"),
        pytest.param("1", 1.0, id="the integer 1"),
        pytest.param("0.5", 0.5, id="a half"),
    ],
)
async def test_a_json_number_from_zero_to_one_is_read_as_a_float(
    call_path: CallPath,
    raw_answer: str,
    probability: float,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with_raw(raw_answer)])

    # Act
    decided = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert decided == Decision(probability=probability)
    assert type(decided.probability) is float


@pytest.mark.parametrize(
    "unread_fields",
    [
        {"usage": {"input_tokens": 10, "output_tokens": 0, "cost": {"total": 0.0001}}},
        {"usage": "unmetered", "id": 7, "provider": None},
        {},
    ],
)
async def test_fields_the_client_does_not_read_cannot_discard_an_answer(
    call_path: CallPath,
    unread_fields: dict[str, object],
) -> None:
    # Arrange: only the answers are read, so only they are validated
    body = {"answers": {"statement": {"type": "noul", "noul": 0.95}}, **unread_fields}
    server = DecisionsServer(responders=[lambda _request: httpx.Response(200, json=body)])

    # Act
    decided = await decide_on_path(server.build_models(), call_path=call_path)

    # Assert
    assert decided == Decision(probability=0.95)


async def test_the_monitor_fails_closed_on_a_reply_without_a_probability(
    call_path: CallPath,
    step: Step,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with_raw("false")])

    # Act
    score = await rate_on_path(server.build_monitor(), step, call_path=call_path)

    # Assert
    assert (score.value, score.explanation, score.failed_closed) == (1.0, UNAVAILABLE_REASON, True)


@pytest.mark.parametrize("environment_key", [None, " \n"], ids=["unset", "blank"])
def test_a_missing_key_is_a_configuration_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str | None,
) -> None:
    # Arrange
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if environment_key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", environment_key)

    # Act
    with pytest.raises(ConfigurationError) as raised:
        read_openrouter_api_key()

    # Assert
    assert "OPENROUTER_API_KEY" in str(raised.value)


def test_the_key_in_the_environment_is_read_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", " environment-key\n")

    # Act
    key = read_openrouter_api_key()

    # Assert
    assert key == "environment-key"


@pytest.mark.parametrize("character", ["\n", "\r", "é"], ids=["newline", "return", "e-acute"])
def test_a_key_no_header_may_carry_is_refused_without_showing_the_key(
    monkeypatch: pytest.MonkeyPatch,
    character: str,
) -> None:
    # Arrange: the character sits inside the key, where stripping leaves it
    monkeypatch.setenv("OPENROUTER_API_KEY", f"sk-secret{character}tail")

    # Act
    with pytest.raises(ConfigurationError, match="control or non-ASCII") as raised:
        read_openrouter_api_key()

    # Assert
    message = str(raised.value)
    assert "secret" not in message
    assert "tail" not in message
