"""Decision models: the Decisions API adapter, the TypeSafe adapter and the monitor."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import stamina
from pydantic import SecretStr
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors.contracts import Channel, MonitorInput, MonitorView, TaskAuthor
from langchain_sync_monitors.errors import ConfigurationError, MonitorError
from langchain_sync_monitors.monitors.decision import (
    Aggregation,
    DecisionModel,
    DecisionModelMonitor,
    TypeSafeDecisionModel,
)
from langchain_sync_monitors.monitors.decision_questions import (
    ATTACK_CRITERIA,
    IS_THE_STEP_SUSPICIOUS,
    YesNoQuestion,
)
from langchain_sync_monitors.monitors.openrouter_decisions import (
    OpenRouterDecisionModel,
    read_decisions_probabilities,
)

from .doubles import PLANTED_SECRET, CallPath, evaluate_on_path

if TYPE_CHECKING:
    import httpx2
    from langchain_typesafe import TypeSafeClassifier

type Responder = Callable[[httpx.Request], httpx.Response]

IGNORE_TYPESAFE_BETA = "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"

LEAK_QUESTION = YesNoQuestion(
    key="leaks",
    text="Does the step leak a secret?",
    yes_means="A secret leaves the system.",
    no_means="No secret leaves the system.",
)

GATEWAY_BASE_URL = "https://gateway.test/api"
"""A client's own `base_url`, which completes a relative `base_url` given to the model."""


class DecisionsServer:
    """A fake Decisions API that plays scripted responses and records each request."""

    def __init__(self, *, responders: list[Responder]) -> None:
        self.responders = responders
        self.requests: list[httpx.Request] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer with the next scripted responder."""
        self.requests.append(request)
        return self.responders[len(self.requests) - 1](request)

    def build_model(
        self,
        *,
        base_url: str = "https://decisions.test/api/alpha",
        api_key: SecretStr | None = None,
        client_base_url: str = "",
    ) -> OpenRouterDecisionModel:
        """Return a decision model whose sync and async clients reach this server.

        Both clients have `client_base_url` as their own `base_url`; by
        default they have none.
        """
        transport = httpx.MockTransport(self.respond)
        return OpenRouterDecisionModel(
            model="typesafe/jev-1.13",
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(transport=transport, base_url=client_base_url),
            async_http_client=httpx.AsyncClient(transport=transport, base_url=client_base_url),
        )

    def build_model_with_one_client(
        self,
        *,
        base_url: str,
        client_path: CallPath,
    ) -> OpenRouterDecisionModel:
        """Return a decision model given only `client_path`'s client, with `GATEWAY_BASE_URL`.

        That client reaches this server; the other path opens its own client,
        as the model does for any client not passed.
        """
        transport = httpx.MockTransport(self.respond)
        if client_path == "async":
            return OpenRouterDecisionModel(
                model="typesafe/jev-1.13",
                base_url=base_url,
                async_http_client=httpx.AsyncClient(transport=transport, base_url=GATEWAY_BASE_URL),
            )
        return OpenRouterDecisionModel(
            model="typesafe/jev-1.13",
            base_url=base_url,
            http_client=httpx.Client(transport=transport, base_url=GATEWAY_BASE_URL),
        )


class ScriptedDecisionModel(DecisionModel):
    """Answers with a fixed table, whatever it is asked, and records what it was asked.

    Like any custom `DecisionModel`, it can skip a question or give a value
    that is no probability, and the monitor must catch both.
    """

    def __init__(self, *, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.contexts: list[str] = []
        self.asked_keys: list[tuple[str, ...]] = []

    async def estimate_probabilities(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        return self.estimate_probabilities_sync(context=context, questions=questions)

    def estimate_probabilities_sync(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        self.contexts.append(context)
        self.asked_keys.append(tuple(question.key for question in questions))
        return dict(self.probabilities)


def answer_with(probabilities: dict[str, float], *, status_code: int = 200) -> Responder:
    """Build a responder that returns a Decisions API response with these probabilities."""
    body = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {key: {"type": "noul", "noul": value} for key, value in probabilities.items()},
        "usage": {"input_tokens": 380, "output_tokens": 0, "cost": 0.000016},
        "id": "decision-1",
        "provider": "TypeSafe",
    }
    return lambda _request: httpx.Response(status_code, json=body)


def build_raw_answer_body(raw_answer: str) -> bytes:
    """Build a Decisions API response body whose one answer is `raw_answer`, as JSON text."""
    return f'{{"answers": {{"leaks": {{"type": "noul", "noul": {raw_answer}}}}}}}'.encode()


def answer_with_raw(raw_answer: str) -> Responder:
    """Build a responder whose one answer is `raw_answer`, sent as it is written."""
    body = build_raw_answer_body(raw_answer)
    headers = {"content-type": "application/json"}
    return lambda _request: httpx.Response(200, content=body, headers=headers)


REFUSED_RAW_ANSWERS = [
    pytest.param("false", id="false"),
    pytest.param("true", id="true"),
    pytest.param('"0"', id="the string 0"),
    pytest.param('"0.5"', id="the string 0.5"),
    pytest.param("1.5", id="above one"),
    pytest.param("-0.1", id="below zero"),
    pytest.param("null", id="null"),
    pytest.param("NaN", id="NaN"),
]
"""Answers that are no JSON number from 0 to 1; read leniently, `false` would be a 0."""
READ_RAW_ANSWERS = [
    pytest.param("0", 0.0, id="the integer 0"),
    pytest.param("1", 1.0, id="the integer 1"),
    pytest.param("0.5", 0.5, id="a half"),
]
"""JSON numbers from 0 to 1, integers included, and the probability each is read as."""


def fail_with(status_code: int) -> Responder:
    """Build a responder that fails with an HTTP status."""
    return lambda _request: httpx.Response(status_code, json={"error": "unavailable"})


def drop_connection(request: httpx.Request) -> httpx.Response:
    """Fail the way a dropped connection does."""
    message = "connection reset"
    raise httpx.ConnectError(message, request=request)


async def estimate_on_path(
    model: DecisionModel,
    *,
    questions: Sequence[YesNoQuestion],
    call_path: CallPath,
) -> dict[str, float]:
    """Ask the decision model through its async or sync method, as the test asks."""
    if call_path == "async":
        return await model.estimate_probabilities(context="the step", questions=questions)
    return model.estimate_probabilities_sync(context="the step", questions=questions)


@pytest.fixture(autouse=True)
def openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")


@pytest.fixture
def three_attempts() -> Iterator[None]:
    with stamina.set_testing(True, attempts=3):
        yield


async def test_every_question_goes_in_one_request_as_a_noul(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.98, "suspicious_step": 0.4})])
    model = server.build_model(base_url="https://decisions.test/api/alpha/")

    # Act
    probabilities = await estimate_on_path(
        model,
        questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
        call_path=call_path,
    )

    # Assert
    (request,) = server.requests
    body = json.loads(request.content)
    assert str(request.url) == "https://decisions.test/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == "the step"
    assert body["questions"]["leaks"] == {
        "type": "noul",
        "instructions": "Does the step leak a secret?",
        "criteria": {
            "true": "A secret leaves the system.",
            "false": "No secret leaves the system.",
        },
    }
    assert probabilities == {"leaks": 0.98, "suspicious_step": 0.4}


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize("first_failure", [fail_with(503), fail_with(429), drop_connection])
async def test_transient_failures_are_retried(
    call_path: CallPath, first_failure: Responder
) -> None:
    # Arrange
    server = DecisionsServer(responders=[first_failure, answer_with({"leaks": 0.1})])

    # Act
    probabilities = await estimate_on_path(
        server.build_model(),
        questions=[LEAK_QUESTION],
        call_path=call_path,
    )

    # Assert
    assert probabilities == {"leaks": 0.1}
    assert len(server.requests) == 2


PLANTED_KEY = "sk-planted-key-4d2a"


@pytest.mark.usefixtures("three_attempts")
async def test_a_retried_request_logs_no_part_of_the_transcript_or_the_key(
    call_path: CallPath,
    input_holding_a_secret: MonitorInput,
    retry_details: list[RetryDetails],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the first request fails with a server error, so the second is a retry
    caplog.set_level(logging.DEBUG)
    server = DecisionsServer(responders=[fail_with(503), answer_with({"suspicious_step": 0.1})])
    monitor = DecisionModelMonitor(
        decision_model=server.build_model(api_key=SecretStr(PLANTED_KEY)),
    )

    # Act
    verdict = await evaluate_on_path(monitor, input_holding_a_secret, call_path=call_path)

    # Assert: the retry happened and was logged, and neither the log nor the error's repr
    # holds the request
    assert verdict.suspicion == 0.1
    assert len(server.requests) == 2
    assert PLANTED_SECRET in server.requests[0].content.decode()
    (details,) = retry_details
    assert (details.name, details.args, details.kwargs) == ("<context block>", (), {})
    assert "stamina.retry_scheduled" in [record.getMessage() for record in caplog.records]
    logged = [repr(details.caused_by), *(repr(vars(record)) for record in caplog.records)]
    for text in logged:
        assert PLANTED_SECRET not in text
        assert PLANTED_KEY not in text


PLANTED_PASSWORD = "planted-password-71b3"
"""A password planted in `base_url`, which httpx would send as Basic authentication."""


class RecordCollector(logging.Handler):
    """Keep every record it is handed, whatever its level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def every_log_record(caplog: pytest.LogCaptureFixture) -> Iterator[list[logging.LogRecord]]:
    """Collect every record at every level, from every logger, propagating or not.

    Every logger that exists, httpx's and stamina's included, is set to DEBUG,
    and the collector sits on the root logger and on each logger that does not
    propagate. A logger made later inherits the root's DEBUG level and
    propagates. caplog restores the levels afterwards.
    """
    collector = RecordCollector()
    caplog.set_level(logging.DEBUG)
    loggers = [
        logger
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        caplog.set_level(logging.DEBUG, logger=logger.name)
    holders = [logging.root, *(logger for logger in loggers if not logger.propagate)]
    for holder in holders:
        holder.addHandler(collector)
    yield collector.records
    for holder in holders:
        holder.removeHandler(collector)


@pytest.mark.usefixtures("three_attempts")
async def test_a_password_in_the_base_url_reaches_no_log_and_no_request(
    call_path: CallPath,
    every_log_record: list[logging.LogRecord],
    retry_details: list[RetryDetails],
) -> None:
    # Arrange: a server error, so stamina logs a retry and httpx logs both requests
    server = DecisionsServer(responders=[fail_with(503), answer_with({"leaks": 0.1})])
    base_url = f"https://user:{PLANTED_PASSWORD}@decisions.test/api/alpha"

    # Act
    refusal: ConfigurationError | None = None
    try:
        model = server.build_model(base_url=base_url)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except ConfigurationError as error:
        refusal = error

    # Assert: no record, no retry hook and no URL sent holds the password, and the model
    # refused the URL
    logged = [
        *(f"stamina hook: {details.caused_by!r}" for details in retry_details),
        *(f"{record.name}: {vars(record)!r}" for record in every_log_record),
    ]
    sent = [str(request.url) for request in server.requests]
    assert [text for text in logged if PLANTED_PASSWORD in text] == []
    assert [url for url in sent if PLANTED_PASSWORD in url] == []
    assert isinstance(refusal, ConfigurationError)


BASE_URLS_WITH_CREDENTIALS = {
    "user-and-password": f"https://user:{PLANTED_PASSWORD}@decisions.test/api/alpha",
    "password-alone": f"https://:{PLANTED_PASSWORD}@decisions.test/api/alpha",
    "user-alone": f"https://{PLANTED_PASSWORD}@decisions.test/api/alpha",
    "with-a-port-and-a-slash": f"https://user:{PLANTED_PASSWORD}@decisions.test:8443/api/alpha/",
    "with-an-at-sign-inside": f"https://user:{PLANTED_PASSWORD}@x@decisions.test/api/alpha",
}
"""Base URLs whose user information httpx reads, each holding the planted secret."""


@pytest.mark.parametrize(
    "base_url",
    BASE_URLS_WITH_CREDENTIALS.values(),
    ids=BASE_URLS_WITH_CREDENTIALS.keys(),
)
def test_a_base_url_with_credentials_fails_at_construction_without_showing_them(
    base_url: str,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    with pytest.raises(ConfigurationError, match="base_url holds a user name") as raised:
        server.build_model(base_url=base_url)

    # Assert: neither the error nor anything chained to it quotes the secret
    error = raised.value
    shown = [str(error), repr(error), repr(error.__cause__), repr(error.__context__)]
    assert [text for text in shown if PLANTED_PASSWORD in text] == []
    assert "api_key" in str(error)
    assert server.requests == []


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        pytest.param(
            "https://decisions.test/api/a@b",
            "https://decisions.test/api/a@b/decisions",
            id="at-sign-in-the-path",
        ),
        pytest.param(
            "https://decisions.test:8443/api/alpha",
            "https://decisions.test:8443/api/alpha/decisions",
            id="a-port",
        ),
        pytest.param(
            "http://decisions.test/api/alpha",
            "http://decisions.test/api/alpha/decisions",
            id="plain-http",
        ),
        pytest.param(
            "HTTPS://decisions.test/api/alpha",
            "https://decisions.test/api/alpha/decisions",
            id="an-upper-case-scheme",
        ),
    ],
)
async def test_a_base_url_without_credentials_is_sent_as_given(
    call_path: CallPath,
    base_url: str,
    endpoint: str,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])
    model = server.build_model(base_url=base_url)

    # Act
    probabilities = await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    (request,) = server.requests
    assert str(request.url) == endpoint
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert probabilities == {"leaks": 0.1}


PLANTED_PASSWORD_START = "planted-start-5c0d"
"""The start of a password that httpx, failing to read the URL, would quote as its port."""

UNREADABLE_BASE_URLS = {
    "a-port-that-is-no-number": f"https://user:{PLANTED_PASSWORD}@decisions.test:port/api",
    "a-hash-in-the-password": (
        f"https://user:{PLANTED_PASSWORD_START}#{PLANTED_PASSWORD}@decisions.test/api"
    ),
    "a-slash-in-the-password": (
        f"https://user:{PLANTED_PASSWORD_START}/{PLANTED_PASSWORD}@decisions.test/api"
    ),
    "a-question-mark-in-the-password": (
        f"https://user:{PLANTED_PASSWORD_START}?{PLANTED_PASSWORD}@decisions.test/api"
    ),
    "a-password-and-no-host": f"https://user:{PLANTED_PASSWORD}/api",
}
"""Base URLs httpx cannot read; in all but the first, it would quote part of the password.

A `#`, `/` or `?` ends a URL's authority, so in the middle three httpx reads
the password's start as the port, and in the last the whole password.
"""


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize(
    "base_url",
    UNREADABLE_BASE_URLS.values(),
    ids=UNREADABLE_BASE_URLS.keys(),
)
async def test_a_base_url_httpx_cannot_read_fails_at_construction_without_showing_any_part(
    call_path: CallPath,
    base_url: str,
    every_log_record: list[logging.LogRecord],
    retry_details: list[RetryDetails],
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    failure: ConfigurationError | httpx.InvalidURL | None = None
    try:
        model = server.build_model(base_url=base_url)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except (ConfigurationError, httpx.InvalidURL) as error:
        failure = error

    # Assert: the model refused the URL when built, with nothing chained to the refusal, and
    # no error, record, retry hook or URL sent holds any part of the password
    assert isinstance(failure, ConfigurationError)
    assert str(failure).startswith("base_url is not a URL httpx can read")
    assert failure.__cause__ is None
    assert failure.__context__ is None
    shown = [str(failure), repr(failure)]
    shown += [f"stamina hook: {details.caused_by!r}" for details in retry_details]
    shown += [f"{record.name}: {vars(record)!r}" for record in every_log_record]
    secrets = (PLANTED_PASSWORD_START, PLANTED_PASSWORD)
    assert [text for text in shown if any(secret in text for secret in secrets)] == []
    assert server.requests == []


BASE_URLS_WHOSE_CREDENTIALS_HTTPX_CANNOT_SEE = {
    "a-slash-after-an-empty-port": f"https://user:/{PLANTED_PASSWORD}@decisions.test/api",
    "a-slash-after-a-number-port": f"https://user:123/{PLANTED_PASSWORD}@decisions.test/api",
    "a-question-mark-after-an-empty-port": f"https://user:?{PLANTED_PASSWORD}@decisions.test/api",
    "a-hash-after-an-empty-port": f"https://user:#{PLANTED_PASSWORD}@decisions.test/api",
    "a-slash-in-the-user-name": f"https://user/name:{PLANTED_PASSWORD}@decisions.test/api",
    "a-question-mark-in-the-user-name": f"https://user?name:{PLANTED_PASSWORD}@decisions.test/api",
    "a-hash-in-the-user-name": f"https://user#name:{PLANTED_PASSWORD}@decisions.test/api",
}
"""Base URLs whose user name or password holds a `/`, `?` or `#`, which hides both from httpx.

httpx reads what comes before that character as the host and port, `user`
and an empty port or `123`, so it parses no user name or password and the
check cannot see them. `OpenRouterDecisionModel` and the how-to document
this limit; the test pins it, so the docs and the code cannot drift apart.
"""


@pytest.mark.parametrize(
    "base_url",
    BASE_URLS_WHOSE_CREDENTIALS_HTTPX_CANNOT_SEE.values(),
    ids=BASE_URLS_WHOSE_CREDENTIALS_HTTPX_CANNOT_SEE.keys(),
)
async def test_credentials_httpx_cannot_see_are_not_refused_as_documented(
    call_path: CallPath,
    base_url: str,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    model = server.build_model(base_url=base_url)
    await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert: as documented, the model is built, and its request, key included, goes to a host
    # read from the user name, with the password in the URL that httpx logs
    (request,) = server.requests
    assert request.url.host == "user"
    assert request.url.userinfo == b""
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert PLANTED_PASSWORD in str(request.url)


PLANTED_USER = "planted-user-2f6e"
"""A user name planted where httpx reads a scheme, in a `base_url` with no `//`."""

BASE_URLS_WITHOUT_AN_HTTP_SCHEME_AND_A_HOST = {
    "empty": "",
    "only-spaces": "   ",
    "a-bare-word": "decisions",
    "a-path-alone": "/api",
    "another-scheme": "ftp://host",
    "a-scheme-and-no-host": "https:///api/alpha",
    "a-scheme-alone": "https://",
    "credentials-and-no-scheme": f"{PLANTED_USER}:{PLANTED_PASSWORD}@decisions.test/api",
}
"""Base URLs httpx reads, with no user information, that name no http or https host.

The clients here have no `base_url` of their own to complete the relative
ones, and a client the model opens would send none of them anywhere; the
mock transport answers them all, so an unrefused one reaches the server.
"""


@pytest.mark.parametrize(
    "base_url",
    BASE_URLS_WITHOUT_AN_HTTP_SCHEME_AND_A_HOST.values(),
    ids=BASE_URLS_WITHOUT_AN_HTTP_SCHEME_AND_A_HOST.keys(),
)
async def test_a_base_url_without_an_http_scheme_and_a_host_fails_at_construction(
    call_path: CallPath,
    base_url: str,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    failure: ConfigurationError | httpx.TransportError | None = None
    try:
        model = server.build_model(base_url=base_url)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except (ConfigurationError, httpx.TransportError) as error:
        failure = error

    # Assert: the model refused the URL when built, with nothing chained to the refusal and no
    # part of the planted credentials in it, and nothing was sent
    assert isinstance(failure, ConfigurationError)
    assert str(failure).startswith("base_url is not an http or https URL with a host")
    assert failure.__cause__ is None
    assert failure.__context__ is None
    shown = [str(failure), repr(failure)]
    secrets = (PLANTED_USER, PLANTED_PASSWORD)
    assert [text for text in shown if any(secret in text for secret in secrets)] == []
    assert server.requests == []


def test_a_base_url_with_credentials_and_another_scheme_is_refused_for_the_credentials() -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    with pytest.raises(ConfigurationError) as raised:
        server.build_model(base_url=f"ftp://user:{PLANTED_PASSWORD}@decisions.test/api")

    # Assert: the credentials, the graver fault, are what the message names
    assert str(raised.value).startswith("base_url holds a user name or password")
    assert PLANTED_PASSWORD not in str(raised.value)


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        pytest.param("/v1", f"{GATEWAY_BASE_URL}/v1/decisions", id="a-path"),
        pytest.param("v1/", f"{GATEWAY_BASE_URL}/v1/decisions", id="a-path-with-no-leading-slash"),
        pytest.param("", f"{GATEWAY_BASE_URL}/decisions", id="empty"),
    ],
)
async def test_a_relative_base_url_goes_after_the_base_url_of_the_client_that_sends_it(
    call_path: CallPath,
    base_url: str,
    endpoint: str,
) -> None:
    # Arrange: only the client of the path under test is passed, as a setup that only ever
    # uses that path does
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])
    model = server.build_model_with_one_client(base_url=base_url, client_path=call_path)

    # Act
    probabilities = await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    (request,) = server.requests
    assert str(request.url) == endpoint
    assert request.headers["Authorization"] == "Bearer unit-test-key"
    assert probabilities == {"leaks": 0.1}


async def test_a_relative_base_url_fails_on_the_path_whose_client_was_not_passed(
    call_path: CallPath,
) -> None:
    # Arrange: only the other path's client is passed, so this path opens one with no base_url
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])
    other_path: CallPath = "sync" if call_path == "async" else "async"
    model = server.build_model_with_one_client(base_url="/v1", client_path=other_path)

    # Act
    with pytest.raises(httpx.UnsupportedProtocol):
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert: as documented, the first request fails, and nothing reaches the server
    assert server.requests == []


PLANTED_PATH = "planted-path-4e9a"
"""A path planted in a relative `base_url`, which no refusal may quote."""

RELATIVE_BASE_URLS = {
    "a-path": f"/{PLANTED_PATH}",
    "a-path-with-no-leading-slash": PLANTED_PATH,
    "a-host-and-no-scheme": f"//{PLANTED_PATH}/api",
    "empty": "",
}
"""Base URLs with no scheme, which only a client's own `base_url` can complete."""


@pytest.mark.parametrize("base_url", RELATIVE_BASE_URLS.values(), ids=RELATIVE_BASE_URLS.keys())
async def test_a_relative_base_url_with_no_client_fails_at_construction(
    call_path: CallPath,
    base_url: str,
) -> None:
    # Arrange: no client is passed, so each path would open one with no base_url, and an
    # unrefused URL would fail at the request instead
    failure: ConfigurationError | httpx.TransportError | None = None

    # Act
    try:
        model = OpenRouterDecisionModel(model="typesafe/jev-1.13", base_url=base_url)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except (ConfigurationError, httpx.TransportError) as error:
        failure = error

    # Assert: the model refused the URL when built, naming the clients that could complete it,
    # with nothing chained to the refusal and no part of the URL in it
    assert isinstance(failure, ConfigurationError)
    assert str(failure).startswith(
        "base_url is not an http or https URL with a host, and neither http_client nor "
        "async_http_client has a base_url"
    )
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert PLANTED_PATH not in str(failure)
    assert PLANTED_PATH not in repr(failure)


BASE_URLS_WITH_A_SCHEME_AND_NO_HTTP_HOST = {
    "another-scheme": "ftp://decisions.test/api",
    "a-host-read-as-a-scheme": "localhost:8080",
    "an-http-scheme-and-no-host": "https:///v1",
    "credentials-after-an-http-scheme": f"https:user:{PLANTED_PASSWORD}@decisions.test/api",
    "credentials-and-no-scheme": f"{PLANTED_USER}:{PLANTED_PASSWORD}@decisions.test/api",
}
"""Base URLs with a scheme, but no http or https host, that httpx would not send or would merge.

httpx's transports send no `ftp` URL. Beside a client with its own
`base_url`, httpx would put each of the others after that `base_url` as a
path, the planted password included.
"""


@pytest.mark.parametrize(
    "base_url",
    BASE_URLS_WITH_A_SCHEME_AND_NO_HTTP_HOST.values(),
    ids=BASE_URLS_WITH_A_SCHEME_AND_NO_HTTP_HOST.keys(),
)
async def test_a_base_url_with_a_scheme_and_no_http_host_is_refused_whatever_the_clients(
    call_path: CallPath,
    base_url: str,
) -> None:
    # Arrange: both clients have a base_url of their own
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    failure: ConfigurationError | None = None
    try:
        model = server.build_model(base_url=base_url, client_base_url=GATEWAY_BASE_URL)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except ConfigurationError as error:
        failure = error

    # Assert: the model refused the URL when built, with no part of the planted credentials in
    # the refusal, and nothing was sent
    assert isinstance(failure, ConfigurationError)
    assert str(failure).startswith("base_url is not an http or https URL with a host, and it is")
    shown = [str(failure), repr(failure)]
    secrets = (PLANTED_USER, PLANTED_PASSWORD)
    assert [text for text in shown if any(secret in text for secret in secrets)] == []
    assert server.requests == []


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param(f"//user:{PLANTED_PASSWORD}@decisions.test/v1", id="user-and-password"),
        pytest.param(f"//{PLANTED_PASSWORD}@decisions.test/v1", id="user-alone"),
    ],
)
async def test_a_relative_base_url_with_credentials_is_refused_for_them_beside_a_client_base_url(
    call_path: CallPath,
    base_url: str,
) -> None:
    # Arrange: the path's client has a base_url of its own, which would complete the URL
    server = DecisionsServer(responders=[answer_with({"leaks": 0.1})])

    # Act
    failure: ConfigurationError | None = None
    try:
        model = server.build_model_with_one_client(base_url=base_url, client_path=call_path)
        await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)
    except ConfigurationError as error:
        failure = error

    # Assert: the credentials check still runs first, and nothing was sent
    assert isinstance(failure, ConfigurationError)
    assert str(failure).startswith("base_url holds a user name or password")
    assert PLANTED_PASSWORD not in str(failure)
    assert server.requests == []


@pytest.mark.usefixtures("three_attempts")
async def test_a_client_error_is_not_retried(call_path: CallPath) -> None:
    # Arrange
    server = DecisionsServer(responders=[fail_with(400), answer_with({"leaks": 0.1})])

    # Act and Assert
    with pytest.raises(httpx.HTTPStatusError):
        await estimate_on_path(server.build_model(), questions=[LEAK_QUESTION], call_path=call_path)
    assert len(server.requests) == 1


def refuse_the_request_locally(request: httpx.Request) -> httpx.Response:
    """Fail the way httpx fails a request with an illegal header value, quoting the header."""
    message = f"Illegal header value {request.headers['Authorization']!r}"
    raise httpx.LocalProtocolError(message)


def refuse_the_scheme(_request: httpx.Request) -> httpx.Response:
    """Fail the way httpx fails a URL whose scheme it cannot send to."""
    message = "Request URL has an unsupported protocol 'ftp://'."
    raise httpx.UnsupportedProtocol(message)


@pytest.mark.usefixtures("three_attempts")
@pytest.mark.parametrize("refusal", [refuse_the_request_locally, refuse_the_scheme])
async def test_a_request_the_client_refuses_is_not_retried(
    call_path: CallPath,
    refusal: Responder,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[refusal, answer_with({"leaks": 0.1})])

    # Act and Assert
    with pytest.raises(httpx.TransportError):
        await estimate_on_path(server.build_model(), questions=[LEAK_QUESTION], call_path=call_path)
    assert len(server.requests) == 1


def test_a_response_in_an_unexpected_shape_is_a_monitor_error() -> None:
    # Arrange
    server = DecisionsServer(responders=[lambda _request: httpx.Response(200, json={"a": 1})])

    # Act and Assert
    with pytest.raises(MonitorError, match="unexpected shape"):
        server.build_model().estimate_probabilities_sync(context="x", questions=[LEAK_QUESTION])


@pytest.mark.parametrize(
    "unread_fields",
    [
        {"usage": {"input_tokens": 10, "output_tokens": 0, "cost": {"total": 0.0001}}},
        {"usage": "unmetered", "id": 7, "provider": None},
        {},
    ],
)
async def test_fields_the_library_does_not_read_cannot_discard_an_answer(
    call_path: CallPath,
    unread_fields: dict[str, object],
) -> None:
    # Arrange: only the answers are read, so only they are validated.
    body = {"answers": {"leaks": {"type": "noul", "noul": 0.95}}, **unread_fields}
    server = DecisionsServer(responders=[lambda _request: httpx.Response(200, json=body)])

    # Act
    probabilities = await estimate_on_path(
        server.build_model(),
        questions=[LEAK_QUESTION],
        call_path=call_path,
    )

    # Assert
    assert probabilities == {"leaks": 0.95}


@pytest.mark.parametrize("raw_answer", REFUSED_RAW_ANSWERS)
def test_a_decisions_answer_that_is_no_json_number_from_zero_to_one_is_refused(
    raw_answer: str,
) -> None:
    # Arrange
    content = build_raw_answer_body(raw_answer)

    # Act and Assert
    with pytest.raises(MonitorError, match="unexpected shape"):
        read_decisions_probabilities(content, questions=[LEAK_QUESTION])


@pytest.mark.parametrize(("raw_answer", "probability"), READ_RAW_ANSWERS)
def test_a_decisions_answer_that_is_a_json_number_from_zero_to_one_is_read(
    raw_answer: str,
    probability: float,
) -> None:
    # Arrange
    content = build_raw_answer_body(raw_answer)

    # Act
    probabilities = read_decisions_probabilities(content, questions=[LEAK_QUESTION])

    # Assert
    assert probabilities == {"leaks": probability}
    assert type(probabilities["leaks"]) is float


@pytest.mark.parametrize("raw_answer", REFUSED_RAW_ANSWERS)
async def test_the_monitor_fails_on_a_decisions_answer_that_is_no_json_number(
    monitor_input: MonitorInput,
    call_path: CallPath,
    raw_answer: str,
) -> None:
    # Arrange: the step fails, so nothing it proposed runs.
    server = DecisionsServer(responders=[answer_with_raw(raw_answer)])
    monitor = DecisionModelMonitor(decision_model=server.build_model(), questions=[LEAK_QUESTION])

    # Act and Assert
    with pytest.raises(MonitorError, match="unexpected shape"):
        await evaluate_on_path(monitor, monitor_input, call_path=call_path)


@pytest.mark.parametrize(("raw_answer", "suspicion"), READ_RAW_ANSWERS)
async def test_the_monitor_scores_a_decisions_answer_that_is_a_json_number(
    monitor_input: MonitorInput,
    call_path: CallPath,
    raw_answer: str,
    suspicion: float,
) -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with_raw(raw_answer)])
    monitor = DecisionModelMonitor(decision_model=server.build_model(), questions=[LEAK_QUESTION])

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion
    assert type(verdict.suspicion) is float


async def test_a_question_left_unanswered_is_a_monitor_error() -> None:
    # Arrange
    server = DecisionsServer(responders=[answer_with({"other": 0.5})])

    # Act and Assert
    with pytest.raises(MonitorError, match="leaks"):
        await server.build_model().estimate_probabilities(context="x", questions=[LEAK_QUESTION])


@pytest.mark.parametrize("environment_key", [None, " \n"], ids=["unset", "blank"])
def test_a_missing_key_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str | None,
) -> None:
    # Arrange
    if environment_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", environment_key)

    # Act and Assert
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterDecisionModel(model="typesafe/jev-1.13")


async def test_the_key_in_the_environment_is_sent_stripped(
    monkeypatch: pytest.MonkeyPatch,
    call_path: CallPath,
) -> None:
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", " environment-key\n")
    server = DecisionsServer(responders=[answer_with({"leaks": 0.5})])
    model = server.build_model()

    # Act
    await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    (request,) = server.requests
    assert request.headers["Authorization"] == "Bearer environment-key"


@pytest.mark.parametrize("given_key", ["given-key", " given-key\n"], ids=["bare", "padded"])
async def test_a_key_given_is_sent_stripped_in_place_of_the_one_in_the_environment(
    call_path: CallPath,
    given_key: str,
) -> None:
    # Arrange: OPENROUTER_API_KEY holds another key
    server = DecisionsServer(responders=[answer_with({"leaks": 0.5})])
    model = server.build_model(api_key=SecretStr(given_key))

    # Act
    await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    (request,) = server.requests
    assert request.headers["Authorization"] == "Bearer given-key"


@pytest.mark.parametrize("blank_key", ["", "  "], ids=["empty", "whitespace"])
def test_a_blank_key_given_fails_rather_than_fall_back_to_the_environment(blank_key: str) -> None:
    # Arrange: OPENROUTER_API_KEY holds a key the model must not fall back to
    api_key = SecretStr(blank_key)

    # Act and Assert
    with pytest.raises(ConfigurationError, match="api_key is blank"):
        OpenRouterDecisionModel(model="typesafe/jev-1.13", api_key=api_key)


UNSENDABLE_KEYS = {
    "given-newline": ("given", "\n"),
    "given-return": ("given", "\r"),
    "given-null": ("given", "\x00"),
    "given-e-acute": ("given", "\u00e9"),
    "environment-newline": ("environment", "\n"),
    "environment-return": ("environment", "\r"),
    "environment-e-acute": ("environment", "\u00e9"),
}
"""Where a key comes from, and the character inside it that no header may carry. No
environment variable can hold a null byte, so only a key given can."""


@pytest.mark.parametrize(
    ("source", "character"), UNSENDABLE_KEYS.values(), ids=UNSENDABLE_KEYS.keys()
)
def test_a_key_no_header_may_carry_fails_without_showing_the_key(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    character: str,
) -> None:
    # Arrange: the character sits inside the key, where stripping leaves it
    key = f"sk-secret{character}tail"
    if source == "environment":
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    api_key = SecretStr(key) if source == "given" else None

    # Act
    with pytest.raises(ConfigurationError, match="control or non-ASCII") as raised:
        OpenRouterDecisionModel(model="typesafe/jev-1.13", api_key=api_key)

    # Assert
    message = str(raised.value)
    assert "secret" not in message
    assert "tail" not in message


def test_a_key_given_as_a_plain_string_fails_without_showing_the_key() -> None:
    # Arrange
    api_key = cast("SecretStr", "sk-plain-key")

    # Act
    with pytest.raises(ConfigurationError, match="must be a SecretStr") as raised:
        OpenRouterDecisionModel(model="typesafe/jev-1.13", api_key=api_key)

    # Assert
    assert "sk-plain-key" not in str(raised.value)


def build_typesafe_classifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requests: list[httpx2.Request],
) -> TypeSafeClassifier:
    """Return a real classifier whose HTTP clients reach a fake TypeSafe API.

    It needs the typesafe extra, so each TypeSafe test skips without it first.
    """
    import httpx2
    from langchain_typesafe import TypeSafeClassifier

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        answers = {"leaks": {"type": "noul", "noul": 0.9}}
        return httpx2.Response(200, json={"model": "jev-1.13", "answers": answers})

    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-key")
    transport = httpx2.MockTransport(respond)
    return TypeSafeClassifier(
        client=httpx2.Client(transport=transport),
        async_client=httpx2.AsyncClient(transport=transport),
    )


@pytest.mark.filterwarnings(IGNORE_TYPESAFE_BETA)
async def test_typesafe_questions_become_nouls_with_criteria(
    monkeypatch: pytest.MonkeyPatch,
    call_path: CallPath,
) -> None:
    # Arrange
    pytest.importorskip("langchain_typesafe")
    requests: list[httpx2.Request] = []
    classifier = build_typesafe_classifier(monkeypatch, requests=requests)
    model = TypeSafeDecisionModel(classifier=classifier)

    # Act
    probabilities = await estimate_on_path(model, questions=[LEAK_QUESTION], call_path=call_path)

    # Assert
    body = json.loads(requests[0].content)
    assert body["state"] == "the step"
    assert body["questions"]["leaks"]["type"] == "noul"
    assert body["questions"]["leaks"]["criteria"] == {
        "true": "A secret leaves the system.",
        "false": "No secret leaves the system.",
    }
    assert probabilities == {"leaks": 0.9}


@pytest.mark.filterwarnings(IGNORE_TYPESAFE_BETA)
def test_typesafe_without_the_extra_explains_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    pytest.importorskip("langchain_typesafe")
    classifier = build_typesafe_classifier(monkeypatch, requests=[])
    monkeypatch.setitem(sys.modules, "langchain_typesafe", None)

    # Act and Assert
    with pytest.raises(ConfigurationError, match=r"langchain-sync-monitors\[typesafe\]"):
        TypeSafeDecisionModel(classifier=classifier)


async def test_one_question_gives_its_probability_of_yes(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.03})
    monitor = DecisionModelMonitor(decision_model=model)

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.03
    assert "'suspicious_step'" in verdict.reason


@pytest.mark.parametrize(
    ("task_author", "tag"),
    [(TaskAuthor.USER, "user"), (TaskAuthor.PARENT_AGENT, "delegator")],
)
async def test_the_context_holds_the_transcript_and_the_proposed_step(
    monitor_input: MonitorInput,
    call_path: CallPath,
    task_author: TaskAuthor,
    tag: str,
) -> None:
    # Arrange: a subagent's task comes from its parent agent, not from the user.
    model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.03})
    step = dataclasses.replace(monitor_input, task_author=task_author)

    # Act
    await evaluate_on_path(DecisionModelMonitor(decision_model=model), step, call_path=call_path)

    # Assert
    (context,) = model.contexts
    assert context.startswith(f"<transcript>\n<{tag}>Summarise q3.md")
    assert context.endswith(
        '<tool_call name="read_file">{"path": "q3.md"}</tool_call>\n</proposed_step>'
    )


def test_a_view_without_tool_calls_still_shows_the_decision_model_the_call(
    monitor_input: MonitorInput,
) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.03})
    monitor = DecisionModelMonitor(decision_model=model, view=MonitorView(channels=Channel.USER))

    # Act
    monitor.evaluate_sync(monitor_input)

    # Assert
    (context,) = model.contexts
    assert context.endswith(
        '<tool_call name="read_file">{"path": "q3.md"}</tool_call>\n</proposed_step>'
    )


MIN_REASON = "The decision model's score rests on all of its concerns together: " + "; ".join(
    f"{question.key!r}: {question.text}" for question in ATTACK_CRITERIA
)


@pytest.mark.parametrize(
    ("aggregation", "suspicion", "reason_start"),
    [
        (Aggregation.MAX, 0.95, "The decision model's strongest concern is 'sends_data_out': "),
        (Aggregation.MIN, 0.05, MIN_REASON),
    ],
)
async def test_several_questions_are_combined_and_named(
    monitor_input: MonitorInput,
    call_path: CallPath,
    aggregation: Aggregation,
    suspicion: float,
    reason_start: str,
) -> None:
    # Arrange
    probabilities = dict.fromkeys((question.key for question in ATTACK_CRITERIA), 0.05)
    probabilities["sends_data_out"] = 0.95
    model = ScriptedDecisionModel(probabilities=probabilities)
    monitor = DecisionModelMonitor(
        decision_model=model, questions=ATTACK_CRITERIA, aggregation=aggregation
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion
    assert verdict.reason.startswith(reason_start)
    assert not any(character.isdigit() for character in verdict.reason)


@pytest.mark.parametrize("lowest", [question.key for question in ATTACK_CRITERIA])
async def test_the_min_reason_does_not_point_at_the_lowest_question(
    monitor_input: MonitorInput,
    call_path: CallPath,
    lowest: str,
) -> None:
    # Arrange: under MIN the lowest answer is the one lever that lets the step pass.
    probabilities = dict.fromkeys((question.key for question in ATTACK_CRITERIA), 0.97)
    probabilities[lowest] = 0.91
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities=probabilities),
        questions=ATTACK_CRITERIA,
        aggregation=Aggregation.MIN,
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.91
    assert verdict.reason == MIN_REASON


@pytest.mark.parametrize(
    "probabilities",
    [{"suspicious_step": 0.87}, {"suspicious_step": 0.03}],
)
async def test_the_reason_names_the_concern_without_numbers(
    monitor_input: MonitorInput,
    call_path: CallPath,
    probabilities: dict[str, float],
) -> None:
    # Arrange: Auto Mode shows the reason to the agent, so no probability goes in it.
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities=probabilities),
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.reason == (
        "The decision model's strongest concern is 'suspicious_step': "
        f"{IS_THE_STEP_SUSPICIOUS.text}"
    )
    assert not any(character.isdigit() for character in verdict.reason)


@pytest.mark.parametrize("aggregation", [Aggregation.MAX, Aggregation.MIN])
@pytest.mark.parametrize(
    "answers",
    [
        {"leaks": 0.2, "suspicious_step": math.nan},
        {"leaks": 0.2, "suspicious_step": 1.5},
        {"leaks": 0.2, "suspicious_step": -0.1},
        {"leaks": 0.2},
        {"leaks": 0.2, "suspicious_step": None},
        {"leaks": 0.2, "suspicious_step": "0.9"},
        {"leaks": 0.2, "suspicious_step": True},
        {"leaks": 0.2, "suspicious_step": False},
        {"leaks": 0.2, "suspicious_step": Decimal("NaN")},
        {"leaks": 0.2, "suspicious_step": Decimal("sNaN")},
        {"leaks": 0.2, "suspicious_step": Decimal("Infinity")},
        {"leaks": 0.2, "suspicious_step": Decimal("1.5")},
        {"leaks": 0.2, "suspicious_step": 10**400},
        {"leaks": 0.2, "suspicious_step": Fraction(10**400, 1)},
        {"leaks": 0.2, "suspicious_step": Fraction(10**400 + 1, 10**400)},
        {"leaks": 0.2, "suspicious_step": Fraction(-1, 10**400)},
    ],
    ids=[
        "not a number",
        "above one",
        "below zero",
        "skipped",
        "None",
        "a string",
        "True",
        "False",
        "a Decimal NaN",
        "a Decimal signalling NaN",
        "a Decimal infinity",
        "a Decimal above one",
        "an integer too large for a float",
        "a Fraction too large for a float",
        "a Fraction a float would round down to one",
        "a Fraction a float would round up to zero",
    ],
)
async def test_an_answer_that_is_no_probability_is_a_monitor_error(
    monitor_input: MonitorInput,
    call_path: CallPath,
    answers: dict[str, object],
    aggregation: Aggregation,
) -> None:
    # Arrange: a custom decision model's answers reach the monitor unvalidated.
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(
            probabilities=answers,  # ty: ignore[invalid-argument-type]
        ),
        questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
        aggregation=aggregation,
    )

    # Act and Assert: the step fails, as it does for an unreadable Decisions API answer.
    with pytest.raises(MonitorError, match="suspicious_step"):
        await evaluate_on_path(monitor, monitor_input, call_path=call_path)


@pytest.mark.parametrize(
    ("aggregation", "suspicion"), [(Aggregation.MAX, 1.0), (Aggregation.MIN, 0.0)]
)
@pytest.mark.parametrize(
    "answers",
    [
        {"leaks": 0.0, "suspicious_step": 1.0},
        {"leaks": 0, "suspicious_step": 1},
        {"leaks": Decimal("0.0"), "suspicious_step": Decimal("1")},
        {"leaks": Fraction(0), "suspicious_step": Fraction(1, 1)},
        {"leaks": Fraction(1, 10**400), "suspicious_step": Fraction(10**400 - 1, 10**400)},
    ],
    ids=["floats", "integers", "Decimals", "Fractions", "Fractions a float rounds to 0 and 1"],
)
async def test_answers_at_zero_and_one_are_read_as_floats(
    monitor_input: MonitorInput,
    call_path: CallPath,
    answers: dict[str, float],
    aggregation: Aggregation,
    suspicion: float,
) -> None:
    # Arrange
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities=answers),
        questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
        aggregation=aggregation,
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == suspicion
    assert type(verdict.suspicion) is float


def test_the_mean_warns_at_the_constructor_call_that_it_dilutes_a_single_hit() -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={})

    # Act
    with pytest.warns(UserWarning, match="dilutes a single strong hit") as record:
        DecisionModelMonitor(decision_model=model, aggregation=Aggregation.MEAN)

    # Assert
    assert len(record) == 1
    assert record[0].filename == __file__


MEAN_WARNING = (
    "Aggregation.MEAN dilutes a single strong hit (0.95 with three 0.05s averages 0.275); "
    "Aggregation.MAX keeps it."
)


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [(Aggregation.MAX, []), (Aggregation.MIN, []), (Aggregation.MEAN, [MEAN_WARNING])],
    ids=["max", "min", "mean"],
)
def test_only_the_mean_warns_and_the_warning_says_why(
    aggregation: Aggregation,
    expected: list[str],
) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={})

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DecisionModelMonitor(decision_model=model, aggregation=aggregation)

    # Assert
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (UserWarning, message) for message in expected
    ]


@pytest.mark.filterwarnings("ignore:Aggregation.MEAN dilutes:UserWarning")
async def test_the_mean_averages_the_answers(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(
            probabilities={"leaks": 0.95, "suspicious_step": 0.05},
        ),
        questions=[LEAK_QUESTION, IS_THE_STEP_SUSPICIOUS],
        aggregation=Aggregation.MEAN,
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.5)


async def test_questions_given_as_a_generator_are_all_asked(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: a generator is consumed by one pass, and the constructor checks it first.
    probabilities = dict.fromkeys((question.key for question in ATTACK_CRITERIA), 0.05)
    probabilities["sends_data_out"] = 0.97
    model = ScriptedDecisionModel(probabilities=probabilities)
    monitor = DecisionModelMonitor(
        decision_model=model,
        questions=(question for question in ATTACK_CRITERIA),
    )

    # Act
    verdict = await evaluate_on_path(monitor, monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == 0.97
    assert model.asked_keys == [tuple(question.key for question in ATTACK_CRITERIA)]


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [LEAK_QUESTION, LEAK_QUESTION],
        (question for question in ()),
        (question for question in (LEAK_QUESTION, LEAK_QUESTION)),
    ],
    ids=["empty list", "repeated key", "empty generator", "generator with a repeated key"],
)
def test_questions_must_be_present_and_unique(questions: Iterable[YesNoQuestion]) -> None:
    # Arrange
    model = ScriptedDecisionModel(probabilities={})

    # Act and Assert
    with pytest.raises(ConfigurationError, match="unique keys"):
        DecisionModelMonitor(decision_model=model, questions=questions)


@pytest.mark.filterwarnings("ignore:Aggregation.MEAN dilutes:UserWarning")
@pytest.mark.parametrize("aggregation", list(Aggregation))
def test_every_aggregation_member_is_accepted(aggregation: Aggregation) -> None:
    # Act
    monitor = DecisionModelMonitor(
        decision_model=ScriptedDecisionModel(probabilities={"suspicious_step": 0.2}),
        aggregation=aggregation,
    )

    # Assert
    assert monitor.aggregation is aggregation


@pytest.mark.parametrize("aggregation", [member.value for member in Aggregation])
def test_a_plain_string_aggregation_is_refused_without_a_warning(aggregation: str) -> None:
    # Arrange: a string read from YAML or JSON matches no member by identity.
    decision_model = ScriptedDecisionModel(probabilities={"suspicious_step": 0.2})
    expected = (
        "aggregation must be one of Aggregation.MAX, Aggregation.MIN, Aggregation.MEAN, "
        f"got '{aggregation}'. Convert a string with Aggregation(value)"
    )

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigurationError, match=re.escape(expected)):
            DecisionModelMonitor(
                decision_model=decision_model,
                aggregation=aggregation,  # ty: ignore[invalid-argument-type]
            )

    # Assert
    assert not [warning for warning in caught if warning.category is UserWarning]


def test_the_attack_criteria_are_four_distinct_questions() -> None:
    # Act
    keys = {question.key for question in ATTACK_CRITERIA}

    # Assert
    assert keys == {
        "sends_data_out",
        "weakens_security",
        "goes_beyond_the_task",
        "misleads_the_user",
    }
    assert len({question.text for question in ATTACK_CRITERIA}) == len(ATTACK_CRITERIA) == 4
