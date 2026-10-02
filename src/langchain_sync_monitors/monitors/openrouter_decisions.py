"""OpenRouter's Decisions API as a decision model, and the payloads it exchanges.

`OpenRouterDecisionModel` sends every yes/no question about a step in one
request to the Decisions API (alpha) and reads the probability of yes for
each; `decision` holds the monitor that asks them.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Literal, TypedDict

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from langchain_sync_monitors._langchain import TraceSpan, open_traced_run, open_traced_run_sync
from langchain_sync_monitors.errors import ConfigurationError, MonitorError
from langchain_sync_monitors.monitors.decision import DecisionModel, select_question_probabilities
from langchain_sync_monitors.monitors.decision_questions import YesNoQuestion
from langchain_sync_monitors.options import (
    check_instance_option,
    check_optional_instance_option,
    read_positive_number_option,
)
from langchain_sync_monitors.retries import call_with_retries, call_with_retries_sync
from langchain_sync_monitors.spans import CLASSIFIER_SPAN_NAME, MONITOR_TAG, MONITOR_WORK_METADATA

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha"
RETRY_ATTEMPTS = 5


class NoulCriteriaBody(TypedDict):
    """What yes (`true`) and no (`false`) mean, as the Decisions API expects them."""

    true: str
    false: str


class NoulQuestionBody(TypedDict):
    """One yes/no question ("noul") in a Decisions API request."""

    type: Literal["noul"]
    instructions: str
    criteria: NoulCriteriaBody


class DecisionsRequestBody(TypedDict):
    """The body of a Decisions API request."""

    model: str
    state: str
    questions: dict[str, NoulQuestionBody]


class DecisionAnswer(BaseModel):
    """One answer from the Decisions API: the probability of yes, a JSON number from 0 to 1."""

    type: Literal["noul"]
    # Strict, since pydantic's lax mode reads `true`, `false` and strings such as "0.5" as numbers
    # [@pydantic2026], and `false` would pass as 0. A JSON integer such as 0 or 1 is still read.
    noul: float = Field(strict=True, ge=0.0, le=1.0)


class DecisionsResponse(BaseModel):
    """The part of a Decisions API response the library reads: the answers.

    pydantic validates this external payload [@pydantic2026]. The response
    also carries `model`, `usage`, `id` and `provider`, as confirmed live in
    September 2026. They are ignored, not validated, so a change in their
    shape cannot discard a valid answer.
    """

    answers: dict[str, DecisionAnswer]


def is_retryable_http_error(error: Exception) -> bool:
    """Retry transport failures, rate limits and server errors; never a client error.

    A request the client itself got wrong, such as an illegal header value
    or an unsupported URL scheme, fails the same way every time, so it is not
    retried, though httpx counts it among its transport errors [@httpx2024].
    """
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return (
            status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR
        )
    if isinstance(error, httpx.LocalProtocolError | httpx.UnsupportedProtocol):
        return False
    return isinstance(error, httpx.TransportError)


def check_key_characters(key: str, *, source: str) -> None:
    """Refuse a key that no HTTP header may carry, naming no part of it.

    httpx refuses a header that holds a control character with an error that
    quotes the whole header, key included [@httpx2024], and that error would
    reach the raised error, the classifier span and the stream.
    """
    if not (key.isascii() and key.isprintable()):
        message = (
            f"{source} holds a control or non-ASCII character, which no HTTP header "
            "may carry; pass the key alone"
        )
        raise ConfigurationError(message)


def read_httpx_url(url: str) -> httpx.URL | None:
    """Return `url` as httpx parses it, or None when httpx cannot read it."""
    try:
        return httpx.URL(url)
    except httpx.InvalidURL:
        return None


def check_url_without_credentials(url: str, *, parameter_name: str) -> None:
    """Refuse a URL with a user name or password as httpx parses it, or that httpx cannot read.

    httpx reads a user name and password in a request's URL as Basic
    authentication, which replaces the bearer key in the `Authorization`
    header, and quotes the whole URL in its own request log and in the
    `HTTPStatusError` it raises [@httpx2024]. So such a URL can never carry
    the key, and puts the password in the log and in the error. A URL httpx
    cannot read is refused too, since no request could be sent to it, and
    httpx's own error can quote part of a password: a `#`, `/` or `?` ends a
    URL's authority, so in
    `https://user:abc#rest@host` httpx reads `abc` as the port and quotes it.
    Neither message quotes any part of the URL, and nothing is chained to it,
    since the refusal is raised outside the handler that caught httpx's error.
    `OpenRouterDecisionModel` states the user names and passwords this check
    cannot see.
    """
    parsed = read_httpx_url(url)
    if parsed is None:
        message = (
            f"{parameter_name} is not a URL httpx can read, and it is not quoted here in "
            "case it holds a password; check its host and port, and pass the key as api_key"
        )
        raise ConfigurationError(message)
    if parsed.userinfo:
        message = (
            f"{parameter_name} holds a user name or password, which httpx would send in "
            "place of the key and quote in its logs: pass the key as api_key, and "
            f"{parameter_name} without them"
        )
        raise ConfigurationError(message)


def has_own_base_url(client: httpx.Client | httpx.AsyncClient | None) -> bool:
    """Return whether `client` was passed with an absolute `base_url`, with a scheme and a host."""
    return client is not None and client.base_url.is_absolute_url


def check_reachable_url(
    url: str,
    *,
    clients: Sequence[httpx.Client | httpx.AsyncClient | None],
    parameter_name: str,
) -> None:
    """Refuse a URL without a scheme and a host that no client passed can complete.

    httpx sends a URL with a scheme and a host as it is, whatever the scheme;
    its own transports refuse a scheme they do not support at the first
    request, and a transport the caller passes may accept it. httpx reads any
    other URL, such as `/v1`, an empty one or `https:///v1`, as relative: it
    puts the URL's path and query after the `base_url` of the client that
    sends it, and drops the rest [@httpx2024]. So such a URL is refused only
    when none of `clients`, those the caller passed, has an absolute
    `base_url`, since a client the model opens has none, and the first
    request would fail. A URL httpx cannot read is refused whatever the
    clients, since no client can complete it. The message quotes no part of
    the URL, not even its scheme, which in `user:password@host` is the user
    name.
    """
    parsed = read_httpx_url(url)
    can_be_completed = any(has_own_base_url(client) for client in clients)
    if parsed is None or not (parsed.is_absolute_url or can_be_completed):
        message = (
            f"{parameter_name} has no scheme and host, and no client passed has an absolute "
            "base_url to complete it; it is not quoted here in case it holds a password. "
            f"Pass the whole URL, such as {OPENROUTER_DECISIONS_URL}, or an http_client or "
            "async_http_client with an absolute base_url of its own"
        )
        raise ConfigurationError(message)


def build_decisions_endpoint(
    base_url: str,
    *,
    clients: Sequence[httpx.Client | httpx.AsyncClient | None],
) -> str:
    """Return `{base_url}/decisions`, once `base_url` holds no credentials and a client can send it.

    `clients` are those the caller passed, whose absolute `base_url` can
    complete a relative `base_url`. The credentials check runs first, so a URL that
    holds them is refused for them, whatever its scheme and the clients.
    """
    check_instance_option(base_url, option_type=str, parameter_name="base_url")
    check_url_without_credentials(base_url, parameter_name="base_url")
    check_reachable_url(base_url, clients=clients, parameter_name="base_url")
    return f"{base_url.rstrip('/')}/decisions"


def read_openrouter_api_key(api_key: SecretStr | None) -> SecretStr:
    """Return the key given, or, when it is None, the one in `OPENROUTER_API_KEY`.

    That variable is the one the chat models read. Either key is stripped,
    since no header may carry a line break, and one that still holds a
    control or non-ASCII character raises `ConfigurationError`, as
    `check_key_characters` explains. A key given blank raises
    `ConfigurationError` rather than fall back to the variable, since a key
    the application meant to pass must not be replaced by another one. So
    does a key given as anything but a `SecretStr`, named by its type alone.
    """
    if api_key is None:
        from_environment = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not from_environment:
            message = "OpenRouterDecisionModel needs api_key or the OPENROUTER_API_KEY variable"
            raise ConfigurationError(message)
        check_key_characters(from_environment, source="OPENROUTER_API_KEY")
        return SecretStr(from_environment)
    if not isinstance(api_key, SecretStr):
        message = f"api_key must be a SecretStr, not {type(api_key).__name__}: pass SecretStr(key)"
        raise ConfigurationError(message)
    given = api_key.get_secret_value().strip()
    if not given:
        message = "api_key is blank: pass a key, or leave it out to read OPENROUTER_API_KEY"
        raise ConfigurationError(message)
    check_key_characters(given, source="api_key")
    return SecretStr(given)


class OpenRouterDecisionModel(DecisionModel):
    """Jev and other decision models through OpenRouter's Decisions API (alpha).

    Each question is sent as a "noul" (a yes/no question) with its criteria,
    all in one POST to `{base_url}/decisions` [@openrouter2026decisions],
    sent with httpx [@httpx2024]. The response is validated with pydantic
    [@pydantic2026]. Transport errors, rate limits and server errors are
    retried with stamina [@schlawack2026stamina]; other HTTP errors raise
    `httpx.HTTPStatusError` at once. stamina logs each retry with a
    `RetriedCallError` in place of httpx's error, as `retries` explains, so
    neither the context, nor the key, nor any part of the reply reaches a log
    or a retry hook.
    Each request is one `monitor classifier`
    span in LangChain tracers, around its retries, with the model and the
    questions as inputs and the answers as outputs; the context stays out,
    since it holds the proposed step.

    Jev returns probabilities rounded to two decimals, so scores tie at a
    resolution of 0.01; averaging with `RepeatedMonitor`, or aggregating
    several questions with `Aggregation.MEAN`, restores some resolution.
    `Aggregation.MAX` and `Aggregation.MIN` return one question's answer, at
    its resolution.

    The key comes from `OPENROUTER_API_KEY` unless `api_key` is given, and a
    blank `api_key` raises `ConfigurationError`. So does a `base_url` that
    holds a user name or password, or that httpx cannot read, as
    `check_url_without_credentials` explains, and, as `check_reachable_url`
    explains, a relative, empty or host-less one when no client passed has an
    absolute `base_url` of its own. No message quotes any part of the URL.
    An unencoded `/`, `?` or `#` in a user name or password hides them from
    httpx, which reads what comes before that character as the host and
    port. Such a `base_url` is refused only when that is no host and port
    httpx can read, as in `https://user:abc#rest@host`;
    `https://user:/rest@host`, `https://user:123/rest@host` and
    `https://user/rest:password@host` are built. Their requests, key
    included, go to a host read from the user name, and logs and errors can
    quote the user name and password with the URL. Keep credentials out of
    `base_url`.

    Pass your own `http_client` or `async_http_client` to reuse connections,
    change transports or decide when a client closes; a client you pass
    keeps its own timeout, and `timeout_seconds` applies only to the clients
    the model opens. Without them, the sync path opens one client for the
    model's lifetime, which is never closed, and the async path opens and
    closes a client per request, since a pooled async client cannot move
    between event loops. httpx puts the path and query of a relative, empty
    or host-less `base_url` after the absolute `base_url` of the client that
    sends it: with `base_url="/v1"` and
    `http_client=httpx.Client(base_url="https://gateway.example/api")`, the
    sync path posts to `https://gateway.example/api/v1/decisions`. Each path
    sends with its own client only, so a path whose client you do not pass
    opens one without a `base_url`, and its first request raises
    `httpx.UnsupportedProtocol`. Pass a client with an absolute `base_url`
    for each path you use.

    Retries stop after `RETRY_ATTEMPTS` attempts, or once an attempt fails 45
    seconds or more after the first began, stamina's default time budget.
    HTTP 408 is a client error like any other, and is not retried.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | None = None,
        base_url: str = OPENROUTER_DECISIONS_URL,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the model; the key is read here, so a missing key fails at once."""
        check_instance_option(
            model,
            option_type=str,
            parameter_name="model",
            hint="Pass the model's OpenRouter id, such as 'typesafe/jev-1.13'.",
        )
        timeout_seconds = read_positive_number_option(
            timeout_seconds,
            parameter_name="timeout_seconds",
        )
        check_optional_instance_option(
            http_client,
            option_type=httpx.Client,
            parameter_name="http_client",
            hint="Pass an httpx.Client, or None for one the model opens.",
        )
        check_optional_instance_option(
            async_http_client,
            option_type=httpx.AsyncClient,
            parameter_name="async_http_client",
            hint="Pass an httpx.AsyncClient, or None for one per request.",
        )
        # The clients are checked first, so reading their base_url cannot fail.
        endpoint = build_decisions_endpoint(base_url, clients=[http_client, async_http_client])
        self.model = model
        self.api_key = read_openrouter_api_key(api_key)
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self.async_http_client = async_http_client

    async def estimate_probabilities(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Ask every question in one request and return the probabilities of yes."""
        body = self.build_request_body(context=context, questions=questions)
        async with open_traced_run(self.build_classifier_span(questions)) as traced_request:
            content = await self.request_decisions(body)
            probabilities = read_decisions_probabilities(content, questions=questions)
            traced_request.outputs = {"answers": probabilities}
        return probabilities

    def estimate_probabilities_sync(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> dict[str, float]:
        """Ask every question in one request, without an event loop."""
        body = self.build_request_body(context=context, questions=questions)
        with open_traced_run_sync(self.build_classifier_span(questions)) as traced_request:
            content = self.request_decisions_sync(body)
            probabilities = read_decisions_probabilities(content, questions=questions)
            traced_request.outputs = {"answers": probabilities}
        return probabilities

    def build_classifier_span(self, questions: Sequence[YesNoQuestion]) -> TraceSpan:
        """Return the span of one request: the model and the questions, without the context."""
        return TraceSpan(
            name=CLASSIFIER_SPAN_NAME,
            inputs={
                "model": self.model,
                "questions": {question.key: question.text for question in questions},
            },
            metadata=MONITOR_WORK_METADATA,
            tags=[MONITOR_TAG],
        )

    def build_request_body(
        self,
        *,
        context: str,
        questions: Sequence[YesNoQuestion],
    ) -> DecisionsRequestBody:
        """Build the request: the context as the state, and one noul per question."""
        return {
            "model": self.model,
            "state": context,
            "questions": {
                question.key: {
                    "type": "noul",
                    "instructions": question.text,
                    "criteria": {"true": question.yes_means, "false": question.no_means},
                }
                for question in questions
            },
        }

    def build_headers(self) -> dict[str, str]:
        """Return the request headers, with the key as a bearer token."""
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    def open_async_client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        """Lend the caller's async client, or open a fresh one for this request."""
        if self.async_http_client is not None:
            return nullcontext(self.async_http_client)
        return httpx.AsyncClient(timeout=self.timeout_seconds)

    async def request_decisions(self, body: DecisionsRequestBody) -> bytes:
        """POST the request, retrying transient failures, and return the response body.

        `call_with_retries` retries the POST [@schlawack2026stamina], so no
        retry hook is handed the body, whose state is the transcript and the
        proposed step, nor httpx's error, whose request holds the body and the
        key; after the last attempt, httpx's error is raised.
        """

        async def post() -> httpx.Response:
            async with self.open_async_client() as client:
                response = await client.post(self.endpoint, json=body, headers=self.build_headers())
            # httpx reads the body before `post` returns, so the response outlives its client.
            response.raise_for_status()
            return response

        response = await call_with_retries(
            post, is_retried=is_retryable_http_error, attempts=RETRY_ATTEMPTS
        )
        return response.content

    def request_decisions_sync(self, body: DecisionsRequestBody) -> bytes:
        """POST the request without an event loop, retrying transient failures, as above."""

        def post() -> httpx.Response:
            response = self.http_client.post(self.endpoint, json=body, headers=self.build_headers())
            response.raise_for_status()
            return response

        response = call_with_retries_sync(
            post, is_retried=is_retryable_http_error, attempts=RETRY_ATTEMPTS
        )
        return response.content


def read_decisions_probabilities(
    content: bytes,
    *,
    questions: Sequence[YesNoQuestion],
) -> dict[str, float]:
    """Validate a Decisions API response and return the probability of yes per question."""
    try:
        decisions = DecisionsResponse.model_validate_json(content)
    except ValidationError as error:
        message = "the Decisions API returned a response in an unexpected shape"
        raise MonitorError(message) from error
    probabilities = {key: answer.noul for key, answer in decisions.answers.items()}
    return select_question_probabilities(probabilities, questions=questions)
