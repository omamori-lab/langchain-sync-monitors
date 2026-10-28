"""OpenRouter's Decisions API as hyoshigi's decision models, over httpx.

hyoshigi's own OpenRouter decision models take the `openrouter` SDK at 1.3.63
or later, and langchain-openrouter 0.2.9, which the `openrouter` extra
installs, pins `openrouter<1.0.0`, so the two cannot share an environment.
This client calls the Decisions API (alpha) [@openrouter2026decisions] with
httpx [@httpx2024] instead, and stays until langchain-openrouter supports
openrouter 1.x.

Each statement is one request, holding one "noul" question whose
instructions are the statement and a state of the question's text parts, as
hyoshigi asks it [@hyoshigi2026]. The key comes from `OPENROUTER_API_KEY`. A
transport failure, a rate limit or a server error is retried with stamina
[@schlawack2026stamina]; after the last attempt it is `Unavailable`, with its
status. A reply that holds no probability from 0 to 1 for the statement is
`Unavailable` too, a 403 is `Refused` and a 413 `Rejected`, as hyoshigi's
OpenRouter models read them. Any other client error raises
`ConfigurationError` naming only its status. stamina logs each retry with a
`RetriedCallError` in place of httpx's error, as `retries` explains, so
neither the state, nor the key, nor any part of the reply reaches a log or a
retry hook.
"""

from __future__ import annotations

import os
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Final, Literal, TypedDict

import httpx
from hyoshigi import (
    AsyncDecisionModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    Question,
    Refused,
    Rejected,
    TextPart,
    Unavailable,
)
from pydantic import BaseModel, Field, ValidationError

from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.options import check_key_characters
from langchain_sync_monitors.retries import call_with_retries, call_with_retries_sync

DECISIONS_ENDPOINT: Final = "https://openrouter.ai/api/alpha/decisions"
"""Where a Decisions API request goes."""

REQUEST_TIMEOUT_SECONDS: Final = 30.0
"""How long one request may take, connecting and reading included."""

RETRY_ATTEMPTS: Final = 5
"""How many times a request is sent before its failure is the outcome."""

QUESTION_NAME: Final = "statement"
"""The name of the one question a request holds."""

UNAVAILABLE_STATUSES: Final = frozenset(
    {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.CONFLICT, HTTPStatus.TOO_MANY_REQUESTS},
)
"""The client error statuses that a later request can clear, as hyoshigi reads them."""


class NoulQuestionBody(TypedDict):
    """One question ("noul") in a Decisions API request."""

    type: Literal["noul"]
    instructions: str


class DecisionsRequestBody(TypedDict):
    """The body of a Decisions API request."""

    model: str
    state: list[str]
    questions: dict[str, NoulQuestionBody]


class DecisionAnswer(BaseModel):
    """One answer from the Decisions API: the probability the statement holds."""

    type: Literal["noul"]
    # Strict, since pydantic's lax mode reads `true`, `false` and strings such as "0.5" as numbers
    # [@pydantic2026], and `false` would pass as 0. A JSON integer such as 0 or 1 is still read.
    noul: float = Field(strict=True, ge=0.0, le=1.0)


class DecisionsResponse(BaseModel):
    """The part of a Decisions API response the client reads: the answers.

    pydantic validates this external payload [@pydantic2026]; the fields
    the client does not read are ignored, so a change in their shape cannot
    discard a valid answer.
    """

    answers: dict[str, DecisionAnswer]


def read_openrouter_api_key() -> str:
    """Return the key in `OPENROUTER_API_KEY`, stripped, or raise `ConfigurationError`.

    A missing or blank key raises, and so does one that holds a character no
    HTTP header may carry, naming no part of it.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        message = "an 'openrouter:' decision model needs the OPENROUTER_API_KEY variable"
        raise ConfigurationError(message)
    check_key_characters(key, source="OPENROUTER_API_KEY")
    return key


def build_request_body(question: Question, *, model: str) -> DecisionsRequestBody:
    """Build a request of one noul question, whose state is the question's text parts."""
    return {
        "model": model,
        "state": [part.text for part in question.state if isinstance(part, TextPart)],
        "questions": {QUESTION_NAME: {"type": "noul", "instructions": question.statement}},
    }


def is_retried(error: Exception) -> bool:
    """Tell whether a request's failure is retried: a rate limit, a server error or a transport
    failure, never another client error.
    """
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR
    return isinstance(error, httpx.TransportError)


def read_reply(content: bytes) -> DecisionOutcome:
    """Return the probability a reply gives the statement, or `Unavailable` for any other reply."""
    try:
        reply = DecisionsResponse.model_validate_json(content)
    except ValidationError:
        return Unavailable()
    answer = reply.answers.get(QUESTION_NAME)
    return Unavailable() if answer is None else Decision(probability=answer.noul)


def read_failure(error: Exception) -> DecisionOutcome:
    """Return the outcome of a request's last failure, or raise what no outcome covers.

    A transport failure is unavailable with no status. A 403 is refused, a
    413 rejected, and a 408, 409, 429 or server error unavailable with its
    status. Any other client error raises `ConfigurationError` naming only
    its status, and any other exception is raised as it is.
    """
    if isinstance(error, httpx.TransportError):
        return Unavailable()
    if not isinstance(error, httpx.HTTPStatusError):
        raise error
    return read_status(error.response.status_code)


def read_status(status: int) -> DecisionOutcome:
    """Return the outcome of a failed status, or raise `ConfigurationError` naming it alone."""
    if status == HTTPStatus.FORBIDDEN:
        return Refused()
    if status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return Rejected(status=status)
    if status in UNAVAILABLE_STATUSES or status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return Unavailable(status=status)
    message = f"OpenRouter answered HTTP {status}"
    raise ConfigurationError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionsClient:
    """One OpenRouter model's requests to the Decisions API, sent with or without an event loop.

    The sync side posts with `client`. The async side posts with
    `async_client`, or opens a client for each request when it is None,
    since a pooled async client cannot move between event loops.
    """

    model: str
    api_key: str = field(repr=False)
    client: httpx.Client
    async_client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        """The request headers, with the key as a bearer token."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def post(self, body: DecisionsRequestBody) -> httpx.Response:
        """Post one request, and raise httpx's error for a failed status."""
        response = self.client.post(DECISIONS_ENDPOINT, json=body, headers=self.headers)
        return response.raise_for_status()

    def open_async_client(self) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        """Lend the async client given, or open a fresh one for this request."""
        if self.async_client is not None:
            return nullcontext(self.async_client)
        return httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    async def post_async(self, body: DecisionsRequestBody) -> httpx.Response:
        """Post one request with an event loop, and raise httpx's error for a failed status."""
        async with self.open_async_client() as client:
            response = await client.post(DECISIONS_ENDPOINT, json=body, headers=self.headers)
        # httpx reads the body before the request returns, so the response outlives its client.
        return response.raise_for_status()

    def decide(self, question: Question) -> DecisionOutcome:
        """Ask whether the statement holds, retrying a transient failure, and read the outcome."""
        body = build_request_body(question, model=self.model)
        try:
            response = call_with_retries_sync(
                lambda: self.post(body),
                is_retried=is_retried,
                attempts=RETRY_ATTEMPTS,
            )
        except Exception as error:
            return read_failure(error)
        return read_reply(response.content)

    async def decide_async(self, question: Question) -> DecisionOutcome:
        """Ask as `decide` does, awaiting each request."""
        body = build_request_body(question, model=self.model)
        try:
            response = await call_with_retries(
                lambda: self.post_async(body),
                is_retried=is_retried,
                attempts=RETRY_ATTEMPTS,
            )
        except Exception as error:
            return read_failure(error)
        return read_reply(response.content)


def build_decision_models(client: DecisionsClient) -> tuple[DecisionModel, AsyncDecisionModel]:
    """Return the sync and async decision models on a client, named `openrouter:<model id>`."""
    name = f"openrouter:{client.model}"
    return (
        DecisionModel(name=name, decide=client.decide),
        AsyncDecisionModel(name=name, decide=client.decide_async),
    )


def open_decision_models(model: str) -> tuple[DecisionModel, AsyncDecisionModel]:
    """Return the decision models of one OpenRouter model, on the key in the environment.

    The sync model posts with one client, opened here and never closed.
    """
    client = DecisionsClient(
        model=model,
        api_key=read_openrouter_api_key(),
        client=httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS),
    )
    return build_decision_models(client)
