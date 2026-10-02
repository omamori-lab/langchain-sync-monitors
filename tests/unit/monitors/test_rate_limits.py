"""Chat monitors call the model again after a rate limit, HTTP 429, and after nothing else."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx
import pytest
import stamina
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from stamina.instrumentation import RetryDetails

from langchain_sync_monitors.contracts import Monitor, MonitorInput
from langchain_sync_monitors.monitors.chat import RATE_LIMIT_ATTEMPTS, LLMMonitor
from langchain_sync_monitors.monitors.guard import GuardModelMonitor, GuardScoring
from tests.support.flaky_models import FlakyChatModel
from tests.support.log_records import find_leaks, find_logged_leaks

from .doubles import PLANTED_SECRET, CallPath, evaluate_on_path

CALM_REPLY = "<reasoning>It reads the file the user named.</reasoning>\n<score>3</score>"
POLICY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Flag steps that send data out. End with violation or no_violation."),
        ("human", "{transcript}\n{proposed_step}"),
    ],
)


class ProviderStatusError(Exception):
    """An error in the shape provider SDKs give one: the HTTP status as `status_code`."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def build_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build the error httpx raises for an HTTP status."""
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"status {status_code}", request=request, response=response)


PLANTED_USER_ID = "user_planted-3c7a91"
"""The account's id, which OpenRouter puts in the body of an error reply as `user_id`."""

PLANTED_HEADER_VALUE = "planted-header-5e02d4"
"""A value in a header of the provider's error reply."""

PLANTED_REPLY_VALUES = (PLANTED_USER_ID, PLANTED_HEADER_VALUE, PLANTED_SECRET)
"""What the provider's error reply holds: the account's id, a header and the request's text."""

RATE_LIMIT_REPLY = {
    "error": {
        "code": 429,
        "message": "Rate limit exceeded",
        "metadata": {"raw": f"Deploy with the token {PLANTED_SECRET}."},
    },
    "user_id": PLANTED_USER_ID,
}
"""OpenRouter's reply to a rate-limited call, whose metadata quotes the request's text."""


def build_rate_limit_response() -> httpx.Response:
    """Build the provider's HTTP 429 reply, with the planted values in its body and headers."""
    request = httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    return httpx.Response(
        429,
        headers={"x-request-id": PLANTED_HEADER_VALUE},
        json=RATE_LIMIT_REPLY,
        request=request,
    )


def build_openrouter_rate_limit() -> Exception:
    """Build the error OpenRouter's SDK raises on HTTP 429, which holds the whole reply."""
    errors = pytest.importorskip("openrouter.errors")
    response = build_rate_limit_response()
    data = errors.TooManyRequestsResponseErrorData.model_validate(RATE_LIMIT_REPLY)
    error: Exception = errors.TooManyRequestsResponseError(data, response, response.text)
    return error


@dataclass(eq=False)
class ReplyHoldingError(Exception):
    """An error in the shape OpenRouter's SDK gives one: a dataclass whose repr holds the reply."""

    status_code: int
    body: str
    headers: dict[str, str]


def build_reply_holding_rate_limit() -> Exception:
    """Build an HTTP 429 error whose repr holds the reply's body and headers, without the SDK."""
    response = build_rate_limit_response()
    return ReplyHoldingError(status_code=429, body=response.text, headers=dict(response.headers))


REPLY_HOLDING_RATE_LIMITS = [
    pytest.param(build_openrouter_rate_limit, id="openrouter-error"),
    pytest.param(build_reply_holding_rate_limit, id="reply-holding-error"),
]


@pytest.fixture(autouse=True)
def library_attempts_without_waiting() -> Iterator[None]:
    """Keep the library's own number of attempts, but skip the waits between them."""
    with stamina.set_testing(True, attempts=RATE_LIMIT_ATTEMPTS + 10, cap=True):
        yield


def build_llm_monitor(model: BaseChatModel) -> Monitor:
    """Return an LLM monitor over `model` that never asks again for an unreadable reply."""
    return LLMMonitor(model=model, max_parse_retries=0)


def build_guard(model: BaseChatModel) -> Monitor:
    """Return a guard over `model` that draws three samples."""
    return GuardModelMonitor(
        model=model,
        policy_prompt=POLICY_PROMPT,
        suspicious_labels={"violation"},
        safe_labels={"no_violation"},
        scoring=GuardScoring.SAMPLE_FRACTION,
        samples=3,
    )


RATE_LIMITS = [
    pytest.param(lambda: ProviderStatusError(429), id="provider-error-429"),
    pytest.param(lambda: build_http_status_error(429), id="httpx-429"),
]
OTHER_FAILURES = [
    pytest.param(lambda: ProviderStatusError(500), id="provider-error-500"),
    pytest.param(lambda: ProviderStatusError(400), id="provider-error-400"),
    pytest.param(lambda: build_http_status_error(503), id="httpx-503"),
    pytest.param(lambda: RuntimeError("429 rate limited"), id="no-status"),
]


@pytest.mark.parametrize("build_error", RATE_LIMITS)
async def test_a_rate_limited_call_is_made_again(
    monitor_input: MonitorInput,
    call_path: CallPath,
    build_error: Callable[[], Exception],
) -> None:
    # Arrange
    model = FlakyChatModel(replies=[build_error(), AIMessage(CALM_REPLY)])

    # Act
    verdict = await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.3)
    assert model.started_calls == 2


@pytest.mark.parametrize("build_error", OTHER_FAILURES)
async def test_a_call_that_fails_otherwise_is_not_made_again(
    monitor_input: MonitorInput,
    call_path: CallPath,
    build_error: Callable[[], Exception],
) -> None:
    # Arrange: a network or server error is the chat model's own `max_retries`' business
    error = build_error()
    model = FlakyChatModel(replies=[error, AIMessage(CALM_REPLY)])

    # Act
    with pytest.raises(type(error)) as raised:
        await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert
    assert raised.value is error
    assert model.started_calls == 1


async def test_a_rate_limit_on_every_attempt_is_raised_after_the_last(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: a reply waits behind the last attempt, so a further attempt would succeed
    rate_limits: list[object] = [ProviderStatusError(429) for _ in range(RATE_LIMIT_ATTEMPTS)]
    model = FlakyChatModel(replies=[*rate_limits, AIMessage(CALM_REPLY)])

    # Act
    with pytest.raises(ProviderStatusError) as raised:
        await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert
    assert raised.value is rate_limits[-1]
    assert model.started_calls == RATE_LIMIT_ATTEMPTS == 4


async def test_each_retry_after_a_rate_limit_waits_longer_from_one_second(
    monitor_input: MonitorInput,
    call_path: CallPath,
    retry_details: list[RetryDetails],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: stamina computes its real waits, and the sleeps themselves are skipped
    sleep_on_the_loop = asyncio.sleep

    async def skip_the_wait(_delay: float) -> None:
        await sleep_on_the_loop(0)

    monkeypatch.setattr(asyncio, "sleep", skip_the_wait)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    model = FlakyChatModel(replies=[ProviderStatusError(429) for _ in range(RATE_LIMIT_ATTEMPTS)])

    # Act
    with stamina.set_testing(False), pytest.raises(ProviderStatusError):
        await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert: each wait doubles from one second, with up to one second of jitter, up to five
    first, second, third = (details.wait_for for details in retry_details)
    assert 1.0 <= first <= 2.0
    assert 2.0 <= second <= 3.0
    assert 4.0 <= third <= 5.0


async def test_a_guard_draws_again_only_the_rate_limited_sample(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: the first call is rate limited; under ainvoke the other two are already drawn
    replies = [ProviderStatusError(429), *(AIMessage(label) for label in ("violation",) * 3)]
    model = FlakyChatModel(replies=replies)

    # Act
    verdict = await evaluate_on_path(build_guard(model), monitor_input, call_path=call_path)

    # Assert: three samples from four calls, not a fresh batch of three
    assert verdict.suspicion == 1.0
    assert model.started_calls == 4
    assert model.finished_calls == 3


@pytest.mark.parametrize(
    "build_monitor", [build_llm_monitor, build_guard], ids=["llm-monitor", "guard"]
)
async def test_a_retried_rate_limit_logs_no_part_of_the_prompt(
    input_holding_a_secret: MonitorInput,
    call_path: CallPath,
    build_monitor: Callable[[BaseChatModel], Monitor],
    retry_details: list[RetryDetails],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.DEBUG)
    reply = AIMessage(CALM_REPLY if build_monitor is build_llm_monitor else "no_violation")
    model = FlakyChatModel(replies=[ProviderStatusError(429), reply, reply, reply])

    # Act
    await evaluate_on_path(build_monitor(model), input_holding_a_secret, call_path=call_path)

    # Assert
    (details,) = retry_details
    assert (details.name, details.args, details.kwargs) == ("<context block>", (), {})
    assert "stamina.retry_scheduled" in [record.getMessage() for record in caplog.records]
    for text in [repr(details.caused_by), *(repr(vars(record)) for record in caplog.records)]:
        assert PLANTED_SECRET not in text


@pytest.mark.parametrize("build_error", REPLY_HOLDING_RATE_LIMITS)
@pytest.mark.parametrize(
    "build_monitor", [build_llm_monitor, build_guard], ids=["llm-monitor", "guard"]
)
async def test_a_retried_rate_limit_logs_no_part_of_the_providers_reply(
    input_holding_a_secret: MonitorInput,
    call_path: CallPath,
    build_monitor: Callable[[BaseChatModel], Monitor],
    build_error: Callable[[], Exception],
    retry_details: list[RetryDetails],
    every_log_record: list[logging.LogRecord],
) -> None:
    # Arrange: the provider's error holds its reply, which quotes the account, a header and the
    # request's text
    error = build_error()
    reply = AIMessage(CALM_REPLY if build_monitor is build_llm_monitor else "no_violation")
    model = FlakyChatModel(replies=[error, reply, reply, reply])

    # Act
    await evaluate_on_path(build_monitor(model), input_holding_a_secret, call_path=call_path)

    # Assert: the error's repr held every planted value, and stamina logged its retry
    assert [value for value in PLANTED_REPLY_VALUES if value not in repr(error)] == []
    assert "stamina.retry_scheduled" in [record.getMessage() for record in every_log_record]

    # Assert: no record on any logger, and nothing a retry hook is handed, holds one
    (details,) = retry_details
    assert find_logged_leaks(every_log_record, secrets=PLANTED_REPLY_VALUES) == []
    assert find_leaks(details.caused_by, secrets=PLANTED_REPLY_VALUES) == []


@pytest.mark.parametrize("build_error", REPLY_HOLDING_RATE_LIMITS)
async def test_a_retried_rate_limit_is_logged_by_its_status_and_type_alone(
    monitor_input: MonitorInput,
    call_path: CallPath,
    build_error: Callable[[], Exception],
    retry_details: list[RetryDetails],
    every_log_record: list[logging.LogRecord],
) -> None:
    # Arrange
    error = build_error()
    model = FlakyChatModel(replies=[error, AIMessage(CALM_REPLY)])

    # Act
    await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert: the retry names the status and the provider's error type, with nothing chained
    expected = f"RateLimitedCallError(status_code=429, error_type={type(error).__name__!r})"
    (details,) = retry_details
    (record,) = [record for record in every_log_record if record.name == "stamina"]
    assert repr(details.caused_by) == vars(record)["stamina.caused_by"] == expected
    assert (details.caused_by.__cause__, details.caused_by.__context__) == (None, None)


@pytest.mark.parametrize("build_error", REPLY_HOLDING_RATE_LIMITS)
async def test_a_rate_limit_on_every_attempt_raises_the_providers_error_and_logs_none_of_it(
    input_holding_a_secret: MonitorInput,
    call_path: CallPath,
    build_error: Callable[[], Exception],
    retry_details: list[RetryDetails],
    every_log_record: list[logging.LogRecord],
) -> None:
    # Arrange: a reply waits behind the last attempt, so a further attempt would succeed
    rate_limits = [build_error() for _ in range(RATE_LIMIT_ATTEMPTS)]
    model = FlakyChatModel(replies=[*rate_limits, AIMessage(CALM_REPLY)])
    monitor = build_llm_monitor(model)

    # Act
    with pytest.raises(type(rate_limits[-1])) as raised:
        await evaluate_on_path(monitor, input_holding_a_secret, call_path=call_path)

    # Assert: the caller gets the provider's own last error, with nothing chained to it
    assert raised.value is rate_limits[-1]
    assert (raised.value.__cause__, raised.value.__context__) == (None, None)
    assert model.started_calls == RATE_LIMIT_ATTEMPTS

    # Assert: and no retry before it logged any part of a reply
    assert len(retry_details) == RATE_LIMIT_ATTEMPTS - 1
    assert find_logged_leaks(every_log_record, secrets=PLANTED_REPLY_VALUES) == []


OPENROUTER_REPLY = {
    "id": "gen-1",
    "object": "chat.completion",
    "created": 0,
    "model": "provider/monitor-model",
    "system_fingerprint": None,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": CALM_REPLY},
        },
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def build_chat_openrouter(respond: Callable[[httpx.Request], httpx.Response]) -> BaseChatModel:
    """Build `ChatOpenRouter` over the OpenRouter SDK, with the retries `max_retries=2` sets.

    Every request, sync or async, goes to `respond`, never to the network.
    """
    openrouter = pytest.importorskip("openrouter")
    langchain_openrouter = pytest.importorskip("langchain_openrouter")
    transport = httpx.MockTransport(respond)
    retries = openrouter.utils.RetryConfig(
        "backoff",
        openrouter.utils.BackoffStrategy(500, 60_000, 1.5, 300_000),
        retry_connection_errors=True,
    )
    client = openrouter.OpenRouter(
        api_key="offline-placeholder",
        server_url="https://openrouter.test/api/v1",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
        retry_config=retries,
    )
    model: BaseChatModel = langchain_openrouter.ChatOpenRouter(
        model="provider/monitor-model",
        api_key="offline-placeholder",
        client=client,
    )
    return model


async def test_chat_openrouter_gets_a_rate_limited_call_made_again(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}})
        return httpx.Response(200, json=OPENROUTER_REPLY)

    model = build_chat_openrouter(respond)

    # Act
    verdict = await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.3)
    assert len(requests) == 2


async def test_chat_openrouter_logs_no_part_of_a_rate_limit_reply(
    input_holding_a_secret: MonitorInput,
    call_path: CallPath,
    retry_details: list[RetryDetails],
    every_log_record: list[logging.LogRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: OpenRouter's 429 reply quotes the account, a header and the request's text, and
    # the SDK's own debug log stays off, as it is unless OPENROUTER_DEBUG is set
    monkeypatch.delenv("OPENROUTER_DEBUG", raising=False)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            headers = {"x-request-id": PLANTED_HEADER_VALUE}
            return httpx.Response(429, headers=headers, json=RATE_LIMIT_REPLY)
        return httpx.Response(200, json=OPENROUTER_REPLY)

    model = build_chat_openrouter(respond)

    # Act
    verdict = await evaluate_on_path(
        build_llm_monitor(model), input_holding_a_secret, call_path=call_path
    )

    # Assert: the SDK's own error reached the monitor, which made the call again
    assert verdict.suspicion == pytest.approx(0.3)
    assert len(requests) == 2
    (details,) = retry_details
    expected = "RateLimitedCallError(status_code=429, error_type='TooManyRequestsResponseError')"
    assert repr(details.caused_by) == expected

    # Assert: no record on any logger, and nothing a retry hook is handed, holds a planted value
    assert find_logged_leaks(every_log_record, secrets=PLANTED_REPLY_VALUES) == []
    assert find_leaks(details.caused_by, secrets=PLANTED_REPLY_VALUES) == []
