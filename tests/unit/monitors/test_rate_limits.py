"""Chat monitors call the model again after a rate limit, HTTP 429, and after nothing else."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator

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


async def test_chat_openrouter_gets_a_rate_limited_call_made_again(
    monitor_input: MonitorInput,
    call_path: CallPath,
) -> None:
    # Arrange: the OpenRouter SDK under ChatOpenRouter, with the retries max_retries=2 sets
    openrouter = pytest.importorskip("openrouter")
    langchain_openrouter = pytest.importorskip("langchain_openrouter")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}})
        return httpx.Response(200, json=OPENROUTER_REPLY)

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
    model = langchain_openrouter.ChatOpenRouter(
        model="provider/monitor-model",
        api_key="offline-placeholder",
        client=client,
    )

    # Act
    verdict = await evaluate_on_path(build_llm_monitor(model), monitor_input, call_path=call_path)

    # Assert
    assert verdict.suspicion == pytest.approx(0.3)
    assert len(requests) == 2
