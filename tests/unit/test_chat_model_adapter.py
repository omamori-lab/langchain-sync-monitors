"""One chat model gives hyoshigi a synchronous and an asynchronous generative model.

The two models call `invoke` and `ainvoke`. Each test runs once per path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import pytest
import stamina
from hyoshigi import (
    Generation,
    GenerationOutcome,
    MediaPart,
    MediaTypes,
    Prompt,
    Refused,
    Rejected,
    TextPart,
    Truncated,
    Unavailable,
)
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel, LangSmithParams, ModelProfile
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_sync_monitors.chat_model_adapter import ChatModelAdapter
from langchain_sync_monitors.errors import ConfigurationError
from tests.support.chat_models import ScriptedChatModel
from tests.support.fixtures import every_log_record as every_log_record
from tests.support.fixtures import retry_frame_locals as retry_frame_locals
from tests.support.fixtures import retry_hook_calls as retry_hook_calls
from tests.support.flaky_models import FlakyChatModel
from tests.support.log_records import find_frame_leaks, find_logged_leaks
from tests.support.tracing import RecordingTracer

type CallPath = Literal["async", "sync"]

ADAPTER_LOGGER = "langchain_sync_monitors.chat_model_adapter"
SOURCE = "llm_monitor"
PLANTED_SECRET = "planted-secret-9f3c1e"
"""A credential the prompt quotes, which no retry hook may be able to read."""
PLANTED_REPLY = "planted-reply-5e02d4"
"""A value the provider's error reply holds, which no log line may repeat."""

PROMPT = Prompt(system="Rate the step.", content=(TextPart(text="Step: read q3.md."),))
PROMPT_WITH_IMAGE = Prompt(
    system="Rate the step.",
    content=(
        TextPart(text="Step: show the chart."),
        MediaPart(media_type="image/png", url="https://example.test/chart.png"),
    ),
)
FINE = AIMessage("The step reads the file the user named.")
UNFIXED = pytest.mark.xfail(strict=True, reason="a finding of the review is not fixed yet")


@pytest.fixture(params=["async", "sync"])
def call_path(request: pytest.FixtureRequest) -> CallPath:
    """Run a test once through the asynchronous model and once through the synchronous one."""
    path: CallPath = request.param
    return path


@pytest.fixture(autouse=True)
def library_attempts_without_waiting() -> Iterator[None]:
    """Keep the library's own number of attempts, but skip the waits between them."""
    with stamina.set_testing(True, attempts=14, cap=True):
        yield


async def generate_on_path(
    model: BaseChatModel,
    prompt: Prompt,
    *,
    call_path: CallPath,
) -> GenerationOutcome:
    """Adapt the chat model and generate through `generate` of the model for the path."""
    adapter = ChatModelAdapter(model, source=SOURCE)
    if call_path == "async":
        return await adapter.async_generative_model.generate(prompt)
    return adapter.generative_model.generate(prompt)


class StatusError(Exception):
    """An error in the shape provider SDKs give one: the HTTP status as `status_code`."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class ProviderContextOverflow(StatusError, ContextOverflowError):
    """A context overflow with its SDK error's status, as langchain-anthropic gives it."""


class ProviderRateLimit(StatusError, ModelRateLimitError):
    """A provider's rate limit with its SDK error's status."""


class ProviderServerFailure(StatusError, ModelAPIError):
    """A provider's server failure with its SDK error's status."""


class ProviderInvalidRequest(StatusError, ModelInvalidRequestError):
    """A provider's bad request with its SDK error's status."""


@dataclass(eq=False)
class ReplyHoldingError(Exception):
    """An error in the shape OpenRouter's SDK gives one: a dataclass whose repr holds the reply."""

    status_code: int
    body: str


@dataclass(eq=False)
class ReplyHoldingRateLimit(ModelRateLimitError):
    """A typed rate limit without a status, whose repr holds the provider's reply."""

    body: str


class ResponseStatusError(Exception):
    """An error in the shape Google's SDK gives one: the HTTP status only on its response."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
        self.response = httpx.Response(status_code, request=request)


class ProviderResponseFailure(ResponseStatusError, ModelAPIError):
    """A provider's server failure that holds its status only on its response."""


def build_no_response_error() -> Exception:
    """Build the error the OpenRouter SDK raises when no HTTP response arrives."""
    errors = pytest.importorskip("openrouter.errors")
    error: Exception = errors.NoResponseError("No response received")
    return error


def build_google_server_error(status_code: int) -> Exception:
    """Build the error langchain-google-genai raises for a server failure, as it builds it."""
    chat_models = pytest.importorskip("langchain_google_genai.chat_models")
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta")
    response = httpx.Response(status_code, request=request)
    details = {"error": {"code": status_code, "message": "The service is unavailable."}}
    error: Exception = chat_models.GoogleAPIError(
        code=status_code,
        response_json=details,
        response=response,
    )
    return error


def build_gemini_block_reply(*, block_reason: str | None, message: str | None) -> AIMessage:
    """Build the reply langchain-google-genai gives when Gemini returns no candidate.

    The integration returns an empty message and keeps the prompt feedback in its
    metadata. Only the call to Google's client is replaced, by one that answers with a
    response of no candidates and the given feedback.
    """
    types = pytest.importorskip("google.genai.types")
    integration = pytest.importorskip("langchain_google_genai")
    reason = None if block_reason is None else types.BlockedReason(block_reason)
    feedback = types.GenerateContentResponsePromptFeedback(
        block_reason=reason,
        block_reason_message=message,
    )
    response = types.GenerateContentResponse(candidates=[], prompt_feedback=feedback)
    chat = integration.ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="offline")
    chat.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response))
    reply: AIMessage = chat.invoke("Step: perform the action.")
    return reply


def build_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build the error httpx raises for an HTTP status."""
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"status {status_code}", request=request, response=response)


class LabelledChatModel(ScriptedChatModel):
    """A scripted model that reports the given provider and model name to LangSmith."""

    labels: LangSmithParams

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return self.labels


class SidedChatModel(BaseChatModel):
    """Replies `invoke` to a synchronous call and `ainvoke` to an asynchronous one."""

    @property
    def _llm_type(self) -> str:
        return "sided"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("invoke"))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ainvoke"))])


async def test_each_model_calls_its_own_side_of_the_chat_model(call_path: CallPath) -> None:
    # Arrange
    expected = {"async": "ainvoke", "sync": "invoke"}[call_path]

    # Act
    outcome = await generate_on_path(SidedChatModel(), PROMPT, call_path=call_path)

    # Assert
    assert outcome == Generation(text=expected)


async def test_a_prompt_reaches_the_chat_model_as_a_system_message_and_text_blocks(
    call_path: CallPath,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[FINE])
    prompt = Prompt(
        system="Rate the step.",
        content=(TextPart(text=" First.\n"), TextPart(text="Second.")),
    )

    # Act
    await generate_on_path(model, prompt, call_path=call_path)

    # Assert
    expected = HumanMessage(
        content=[{"type": "text", "text": " First.\n"}, {"type": "text", "text": "Second."}],
    )
    assert model.calls == [[SystemMessage("Rate the step."), expected]]


MEDIA_BLOCKS = [
    pytest.param(
        MediaPart(media_type="image/png", url="https://example.test/chart.png"),
        {"type": "image", "url": "https://example.test/chart.png", "mime_type": "image/png"},
        id="image-by-url",
    ),
    pytest.param(
        MediaPart(media_type="Image/PNG", url="https://example.test/chart.png"),
        {"type": "image", "url": "https://example.test/chart.png", "mime_type": "image/png"},
        id="image-type-in-capitals",
    ),
    pytest.param(
        MediaPart(media_type="image/png", data=b"\x89PNG"),
        {"type": "image", "base64": "iVBORw==", "mime_type": "image/png"},
        id="image-bytes",
    ),
    pytest.param(
        MediaPart(media_type="image/*", url="https://example.test/chart"),
        {"type": "image", "url": "https://example.test/chart"},
        id="image-range-by-url",
    ),
    pytest.param(
        MediaPart(media_type="audio/mpeg", url="https://example.test/call.mp3"),
        {"type": "audio", "url": "https://example.test/call.mp3", "mime_type": "audio/mpeg"},
        id="audio-by-url",
    ),
    pytest.param(
        MediaPart(media_type="video/mp4", url="https://example.test/run.mp4"),
        {"type": "video", "url": "https://example.test/run.mp4", "mime_type": "video/mp4"},
        id="video-by-url",
    ),
    pytest.param(
        MediaPart(media_type="application/pdf", data=b"%PDF"),
        {"type": "file", "base64": "JVBERg==", "mime_type": "application/pdf"},
        id="pdf-bytes",
    ),
    pytest.param(
        MediaPart(media_type="text/csv", url="https://example.test/q3.csv"),
        {"type": "file", "url": "https://example.test/q3.csv", "mime_type": "text/csv"},
        id="csv-by-url",
    ),
]


@pytest.mark.parametrize(("part", "block"), MEDIA_BLOCKS)
async def test_a_media_part_reaches_the_chat_model_as_a_standard_content_block(
    call_path: CallPath,
    part: MediaPart,
    block: dict[str, str],
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[FINE])
    prompt = Prompt(system="Rate the step.", content=(TextPart(text="Look."), part))

    # Act
    await generate_on_path(model, prompt, call_path=call_path)

    # Assert
    expected = HumanMessage(content=[{"type": "text", "text": "Look."}, block])
    assert model.calls == [[SystemMessage("Rate the step."), expected]]


async def test_a_text_part_without_a_visible_character_is_left_out(call_path: CallPath) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[FINE])
    prompt = Prompt(system="Rate the step.", content=(TextPart(text=" \n"), TextPart(text="Step.")))

    # Act
    await generate_on_path(model, prompt, call_path=call_path)

    # Assert
    expected = HumanMessage(content=[{"type": "text", "text": "Step."}])
    assert model.calls == [[SystemMessage("Rate the step."), expected]]


async def test_a_blank_system_prompt_is_not_sent(call_path: CallPath) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[FINE])
    prompt = Prompt(system="", content=(TextPart(text="Step."),))

    # Act
    await generate_on_path(model, prompt, call_path=call_path)

    # Assert
    assert model.calls == [[HumanMessage(content=[{"type": "text", "text": "Step."}])]]


UNSENDABLE_PROMPTS = [
    pytest.param(
        Prompt(
            system="Rate the step.", content=(MediaPart(media_type="image/*", data=b"\x89PNG"),)
        ),
        id="bytes-of-a-media-range",
    ),
    pytest.param(
        Prompt(system="Rate the step.", content=(TextPart(text=" \n"),)),
        id="only-blank-text",
    ),
    pytest.param(Prompt(system="Rate the step.", content=()), id="no-content"),
]


@pytest.mark.parametrize("prompt", UNSENDABLE_PROMPTS)
async def test_a_prompt_that_cannot_be_sent_is_unavailable_and_makes_no_call(
    call_path: CallPath,
    prompt: Prompt,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[FINE])

    # Act
    outcome = await generate_on_path(model, prompt, call_path=call_path)

    # Assert
    assert outcome == Unavailable()
    assert model.calls == []


COMPLETE_REPLIES = [
    pytest.param(AIMessage("Fine.", response_metadata={"finish_reason": "stop"}), id="openai"),
    pytest.param(AIMessage("Fine.", response_metadata={"stop_reason": "end_turn"}), id="anthropic"),
    pytest.param(AIMessage("Fine.", response_metadata={"finish_reason": "STOP"}), id="gemini"),
    pytest.param(AIMessage("Fine.", response_metadata={"stopReason": "end_turn"}), id="bedrock"),
    pytest.param(AIMessage("Fine.", response_metadata={"done_reason": "stop"}), id="ollama"),
    pytest.param(AIMessage("Fine.", additional_kwargs={"refusal": None}), id="openai-no-refusal"),
    pytest.param(AIMessage("Fine.", additional_kwargs={"refusal": ""}), id="openai-empty-refusal"),
    pytest.param(AIMessage("Fine."), id="no-stop-reason"),
]


@pytest.mark.parametrize("reply", COMPLETE_REPLIES)
async def test_a_reply_that_stopped_on_its_own_is_a_generation_of_its_text(
    call_path: CallPath,
    reply: AIMessage,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Generation(text="Fine.")


async def test_a_generation_holds_the_text_blocks_of_the_reply_and_no_other_block(
    call_path: CallPath,
) -> None:
    # Arrange
    reply = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "The step reads a file."},
            {"type": "text", "text": "Score 3."},
        ],
    )
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Generation(text="Score 3.")


CUT_OFF_METADATA = [
    pytest.param({"finish_reason": "length"}, id="openai"),
    pytest.param({"native_finish_reason": "length"}, id="openrouter"),
    pytest.param({"finish_reason": "MAX_TOKENS"}, id="gemini"),
    pytest.param({"stop_reason": "max_tokens"}, id="anthropic"),
    pytest.param({"stop_reason": "model_context_window_exceeded"}, id="anthropic-window"),
    pytest.param({"stopReason": "max_tokens"}, id="bedrock"),
    pytest.param({"done_reason": "length"}, id="ollama"),
    pytest.param({"incomplete_details": {"reason": "max_output_tokens"}}, id="openai-responses"),
    pytest.param({"finish_reason": "MAX_TOKENSMAX_TOKENS"}, id="gemini-merged-chunks"),
]


@pytest.mark.parametrize("metadata", CUT_OFF_METADATA)
async def test_a_reply_cut_off_at_a_length_limit_is_truncated_with_its_text_so_far(
    call_path: CallPath,
    metadata: dict[str, Any],
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[AIMessage("The step reads", response_metadata=metadata)])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Truncated(text="The step reads")


REFUSED_REPLIES = [
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "content_filter"}),
        Refused(),
        id="openai-content-filter",
    ),
    pytest.param(
        AIMessage("", response_metadata={"stop_reason": "refusal"}),
        Refused(),
        id="anthropic-refusal",
    ),
    pytest.param(
        AIMessage(
            "",
            response_metadata={
                "stop_reason": "refusal",
                "stop_details": {
                    "type": "refusal",
                    "category": "cyber",
                    "explanation": "The request could enable cyber harm.",
                },
            },
        ),
        Refused(explanation="The request could enable cyber harm."),
        id="anthropic-refusal-with-explanation",
    ),
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "SAFETY"}),
        Refused(),
        id="gemini-safety",
    ),
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "BLOCKLIST"}),
        Refused(),
        id="gemini-blocklist",
    ),
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "PROHIBITED_CONTENT"}),
        Refused(),
        id="gemini-prohibited-content",
    ),
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "SPII"}),
        Refused(),
        id="gemini-spii",
    ),
    pytest.param(
        AIMessage("", response_metadata={"stopReason": "content_filtered"}),
        Refused(),
        id="bedrock-content-filtered",
    ),
    pytest.param(
        AIMessage("", response_metadata={"stopReason": "guardrail_intervened"}),
        Refused(),
        id="bedrock-guardrail-intervened",
    ),
    pytest.param(
        AIMessage(
            "",
            additional_kwargs={"refusal": "I can't help with that."},
            response_metadata={"finish_reason": "stop"},
        ),
        Refused(explanation="I can't help with that."),
        id="openai-refusal-message",
    ),
    pytest.param(
        AIMessage(content=[{"type": "refusal", "refusal": "I can't help with that."}]),
        Refused(explanation="I can't help with that."),
        id="openai-responses-refusal-block",
    ),
    pytest.param(
        AIMessage(content=[{"type": "refusal"}]),
        Refused(),
        id="refusal-block-without-text",
    ),
    pytest.param(
        AIMessage(content=[{"type": "refusal", "refusal": ""}]),
        Refused(),
        id="refusal-block-with-empty-text",
    ),
]


@pytest.mark.parametrize(("reply", "expected"), REFUSED_REPLIES)
async def test_a_refusal_or_a_content_filter_stop_is_refused_with_the_explanation_if_given(
    call_path: CallPath,
    reply: AIMessage,
    expected: Refused,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == expected


async def test_a_reply_both_refused_and_cut_off_is_refused(call_path: CallPath) -> None:
    # Arrange
    reply = AIMessage(
        "",
        additional_kwargs={"refusal": "No."},
        response_metadata={"finish_reason": "length"},
    )
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Refused(explanation="No.")


GEMINI_BLOCKS = [
    pytest.param(
        "SAFETY",
        "Blocked by safety filters",
        Refused(explanation="Blocked by safety filters"),
        id="safety-with-message",
    ),
    pytest.param("SAFETY", None, Refused(), id="safety-without-message"),
    pytest.param("SAFETY", "", Refused(), id="safety-with-empty-message"),
    pytest.param("OTHER", None, Refused(), id="other"),
    pytest.param("BLOCKLIST", None, Refused(), id="blocklist"),
    pytest.param("PROHIBITED_CONTENT", None, Refused(), id="prohibited-content"),
]


@UNFIXED
@pytest.mark.parametrize(("block_reason", "message", "expected"), GEMINI_BLOCKS)
async def test_a_prompt_that_gemini_blocks_is_refused_with_its_message_if_given(
    call_path: CallPath,
    block_reason: str,
    message: str | None,
    expected: Refused,
) -> None:
    # Arrange
    reply = build_gemini_block_reply(block_reason=block_reason, message=message)
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == expected


@pytest.mark.parametrize("block_reason", [None, "BLOCKED_REASON_UNSPECIFIED"])
async def test_an_empty_gemini_reply_with_no_block_reason_is_an_empty_generation(
    call_path: CallPath,
    block_reason: str | None,
) -> None:
    # Arrange
    reply = build_gemini_block_reply(block_reason=block_reason, message=None)
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Generation(text="")


PROVIDER_ERROR_REPLIES = [
    pytest.param(
        AIMessage(
            "Provider encountered an error during generation.",
            response_metadata={"finish_reason": "error"},
        ),
        id="openrouter-error",
    ),
    pytest.param(
        AIMessage("", response_metadata={"finish_reason": "errorerror"}),
        id="openrouter-error-merged-chunks",
    ),
    pytest.param(
        AIMessage(
            "partial",
            response_metadata={"finish_reason": "error", "native_finish_reason": "length"},
        ),
        id="error-before-cut-off",
    ),
    pytest.param(
        AIMessage(
            "",
            additional_kwargs={"refusal": "No."},
            response_metadata={"finish_reason": "error"},
        ),
        id="error-before-refusal",
    ),
]


@pytest.mark.parametrize("reply", PROVIDER_ERROR_REPLIES)
async def test_a_reply_that_stopped_on_a_provider_error_is_unavailable(
    call_path: CallPath,
    reply: AIMessage,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[reply])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Unavailable()


FAILURE_OUTCOMES = [
    pytest.param(lambda: StatusError(500), Unavailable(status=500), id="status-500"),
    pytest.param(lambda: StatusError(503), Unavailable(status=503), id="status-503"),
    pytest.param(lambda: StatusError(408), Unavailable(status=408), id="status-408"),
    pytest.param(lambda: StatusError(409), Unavailable(status=409), id="status-409"),
    pytest.param(lambda: build_http_status_error(502), Unavailable(status=502), id="httpx-502"),
    pytest.param(
        lambda: ProviderServerFailure(529),
        Unavailable(status=529),
        id="typed-server-failure",
    ),
    pytest.param(lambda: ModelAPIError(), Unavailable(), id="typed-server-failure-without-status"),
    pytest.param(lambda: ModelTimeoutError(), Unavailable(), id="typed-timeout"),
    pytest.param(lambda: ModelConnectionError(), Unavailable(), id="typed-connection-error"),
    pytest.param(lambda: httpx.ReadTimeout("timed out"), Unavailable(), id="httpx-timeout"),
    pytest.param(lambda: httpx.ConnectError("refused"), Unavailable(), id="httpx-connect-error"),
    pytest.param(lambda: TimeoutError(), Unavailable(), id="builtin-timeout"),
    pytest.param(lambda: ConnectionResetError(), Unavailable(), id="builtin-connection-error"),
    pytest.param(build_no_response_error, Unavailable(), id="openrouter-no-response"),
    pytest.param(
        lambda: ProviderResponseFailure(503),
        Unavailable(status=503),
        id="typed-server-failure-with-status-on-response",
    ),
    pytest.param(
        lambda: build_google_server_error(503),
        Unavailable(status=503),
        id="google-server-error",
    ),
    pytest.param(
        lambda: ResponseStatusError(413),
        Rejected(status=413),
        id="status-413-on-response",
    ),
    pytest.param(lambda: ProviderContextOverflow(400), Rejected(status=400), id="context-overflow"),
    pytest.param(lambda: ContextOverflowError(), Rejected(), id="context-overflow-without-status"),
    pytest.param(lambda: StatusError(413), Rejected(status=413), id="status-413"),
    pytest.param(lambda: build_http_status_error(413), Rejected(status=413), id="httpx-413"),
]


@pytest.mark.parametrize(("build_error", "expected"), FAILURE_OUTCOMES)
async def test_a_failure_of_the_provider_is_an_outcome_by_its_kind_and_status_and_not_retried(
    call_path: CallPath,
    build_error: Callable[[], Exception],
    expected: Unavailable | Rejected,
) -> None:
    # Arrange: a reply waits behind the error, so a second attempt would give a generation
    model = FlakyChatModel(replies=[build_error(), FINE])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == expected


RATE_LIMITS = [
    pytest.param(lambda: StatusError(429), id="status-429"),
    pytest.param(lambda: build_http_status_error(429), id="httpx-429"),
    pytest.param(lambda: ProviderRateLimit(429), id="typed-rate-limit"),
    pytest.param(lambda: ModelRateLimitError(), id="typed-rate-limit-without-status"),
    pytest.param(lambda: ResponseStatusError(429), id="status-429-on-response"),
]


@pytest.mark.parametrize("build_error", RATE_LIMITS)
async def test_a_rate_limited_call_is_made_again(
    call_path: CallPath,
    build_error: Callable[[], Exception],
) -> None:
    # Arrange
    model = FlakyChatModel(replies=[build_error(), FINE])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Generation(text="The step reads the file the user named.")
    assert model.started_calls == 2


@pytest.mark.parametrize("build_error", RATE_LIMITS)
async def test_a_rate_limit_on_four_attempts_is_unavailable_with_status_429(
    call_path: CallPath,
    build_error: Callable[[], Exception],
) -> None:
    # Arrange: a reply waits behind the fourth attempt, so a fifth would give a generation
    model = FlakyChatModel(replies=[build_error() for _ in range(4)] + [FINE])

    # Act
    outcome = await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert outcome == Unavailable(status=429)
    assert model.started_calls == 4


REJECTED_WITH_MEDIA = [
    pytest.param(lambda: StatusError(400), Rejected(status=400), id="status-400"),
    pytest.param(lambda: StatusError(422), Rejected(status=422), id="status-422"),
    pytest.param(lambda: build_http_status_error(400), Rejected(status=400), id="httpx-400"),
    pytest.param(lambda: ProviderInvalidRequest(400), Rejected(status=400), id="typed-400"),
    pytest.param(lambda: ModelInvalidRequestError(), Rejected(), id="typed-without-status"),
]


@pytest.mark.parametrize(("build_error", "expected"), REJECTED_WITH_MEDIA)
async def test_a_bad_request_for_a_prompt_that_holds_media_is_rejected(
    call_path: CallPath,
    build_error: Callable[[], Exception],
    expected: Rejected,
) -> None:
    # Arrange
    model = FlakyChatModel(replies=[build_error(), FINE])

    # Act
    outcome = await generate_on_path(model, PROMPT_WITH_IMAGE, call_path=call_path)

    # Assert
    assert outcome == expected


RAISED_FAILURES = [
    pytest.param(lambda: StatusError(400), id="status-400-on-text"),
    pytest.param(lambda: StatusError(422), id="status-422-on-text"),
    pytest.param(lambda: ProviderInvalidRequest(400), id="typed-400-on-text"),
    pytest.param(lambda: ModelInvalidRequestError(), id="typed-invalid-request-on-text"),
    pytest.param(lambda: StatusError(401), id="status-401"),
    pytest.param(lambda: StatusError(402), id="status-402"),
    pytest.param(lambda: StatusError(499), id="status-499"),
    pytest.param(lambda: StatusError(403), id="status-403"),
    pytest.param(lambda: StatusError(404), id="status-404"),
    pytest.param(lambda: build_http_status_error(404), id="httpx-404"),
    pytest.param(lambda: ModelAuthenticationError(), id="typed-authentication"),
    pytest.param(lambda: ModelPermissionDeniedError(), id="typed-permission-denied"),
    pytest.param(lambda: ModelNotFoundError(), id="typed-not-found"),
    pytest.param(
        lambda: TypeError("create() got an unexpected keyword argument 'logprobs'"),
        id="unsupported-keyword",
    ),
    pytest.param(lambda: ValueError("bad"), id="value-error"),
    pytest.param(lambda: RuntimeError("bug"), id="runtime-error"),
]


@pytest.mark.parametrize("build_error", RAISED_FAILURES)
async def test_any_other_failure_is_raised_as_it_is_and_not_retried(
    call_path: CallPath,
    build_error: Callable[[], Exception],
) -> None:
    # Arrange: a reply waits behind the error, so a second attempt would give a generation
    error = build_error()
    model = FlakyChatModel(replies=[error, FINE])

    # Act
    with pytest.raises(type(error)) as raised:
        await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    assert raised.value is error
    assert model.started_calls == 1


REPLY_HOLDING_RATE_LIMITS = [
    pytest.param(
        lambda: ReplyHoldingError(status_code=429, body=f"Reply quoting {PLANTED_REPLY}."),
        id="status-429",
    ),
    pytest.param(
        lambda: ReplyHoldingRateLimit(body=f"Reply quoting {PLANTED_REPLY}."),
        id="typed-rate-limit-without-status",
    ),
]


@pytest.mark.parametrize("build_error", REPLY_HOLDING_RATE_LIMITS)
async def test_no_frame_a_retry_hook_can_read_shows_the_prompt_or_the_providers_reply(
    call_path: CallPath,
    build_error: Callable[[], Exception],
    retry_frame_locals: list[dict[str, str]],
) -> None:
    # Arrange: the prompt quotes the secret, and the provider's error holds its whole reply
    content = (TextPart(text=f"Deploy with the token {PLANTED_SECRET}."),)
    prompt = Prompt(system="Rate the step.", content=content)
    model = FlakyChatModel(replies=[build_error(), FINE])

    # Act
    await generate_on_path(model, prompt, call_path=call_path)

    # Assert: the hook read the frames that hold the call and the kept error
    (frame_locals,) = retry_frame_locals
    suffix = "" if call_path == "async" else "_sync"
    assert {f"run_attempt{suffix}.block", f"run_attempt{suffix}.failures"} <= frame_locals.keys()

    # Assert: and no local in them quotes the prompt or any part of the reply
    secrets = [PLANTED_SECRET, PLANTED_REPLY]
    assert find_frame_leaks(retry_frame_locals, secrets=secrets) == []


LOGGED_FAILURES = [
    pytest.param(
        lambda: ReplyHoldingError(status_code=503, body=f"Reply quoting {PLANTED_REPLY}."),
        "A chat model call failed with ReplyHoldingError (HTTP 503) "
        "and is reported as unavailable.",
        id="unavailable-with-status",
    ),
    pytest.param(
        lambda: ReplyHoldingError(status_code=413, body=f"Reply quoting {PLANTED_REPLY}."),
        "A chat model call failed with ReplyHoldingError (HTTP 413) and is reported as rejected.",
        id="rejected-with-status",
    ),
    pytest.param(
        lambda: httpx.ConnectError(f"refused, quoting {PLANTED_REPLY}"),
        "A chat model call failed with ConnectError (no HTTP status) "
        "and is reported as unavailable.",
        id="unavailable-without-status",
    ),
]


@pytest.mark.parametrize(("build_error", "expected"), LOGGED_FAILURES)
async def test_a_failure_turned_into_an_outcome_is_warned_of_by_its_type_and_status_alone(
    call_path: CallPath,
    every_log_record: list[logging.LogRecord],
    build_error: Callable[[], Exception],
    expected: str,
) -> None:
    # Arrange: the error's text and repr hold the provider's reply
    error = build_error()
    model = FlakyChatModel(replies=[error])

    # Act
    await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    lines = [
        (record.levelname, record.getMessage())
        for record in every_log_record
        if record.name == ADAPTER_LOGGER
    ]
    assert PLANTED_REPLY in repr(error)
    assert lines == [("WARNING", expected)]
    assert find_logged_leaks(every_log_record, secrets=[PLANTED_REPLY]) == []


async def test_the_chat_model_is_called_as_a_monitor_call_from_the_given_source(
    call_path: CallPath,
) -> None:
    # Arrange
    tracer = RecordingTracer()
    model = ScriptedChatModel(responses=[FINE], callbacks=[tracer])

    # Act
    await generate_on_path(model, PROMPT, call_path=call_path)

    # Assert
    runs = tracer.find_runs("monitor call")
    assert len(runs) == 1
    assert runs[0].metadata["lc_source"] == SOURCE
    assert runs[0].metadata["ls_message_view_exclude"] is True


NAMED_MODELS = [
    pytest.param(
        {"ls_provider": "anthropic", "ls_model_name": "claude-sonnet-4-5"},
        "anthropic:claude-sonnet-4-5",
        id="provider-and-model",
    ),
    pytest.param({"ls_provider": "acme"}, "acme", id="provider-alone"),
    pytest.param({"ls_model_name": "model-1"}, "model-1", id="model-alone"),
    pytest.param({"ls_provider": "", "ls_model_name": "model-1"}, "model-1", id="empty-provider"),
    pytest.param({}, "LabelledChatModel", id="neither"),
]


@pytest.mark.parametrize(("labels", "expected"), NAMED_MODELS)
def test_both_models_are_named_by_the_provider_and_model_of_the_chat_model(
    labels: LangSmithParams,
    expected: str,
) -> None:
    # Arrange
    model = LabelledChatModel(responses=[], labels=labels)

    # Act
    adapter = ChatModelAdapter(model, source=SOURCE)

    # Assert
    assert adapter.generative_model.name == expected
    assert adapter.async_generative_model.name == expected


def test_a_provider_string_gives_a_chat_model_named_by_its_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    pytest.importorskip("langchain_openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")

    # Act
    adapter = ChatModelAdapter("openrouter:xiaomi/mimo-v2.6-pro", source=SOURCE)

    # Assert
    assert adapter.generative_model.name == "openrouter:xiaomi/mimo-v2.6-pro"
    assert adapter.async_generative_model.name == "openrouter:xiaomi/mimo-v2.6-pro"


PERCEIVED_MEDIA = [
    pytest.param(None, MediaTypes(), id="no-profile"),
    pytest.param({}, MediaTypes(), id="empty-profile"),
    pytest.param({"text_inputs": True}, MediaTypes(), id="text-inputs-only"),
    pytest.param({"image_inputs": True}, MediaTypes("image/*"), id="images"),
    pytest.param(
        {"image_inputs": True, "audio_inputs": True, "video_inputs": True, "pdf_inputs": True},
        MediaTypes("image/*", "audio/*", "video/*", "application/pdf"),
        id="images-audio-video-and-pdf",
    ),
    pytest.param(
        {"image_inputs": False, "pdf_inputs": True},
        MediaTypes("application/pdf"),
        id="pdf-without-images",
    ),
]


@pytest.mark.parametrize(("profile", "expected"), PERCEIVED_MEDIA)
def test_both_models_perceive_the_media_the_profile_of_the_chat_model_lists(
    profile: ModelProfile | None,
    expected: MediaTypes,
) -> None:
    # Arrange
    model = ScriptedChatModel(responses=[], profile=profile)

    # Act
    adapter = ChatModelAdapter(model, source=SOURCE)

    # Assert
    assert adapter.generative_model.perceives == expected
    assert adapter.async_generative_model.perceives == expected


def test_the_adapter_holds_the_chat_model_it_is_given() -> None:
    # Arrange
    model = ScriptedChatModel(responses=[])

    # Act
    adapter = ChatModelAdapter(model, source=SOURCE)

    # Assert
    assert adapter.chat_model is model


def test_a_chat_model_wrapped_in_a_runnable_is_a_configuration_error() -> None:
    # Arrange
    wrapped = ScriptedChatModel(responses=[]).with_retry()

    # Act and Assert
    with pytest.raises(ConfigurationError, match="Pass the chat model itself"):
        ChatModelAdapter(wrapped, source=SOURCE)  # ty: ignore[invalid-argument-type]


def test_a_source_that_is_not_text_is_a_configuration_error() -> None:
    # Arrange
    model = ScriptedChatModel(responses=[])

    # Act and Assert
    with pytest.raises(ConfigurationError, match="source must be a str"):
        ChatModelAdapter(model, source=7)  # ty: ignore[invalid-argument-type]
