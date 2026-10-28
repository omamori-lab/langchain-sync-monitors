"""A LangChain chat model as hyoshigi's synchronous and asynchronous generative models.

hyoshigi's monitors rate an action through a `GenerativeModel`, or an
`AsyncGenerativeModel` for the asynchronous twin: a name, a function from a
prompt to an outcome, and the media the model perceives [@hyoshigi2026].
`ChatModelAdapter` builds both from one chat model, so a configuration that
names a chat model serves `invoke` and `ainvoke` alike. The outcomes are the
ones hyoshigi's own provider models give: a failure the provider causes is a
value, and any other failure is raised.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from http import HTTPStatus

import httpx
from hyoshigi import (
    AsyncGenerativeModel,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Refused,
    Rejected,
    Truncated,
    Unavailable,
)
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput, ModelProfile
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_sync_monitors._langchain import read_chat_model_labels
from langchain_sync_monitors.errors import ConfigurationError
from langchain_sync_monitors.log_probabilities import (
    add_log_probabilities,
    build_request_options,
    is_rejected_keyword,
)
from langchain_sync_monitors.model_calls import build_internal_call_config, resolve_chat_model
from langchain_sync_monitors.options import check_instance_option, read_count_option
from langchain_sync_monitors.prompt_messages import build_messages, has_media_part
from langchain_sync_monitors.retries import (
    call_with_retries,
    call_with_retries_sync,
    read_http_status,
)

logger = logging.getLogger(__name__)

STOP_REASON_KEYS = (
    "finish_reason",
    "native_finish_reason",
    "stop_reason",
    "stopReason",
    "done_reason",
)
"""Where providers put why a reply stopped.

OpenAI, OpenRouter and Gemini use the first two, Anthropic the third,
Bedrock Converse the fourth and Ollama the last.
"""

CUT_OFF_STOP_REASONS = ("length", "max_tokens", "max_output_tokens", "context_window_exceeded")
"""Stop reasons that mean the reply hit a length limit before the model finished it."""

REFUSAL_STOP_REASONS = (
    "content_filter",
    "refusal",
    "safety",
    "blocklist",
    "prohibited_content",
    "spii",
    "recitation",
    "language",
    "guardrail_intervened",
)
"""Stop reasons that mean the provider declined to answer, or filtered what it wrote.

OpenAI's chat completions report `content_filter` [@openaipythonsdk2026],
Anthropic `refusal` [@anthropic2026stopreasons], Gemini `SAFETY`, `BLOCKLIST`,
`PROHIBITED_CONTENT`, `SPII`, `RECITATION` and `LANGUAGE`
[@google2026generatecontent], and Bedrock Converse `content_filtered` and
`guardrail_intervened` [@aws2026converse]. A stop reason counts when it
contains a reason, so `recitation` covers `IMAGE_RECITATION`.
"""

FAILED_STOP_REASONS = ("error",)
"""Stop reasons that mean the provider failed while it wrote the reply.

OpenRouter reports a failure upstream as `finish_reason: "error"`, one of the
values of its `ChatFinishReasonEnum` [@openrouterpythonsdk2026].
"""

RATE_LIMIT_ATTEMPTS = 4
"""How many times a call is tried in all while the provider answers HTTP 429."""
RATE_LIMIT_FIRST_WAIT_SECONDS = 1.0
"""The wait before the first retry of a rate-limited call; each later wait doubles, up to 5 s."""

PROFILE_MEDIA = {
    "image_inputs": "image/*",
    "audio_inputs": "audio/*",
    "video_inputs": "video/*",
    "pdf_inputs": "application/pdf",
}
"""The media a model profile says a chat model takes, by the profile key that says so."""

UNAVAILABLE_STATUSES = frozenset(
    {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.CONFLICT, HTTPStatus.TOO_MANY_REQUESTS},
)
"""The client error statuses that a later call can clear, as hyoshigi's models read them."""

BAD_REQUEST_STATUSES = frozenset({HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY})
"""The statuses of a request the provider found invalid, which a media part can cause."""

NO_ANSWER_ERRORS = (httpx.TransportError, TimeoutError, ConnectionError)
"""The errors of a call that got no reply: a timeout or a connection that failed."""


def is_rate_limit_error(error: Exception) -> bool:
    """Tell whether a chat model's call failed on a rate limit, HTTP 429.

    langchain-core types the error as a `ModelRateLimitError`, which provider
    integrations raise and which some give no status [@langchaincore2026].
    Otherwise the status decides. A chat model's own `max_retries` does not
    always cover one: `ChatOpenRouter` hands its retries to the OpenRouter SDK
    [@langchainopenrouter2026], which retries a chat completion on HTTP 5xx
    and network errors alone, and whose retry settings name no status code
    [@openrouterpythonsdk2026]. Provider SDKs put the status on their errors
    as `status_code`, OpenRouter's, OpenAI's and Anthropic's among them, and
    the last two retry a 429 themselves too
    [@openaipythonsdk2026; @anthropicpythonsdk2026]; httpx puts the status
    on the error's response [@httpx2024].
    """
    return (
        isinstance(error, ModelRateLimitError)
        or read_http_status(error) == HTTPStatus.TOO_MANY_REQUESTS
    )


def has_stop_reason(reply: AIMessage, *, reasons: tuple[str, ...]) -> bool:
    """Tell whether a stop reason the provider reports contains one of the reasons.

    Merging streamed chunks can repeat a value, as in `"MAX_TOKENSMAX_TOKENS"`,
    so a stop reason counts when it contains a reason, in any case.
    """
    metadata = reply.response_metadata
    stop_reasons = [metadata.get(key) for key in STOP_REASON_KEYS]
    incomplete_details = metadata.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        stop_reasons.append(incomplete_details.get("reason"))
    return any(
        isinstance(stop_reason, str) and any(reason in stop_reason.lower() for reason in reasons)
        for stop_reason in stop_reasons
    )


def is_reply_cut_off(reply: AIMessage) -> bool:
    """Tell whether the provider reports that the reply stopped at a length limit.

    OpenAI and OpenRouter report `finish_reason: "length"`, Gemini
    `finish_reason: "MAX_TOKENS"`, Anthropic `stop_reason: "max_tokens"` or
    `"model_context_window_exceeded"`, and OpenRouter keeps the upstream value
    in `native_finish_reason`; the OpenAI Responses API reports
    `incomplete_details: {"reason": "max_output_tokens"}`. `ChatBedrockConverse`
    keeps the Converse response's `stopReason`, `"max_tokens"` or
    `"model_context_window_exceeded"` [@langchainaws2026; @aws2026converse],
    and `ChatOllama` keeps Ollama's `done_reason: "length"`
    [@langchainollama2026]. Merging streamed
    chunks can repeat a value, as in `"MAX_TOKENSMAX_TOKENS"`, so a stop
    reason counts when it contains a cut-off reason.
    """
    return has_stop_reason(reply, reasons=CUT_OFF_STOP_REASONS)


def find_refusal_blocks(reply: AIMessage) -> Iterator[dict[str, object]]:
    """Yield each block of the reply's content that is of the type `refusal`.

    langchain-openai gives a Responses API refusal such a block, with the
    refusal text under `refusal` [@langchainopenai2026].
    """
    if isinstance(reply.content, list):
        for block in reply.content:
            if isinstance(block, dict) and block.get("type") == "refusal":
                yield block


def read_prompt_feedback(reply: AIMessage) -> dict[str, object]:
    """Return the prompt feedback in the reply's metadata, or an empty one.

    When Gemini returns no candidate, langchain-google-genai gives an empty
    reply and keeps the feedback on the prompt there, with the reason it was
    blocked [@langchaingooglegenai2026].
    """
    feedback = reply.response_metadata.get("prompt_feedback")
    return feedback if isinstance(feedback, dict) else {}


def is_prompt_blocked(reply: AIMessage) -> bool:
    """Tell whether the prompt feedback gives a reason Gemini blocked the prompt.

    The reason is a string, such as `SAFETY` or `BLOCKLIST`. A reply with none,
    or with the unspecified value, which the API documents as unused, is not
    blocked [@google2026generatecontent].
    """
    reason = read_prompt_feedback(reply).get("block_reason")
    return isinstance(reason, str) and bool(reason) and "unspecified" not in reason.lower()


def find_refusal_texts(reply: AIMessage) -> Iterator[object]:
    """Yield what the reply holds where a provider puts its explanation of a refusal.

    That is the `refusal` of a chat completion's message, which
    langchain-openai keeps in `additional_kwargs`, and the text of a `refusal`
    block [@langchainopenai2026], the `explanation` of Anthropic's
    `stop_details` [@anthropicpythonsdk2026], and the `block_reason_message`
    of the feedback on a blocked prompt.
    """
    yield reply.additional_kwargs.get("refusal")
    for block in find_refusal_blocks(reply):
        yield block.get("refusal")
    stop_details = reply.response_metadata.get("stop_details")
    if isinstance(stop_details, dict):
        yield stop_details.get("explanation")
    if is_prompt_blocked(reply):
        yield read_prompt_feedback(reply).get("block_reason_message")


def has_refusal_marker(reply: AIMessage) -> bool:
    """Tell whether the reply says, without an explanation, that the provider refused.

    That is a stop reason that names a refusal (`REFUSAL_STOP_REASONS`), a
    `refusal` block, whether or not it has text, or a blocked prompt.
    """
    return (
        has_stop_reason(reply, reasons=REFUSAL_STOP_REASONS)
        or next(find_refusal_blocks(reply), None) is not None
        or is_prompt_blocked(reply)
    )


def read_refusal(reply: AIMessage) -> Refused | None:
    """Return the refusal a reply holds, or `None` when it holds none.

    A reply is a refusal when it carries a marker of one or a provider's
    explanation of one. The explanation is the first of those that is a
    non-empty string.
    """
    explanation = next(
        (text for text in find_refusal_texts(reply) if isinstance(text, str) and text),
        None,
    )
    if explanation is None and not has_refusal_marker(reply):
        return None
    return Refused(explanation=explanation)


def read_reply(reply: AIMessage) -> Generation | Truncated | Refused | Unavailable:
    """Read a reply as unavailable if the provider failed while writing it, else as a refusal.

    A reply that is neither is read as cut off at a length limit, else as
    complete. The provider's failure comes first: the text of a reply that
    failed is not an answer, whatever else the metadata says. The text is that
    of the reply's text blocks alone: reasoning and tool calls are not part of
    it.
    """
    if has_stop_reason(reply, reasons=FAILED_STOP_REASONS):
        return Unavailable()
    refusal = read_refusal(reply)
    if refusal is not None:
        return refusal
    text = str(reply.text)
    if is_reply_cut_off(reply):
        return Truncated(text=text)
    return Generation(text=text)


def read_typed_failure(
    error: Exception,
    *,
    status: int | None,
    holds_media: bool,
) -> Rejected | Unavailable | None:
    """Read the failure langchain-core types, or return `None` if it types it otherwise.

    A context overflow is rejected, and so is a bad request when the prompt
    holds media. A rate limit is unavailable with status 429, which
    langchain-core defines it by, whether or not the error carries one; so is
    any other error it calls retryable, with the status the error carries
    [@langchaincore2026].
    """
    match error:
        case ContextOverflowError():
            return Rejected(status=status)
        case ModelRateLimitError():
            return Unavailable(status=int(HTTPStatus.TOO_MANY_REQUESTS))
        case ModelInvalidRequestError() if holds_media:
            return Rejected(status=status)
        case ModelError(is_retryable=True):
            return Unavailable(status=status)
    return None


def is_no_response_error(error: Exception) -> bool:
    """Tell whether the error is the OpenRouter SDK's `NoResponseError`.

    The SDK raises it when a call gets no HTTP response, and it is neither a
    transport error nor a status error [@openrouterpythonsdk2026]. An instance
    can only exist once the SDK has loaded its errors module, so the class is
    taken from the modules already loaded, and nothing is imported: the
    `openrouter` extra stays optional, and unloaded until a call uses it.
    """
    errors = sys.modules.get("openrouter.errors")
    no_response_error = getattr(errors, "NoResponseError", None)
    return no_response_error is not None and isinstance(error, no_response_error)


def is_no_answer_error(error: Exception) -> bool:
    """Tell whether the error is of a call that got no reply: a timeout or a connection failure."""
    return isinstance(error, NO_ANSWER_ERRORS) or is_no_response_error(error)


def is_rejected_status(status: int, *, holds_media: bool) -> bool:
    """Tell whether a status says the prompt cannot be used: a 413, or a 400 or 422 about media."""
    return status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE or (
        holds_media and status in BAD_REQUEST_STATUSES
    )


def read_status_failure(
    error: Exception,
    *,
    status: int | None,
    holds_media: bool,
) -> Rejected | Unavailable | None:
    """Read a failure by its HTTP status, or by its type when it has none.

    A 413 is rejected, and so is a 400 or a 422 when the prompt holds media.
    A 408, a 409, a 429 or a 5xx is unavailable with its status, and so is a
    timeout or a connection that failed, with none; the OpenRouter SDK's
    `NoResponseError` is one. Any other failure is `None`.
    """
    if status is None:
        return Unavailable() if is_no_answer_error(error) else None
    if is_rejected_status(status, holds_media=holds_media):
        return Rejected(status=status)
    if status in UNAVAILABLE_STATUSES or status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return Unavailable(status=status)
    return None


def read_failure(error: Exception, *, holds_media: bool) -> Rejected | Unavailable | None:
    """Read a chat model's failure as an outcome, or return `None` for a failure to raise.

    The errors langchain-core types are read first, and the HTTP status of
    the rest. The prompt causes a context overflow, an oversize body and a bad
    request about media, so each is rejected; a failure that gives no answer is
    unavailable. A 401, a 403, a 404, a bad request about a text prompt and an
    exception of any other kind are a misuse or a bug, and are raised. The
    rules are those of hyoshigi's own provider models, so the same prompt gets
    the same outcome on either path [@hyoshigi2026].
    """
    status = read_http_status(error)
    outcome = read_typed_failure(error, status=status, holds_media=holds_media)
    if outcome is None:
        outcome = read_status_failure(error, status=status, holds_media=holds_media)
    return outcome


def settle_failure(error: Exception, *, holds_media: bool) -> Rejected | Unavailable:
    """Return the outcome of a chat model's failure, or raise the failure as it is.

    A failure turned into an outcome is logged by its type and status alone,
    since its message and repr can hold the provider's reply.
    """
    outcome = read_failure(error, holds_media=holds_media)
    if outcome is None:
        raise error
    logger.warning(
        "A chat model call failed with %s (%s) and is reported as %s.",
        type(error).__name__,
        "no HTTP status" if outcome.status is None else f"HTTP {outcome.status}",
        type(outcome).__name__.lower(),
    )
    return outcome


def name_chat_model(model: BaseChatModel) -> str:
    """Name a chat model by its provider and model name, such as `anthropic:claude-sonnet-4-5`.

    A model that reports only one gives it alone, and one that reports neither
    gives its class name.
    """
    labels = read_chat_model_labels(model)
    parts = [
        label
        for label in (labels.get("ls_provider"), labels.get("ls_model_name"))
        if isinstance(label, str) and label
    ]
    return ":".join(parts) or type(model).__name__


def read_perceived_media(profile: ModelProfile | None) -> MediaTypes:
    """Return the media a model profile says the chat model takes; text only without a profile.

    A profile key counts only when it is `True`. LangChain's model profiles
    are a beta feature that lists what a model takes by input kind
    [@langchaincore2026].
    """
    flags: Mapping[str, object] = profile or {}
    return MediaTypes(*(media for key, media in PROFILE_MEDIA.items() if flags.get(key) is True))


class ChatModelAdapter:
    """One chat model as a `GenerativeModel` and an `AsyncGenerativeModel`.

    The two models share the chat model, the name and the media it perceives.
    The synchronous one calls `invoke` and the asynchronous one `ainvoke`,
    each with the internal-call config of the `source`, which names the call
    and keeps it out of LangSmith's Trajectory view. Each call that the
    provider answers with HTTP 429 is made again, up to `RATE_LIMIT_ATTEMPTS`
    attempts in all, as `call_with_retries` retries; every other error is left
    to the chat model's own `max_retries`.

    A failure is an outcome when the prompt or the provider's state causes it
    (`read_failure` gives the rules), and any other is raised as it is.

    At a `top_log_probabilities` above 0, each call asks the chat model for
    that many alternatives to every token, and both models declare the count,
    which a hyoshigi guard scores from [@hyoshigi2026choosing]. A complete
    reply carries them as `Generation.log_probabilities`; one whose
    log-probabilities are not in the documented shape is unavailable, and one
    without any carries none. At 0 nothing is asked for and nothing is read.
    `log_probabilities` gives the request and the format.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        source: str,
        top_log_probabilities: int = 0,
    ) -> None:
        """Resolve the chat model and build both generative models from it.

        `source` goes into the `lc_source` metadata of every call and must be
        a string. `top_log_probabilities` must be a whole number of at least 0.
        A model that is neither a string nor a chat model raises
        `ConfigurationError`, as `resolve_chat_model` says.
        """
        check_instance_option(source, option_type=str, parameter_name="source")
        self.top_log_probabilities = read_count_option(
            top_log_probabilities,
            parameter_name="top_log_probabilities",
            minimum=0,
        )
        self.chat_model = resolve_chat_model(model)
        options = build_request_options(self.chat_model, count=self.top_log_probabilities)
        self.runnable: Runnable[LanguageModelInput, AIMessage] = (
            self.chat_model.bind(**options) if options else self.chat_model
        )
        self.call_config = build_internal_call_config(source=source)
        name = name_chat_model(self.chat_model)
        perceives = read_perceived_media(self.chat_model.profile)
        self.generative_model = GenerativeModel(
            name=name,
            generate=self.generate_sync,
            perceives=perceives,
            top_log_probabilities=self.top_log_probabilities,
        )
        self.async_generative_model = AsyncGenerativeModel(
            name=name,
            generate=self.generate,
            perceives=perceives,
            top_log_probabilities=self.top_log_probabilities,
        )

    async def generate(self, prompt: Prompt) -> GenerationOutcome:
        """Call `ainvoke` with the prompt, and read the reply or the failure as an outcome."""
        messages = build_messages(prompt)
        if messages is None:
            return Unavailable()

        async def call_model() -> AIMessage:
            return await self.runnable.ainvoke(messages, config=self.call_config)

        try:
            reply = await call_with_retries(
                call_model,
                is_retried=is_rate_limit_error,
                attempts=RATE_LIMIT_ATTEMPTS,
                wait_initial=RATE_LIMIT_FIRST_WAIT_SECONDS,
            )
        except Exception as error:
            return self.settle_error(error, holds_media=has_media_part(prompt))
        return self.read_outcome(reply)

    def generate_sync(self, prompt: Prompt) -> GenerationOutcome:
        """Call `invoke` with the prompt, and read the reply or the failure as an outcome."""
        messages = build_messages(prompt)
        if messages is None:
            return Unavailable()

        def call_model() -> AIMessage:
            return self.runnable.invoke(messages, config=self.call_config)

        try:
            reply = call_with_retries_sync(
                call_model,
                is_retried=is_rate_limit_error,
                attempts=RATE_LIMIT_ATTEMPTS,
                wait_initial=RATE_LIMIT_FIRST_WAIT_SECONDS,
            )
        except Exception as error:
            return self.settle_error(error, holds_media=has_media_part(prompt))
        return self.read_outcome(reply)

    def read_outcome(self, reply: AIMessage) -> GenerationOutcome:
        """Read the reply as `read_reply` does, and add its log-probabilities to a generation.

        Only a complete reply is read for them, and only at a count above 0.
        """
        outcome = read_reply(reply)
        if self.top_log_probabilities == 0 or not isinstance(outcome, Generation):
            return outcome
        return add_log_probabilities(outcome, reply=reply)

    def settle_error(self, error: Exception, *, holds_media: bool) -> Rejected | Unavailable:
        """Settle the failure as `settle_failure` does, but explain a refused `logprobs` keyword.

        A `TypeError` that refuses the `logprobs` keyword, raised by a call
        that asked for log-probabilities, is a `ConfigurationError` naming the
        count to use. Every other error passes through unchanged, including a
        `TypeError` that merely mentions `logprobs`.
        """
        if self.top_log_probabilities > 0 and is_rejected_keyword(error, keyword="logprobs"):
            message = (
                f"{name_chat_model(self.chat_model)} does not accept the request for "
                "log-probabilities that top_log_probabilities makes; use "
                "top_log_probabilities=0 with this model"
            )
            raise ConfigurationError(message) from error
        return settle_failure(error, holds_media=holds_media)
