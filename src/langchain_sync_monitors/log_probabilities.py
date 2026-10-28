"""The log-probabilities of a chat completions reply, and the request for them.

A chat model asked with `logprobs` and `top_logprobs` returns each generated
token with its log-probability and its likeliest alternatives
[@openaipythonsdk2026]. langchain-openai and langchain-openrouter put them in
the choice's `generation_info`, and langchain-core merges that into the
reply's `response_metadata["logprobs"]`
[@langchainopenai2026; @langchainopenrouter2026; @langchaincore2026]. How
hyoshigi's guard scores from them, and when it fails closed, is hyoshigi's to
say [@hyoshigi2026choosing].
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import TypedDict

from hyoshigi import (
    AlternativeToken,
    Generation,
    TokenLogProbability,
    Unavailable,
)
from hyoshigi import ConfigurationError as HyoshigiConfigurationError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError

logger = logging.getLogger(__name__)

type ProviderPreferences = dict[str, object]
"""The provider preferences of an OpenRouter request, by name [@openrouter2026providerrouting]."""


class RequestOptions(TypedDict, total=False):
    """The keywords that ask a chat model for log-probabilities."""

    logprobs: bool
    top_logprobs: int
    provider: ProviderPreferences


class AlternativePayload(BaseModel):
    """One of the likeliest tokens at a position: a token and its log-probability.

    This is OpenAI's `TopLogprob` [@openaipythonsdk2026]. Keys other than the
    two are ignored, and nothing is coerced, so a `bool` or a string is not a
    log-probability [@pydantic2026].
    """

    token: StrictStr
    logprob: StrictFloat


class PositionPayload(BaseModel):
    """A generated token, its log-probability and the alternatives to it.

    This is OpenAI's `ChatCompletionTokenLogprob` [@openaipythonsdk2026]. The
    alternatives are required and may be `None`.
    """

    token: StrictStr
    logprob: StrictFloat
    top_logprobs: list[AlternativePayload] | None


class LogProbabilitiesPayload(BaseModel):
    """The `logprobs` of a chat completions choice: one position per generated token.

    `content` is required and may be `None`, as it is when the model refused.
    """

    content: list[PositionPayload] | None


def build_position(position: PositionPayload) -> TokenLogProbability:
    """Return the position as hyoshigi's `TokenLogProbability`, with its alternatives."""
    alternatives = tuple(
        AlternativeToken(token=alternative.token, log_probability=alternative.logprob)
        for alternative in position.top_logprobs or []
    )
    return TokenLogProbability(
        token=position.token,
        log_probability=position.logprob,
        alternatives=alternatives,
    )


def read_log_probabilities(payload: object) -> tuple[TokenLogProbability, ...] | None:
    """Return the positions of a choice's `logprobs`, or `None` if they are unreadable.

    A payload of `None`, or with no `content`, holds no positions and reads as
    `()`; nothing is invented for it. A value that hyoshigi refuses as a
    log-probability, such as a positive one or NaN, makes the payload
    unreadable too. The warning names the payload's type alone, since its
    tokens are the model's reply.
    """
    if payload is None:
        return ()
    try:
        positions = LogProbabilitiesPayload.model_validate(payload).content or []
        return tuple(build_position(position) for position in positions)
    except (ValidationError, HyoshigiConfigurationError):
        logger.warning(
            "The log-probabilities of a chat model's reply (a %s) are not in the chat "
            "completions format; the reply is reported as unavailable.",
            type(payload).__name__,
        )
        return None


def add_log_probabilities(generation: Generation, *, reply: AIMessage) -> Generation | Unavailable:
    """Return the generation with the log-probabilities of the reply it was read from.

    A reply with none gives a generation with none. A reply whose
    log-probabilities are not in the documented shape is unavailable.
    """
    positions = read_log_probabilities(reply.response_metadata.get("logprobs"))
    if positions is None:
        return Unavailable()
    return Generation(text=generation.text, log_probabilities=positions)


def is_chat_openrouter(model: BaseChatModel) -> bool:
    """Tell whether the chat model is a `ChatOpenRouter`.

    An instance can only exist once langchain-openrouter has loaded, so the
    class is taken from the modules already loaded and nothing is imported:
    the `openrouter` extra stays optional [@langchainopenrouter2026].
    """
    chat_class = getattr(sys.modules.get("langchain_openrouter"), "ChatOpenRouter", None)
    return chat_class is not None and isinstance(model, chat_class)


def read_provider_preferences(model: BaseChatModel) -> ProviderPreferences:
    """Return the provider preferences a `ChatOpenRouter` sends, or `{}` if it has none.

    That is its `openrouter_provider` when set, else the `provider` among its
    `model_kwargs`, where `ChatOpenRouter(provider=...)` also lands
    [@langchainopenrouter2026].
    """
    preferences = getattr(model, "openrouter_provider", None)
    if preferences is None:
        preferences = getattr(model, "model_kwargs", {}).get("provider")
    return dict(preferences) if isinstance(preferences, Mapping) else {}


def build_request_options(model: BaseChatModel, *, count: int) -> RequestOptions:
    """Return the keywords that ask the chat model for `count` alternatives to each token.

    A count of 0 asks for none. For a `ChatOpenRouter` the options also hold
    the provider preferences with `require_parameters` true, which routes the
    request only to providers that support every parameter in it
    [@openrouter2026providerrouting]; a `require_parameters` the user set
    stays as they set it, and so do their other preferences.
    """
    if count == 0:
        return {}
    options = RequestOptions(logprobs=True, top_logprobs=count)
    if is_chat_openrouter(model):
        options["provider"] = {"require_parameters": True, **read_provider_preferences(model)}
    return options


def is_rejected_keyword(error: Exception, *, keyword: str) -> bool:
    """Tell whether Python refused a call for an unexpected keyword argument named `keyword`.

    Some chat models, such as `ChatAnthropic` and `ChatBedrockConverse`, pass
    a bound keyword on to a function that does not take it, and Python raises
    a `TypeError` before any request leaves [@langchainanthropic2026;
    @langchainaws2026].
    """
    text = str(error).lower()
    is_rejection = "unexpected keyword argument" in text and keyword in text
    return isinstance(error, TypeError) and is_rejection
