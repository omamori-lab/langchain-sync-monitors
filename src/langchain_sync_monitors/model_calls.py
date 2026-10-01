"""How the library obtains chat models and tags the model calls it makes itself."""

from __future__ import annotations

import importlib.util

from langchain.agents.middleware.internal_call_transformer import internal_call_metadata
from langchain.chat_models import init_chat_model
from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from langchain_sync_monitors.errors import ConfigurationError, MissingExtraError

OPENROUTER_PREFIX = "openrouter:"
OPENROUTER_INSTALL_HINT = (
    "An 'openrouter:' model string needs the openrouter extra: "
    "pip install 'langchain-sync-monitors[openrouter]'"
)


def resolve_chat_model(model: str | BaseChatModel) -> BaseChatModel:
    """Return the model itself, or initialise one from a provider string.

    A string such as ``"openrouter:xiaomi/mimo-v2.6-pro"`` goes through
    LangChain's ``init_chat_model``, the same way LangChain's own middleware
    accepts a second model [@langchain2026]. The library never picks a model.

    An ``openrouter:`` string needs the ``openrouter`` extra, which installs
    langchain-openrouter [@langchainopenrouter2026]; that package calls
    OpenRouter through its Python SDK [@openrouterpythonsdk2026]. Without the
    extra, such a string raises `MissingExtraError` naming it. The check only
    looks for the package, without importing it; a package that is present
    but broken, and every other provider, raise LangChain's own
    ``ImportError``.

    Anything else, such as a chat model wrapped in a Runnable by
    ``with_retry()`` or ``bind()``, raises `ConfigurationError` naming its
    type. The message holds the type alone, since a Runnable's repr carries
    its bound arguments.
    """
    if isinstance(model, BaseChatModel):
        return model
    if not isinstance(model, str):
        message = (
            "model must be a LangChain chat model or a provider string such as "
            f"'openrouter:xiaomi/mimo-v2.6-pro', got {type(model).__name__}. Pass the chat "
            "model itself, not a Runnable wrapped around it, and set its retries through its "
            "own max_retries."
        )
        raise ConfigurationError(message)
    if model.startswith(OPENROUTER_PREFIX) and not is_package_installed("langchain_openrouter"):
        raise MissingExtraError(OPENROUTER_INSTALL_HINT)
    return init_chat_model(model)


class CachedResampleWarning(UserWarning):
    """A model is asked the same request more than once while a response cache is active.

    LangChain answers an identical request from its cache [@langchaincore2026],
    so every repeat is a copy of the first reply. A protocol's resample can
    then never find a safer step: resampling helps less the more
    deterministic the model is [@bhatt2025ctrlzpost], and a cached model is
    fully deterministic. A monitor's samples all carry the first one's label,
    and its request after an unreadable reply gets the same reply back.
    """


def is_response_cache_active(model: BaseChatModel) -> bool:
    """Tell whether LangChain answers this model's calls from a response cache.

    A model's own `cache` wins: an instance is used, `False` turns caching
    off, and `True` or `None` use the global cache when one is set. LangChain
    types an agent's model as a chat model, but it may be a Runnable around
    one. A wrapper such as `bind()` or `with_fallbacks()` passes the read on
    to the model inside. One from `init_chat_model` with configurable fields
    builds its default model to answer, and raises what building raises,
    `AttributeError` or `TypeError` without a default model [@langchain2026].
    A model whose `cache` cannot be read counts as uncached: nothing is known
    of it, and a warning must not fail the step.
    """
    try:
        cache = model.cache
    except Exception:
        return False
    if isinstance(cache, BaseCache):
        return True
    return cache is not False and get_llm_cache() is not None


def is_package_installed(name: str) -> bool:
    """Tell whether a top-level package can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


MESSAGE_VIEW_EXCLUDE_KEY = "ls_message_view_exclude"
"""The metadata key that keeps one run out of LangSmith's Trajectory view.

LangSmith checks it by presence, and documents it for classification calls,
safety filters and routing or guardrail decisions [@langsmith2026trajectory].
"""

MONITOR_CALL_NAME = "monitor call"
"""The run name of every model call a monitor makes through LangChain, whatever its model.

A tracer otherwise names a call after its chat model's class, the same name
as the agent's own calls. With this fixed name, LangSmith and Langfuse filter
the monitor's calls by name, beside the spans in `spans`
[@langsmith2026traces; @langfuse2026]. `OpenRouterDecisionModel` sends its
request without LangChain, so it shows as a `monitor classifier` span instead.
"""


def build_internal_call_config(*, source: str) -> RunnableConfig:
    """Tag a model call the library makes itself, such as a monitor's call.

    The run is named `MONITOR_CALL_NAME`, which replaces any name the model
    was given, so a model built with ``name="security monitor"`` shows as
    ``monitor call`` too. The model still shows as the call's model and in
    the ``ls_model_name`` metadata, which the chat model adds itself, and the
    judgement span around the call names the monitor. The name goes to the
    outermost run the config reaches, so a Runnable that wraps the model,
    such as one from ``with_retry()``, takes the name, and the calls it makes
    keep their own; they still carry this metadata. ``source`` goes into the
    ``lc_source`` metadata.

    The metadata drops the call from the experimental
    ``stream_events(version="v3")`` projection, through LangChain's
    ``InternalCallTransformer`` [@langchain2026], and keeps it out of
    LangSmith's Trajectory view [@langsmith2026trajectory]. It does not
    filter ``stream_mode="messages"``: the middleware's ``nostream`` block
    keeps the call out of that stream.
    """
    return RunnableConfig(
        run_name=MONITOR_CALL_NAME,
        metadata={
            "lc_source": source,
            MESSAGE_VIEW_EXCLUDE_KEY: True,
            **internal_call_metadata(),
        },
    )
