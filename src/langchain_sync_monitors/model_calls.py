"""How the library obtains chat models and tags the model calls it makes itself."""

from __future__ import annotations

from langchain.agents.middleware.internal_call_transformer import internal_call_metadata
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


def resolve_chat_model(model: str | BaseChatModel) -> BaseChatModel:
    """Return the model itself, or initialise one from a provider string.

    A string such as ``"openrouter:xiaomi/mimo-v2.6-pro"`` goes through
    LangChain's ``init_chat_model``, the same way LangChain's own middleware
    accepts a second model [@langchain2026]. The library never picks a model.
    """
    if isinstance(model, BaseChatModel):
        return model
    return init_chat_model(model)


def build_internal_call_config(*, source: str) -> RunnableConfig:
    """Tag a model call the library makes itself, such as a monitor's call.

    The tag keeps the call out of the agent's message stream, the mechanism
    LangChain's ``InternalCallTransformer`` provides for middleware
    [@langchain2026]. ``source`` names the caller in traces.
    """
    return RunnableConfig(metadata={"lc_source": source, **internal_call_metadata()})
