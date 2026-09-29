"""How the library obtains chat models and tags the model calls it makes itself."""

from __future__ import annotations

import importlib.util

from langchain.agents.middleware.internal_call_transformer import internal_call_metadata
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from langchain_sync_monitors.errors import MissingExtraError

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
    """
    if isinstance(model, BaseChatModel):
        return model
    if model.startswith(OPENROUTER_PREFIX) and not is_package_installed("langchain_openrouter"):
        raise MissingExtraError(OPENROUTER_INSTALL_HINT)
    return init_chat_model(model)


def is_package_installed(name: str) -> bool:
    """Tell whether a top-level package can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


def build_internal_call_config(*, source: str) -> RunnableConfig:
    """Tag a model call the library makes itself, such as a monitor's call.

    The tag keeps the call out of the agent's message stream, the mechanism
    LangChain's ``InternalCallTransformer`` provides for middleware
    [@langchain2026]. ``source`` names the caller in traces.
    """
    return RunnableConfig(metadata={"lc_source": source, **internal_call_metadata()})
