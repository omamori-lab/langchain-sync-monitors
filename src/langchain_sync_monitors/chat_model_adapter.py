"""A LangChain chat model as hyoshigi's synchronous and asynchronous generative models."""

from __future__ import annotations

from hyoshigi import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Refused,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class ChatModelAdapter:
    """One chat model as a `GenerativeModel` and an `AsyncGenerativeModel`."""

    def __init__(self, model: str | BaseChatModel, *, source: str) -> None:
        """Declare the attributes; the models they hold answer every prompt with a refusal."""

        def generate(prompt: Prompt) -> GenerationOutcome:
            return Refused(explanation="stub")

        async def generate_async(prompt: Prompt) -> GenerationOutcome:
            return Refused(explanation="stub")

        self.chat_model = FakeListChatModel(responses=[])
        self.generative_model = GenerativeModel(
            name="stub",
            generate=generate,
            perceives=MediaTypes("application/octet-stream"),
        )
        self.async_generative_model = AsyncGenerativeModel(
            name="stub",
            generate=generate_async,
            perceives=MediaTypes("application/octet-stream"),
        )
