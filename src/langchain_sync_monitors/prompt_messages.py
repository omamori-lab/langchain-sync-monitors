"""A hyoshigi prompt as the LangChain messages a chat model takes.

A prompt is a system prompt and parts of text and media [@hyoshigi2026]. The
messages are a system message and a user message of standard content blocks
[@langchaincore2026].
"""

from __future__ import annotations

from base64 import b64encode
from typing import TypedDict, assert_never

from hyoshigi import MediaPart, Prompt, PromptPart, TextPart
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import (
    AudioContentBlock,
    ContentBlock,
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    VideoContentBlock,
)


class MediaSource(TypedDict, total=False):
    """Where a content block's media is and what its type is."""

    url: str
    base64: str
    mime_type: str


def read_media_source(part: MediaPart) -> MediaSource | None:
    """Return the media's URL or its bytes in base64, with its type unless that is a range.

    The bytes of a range such as `image/*` have no type to name, so there is
    no source for them.
    """
    media_type = part.media_type.lower()
    is_range = media_type.endswith("/*")
    if part.url is not None:
        source = MediaSource(url=part.url)
    elif part.data is not None and not is_range:
        source = MediaSource(base64=b64encode(part.data).decode("ascii"))
    else:
        return None
    if not is_range:
        source["mime_type"] = media_type
    return source


def build_media_block(part: MediaPart) -> ContentBlock | None:
    """Return the standard content block for media: image, audio or video, else a file.

    The block is a literal and not one of LangChain's `create_*_block`
    functions, which give each a random id, so a repeated prompt would differ
    and miss the response cache [@langchaincore2026].
    """
    source = read_media_source(part)
    if source is None:
        return None
    match part.media_type.lower().partition("/")[0]:
        case "image":
            return ImageContentBlock(type="image", **source)
        case "audio":
            return AudioContentBlock(type="audio", **source)
        case "video":
            return VideoContentBlock(type="video", **source)
        case _:
            return FileContentBlock(type="file", **source)


def build_content_block(part: PromptPart) -> ContentBlock | None:
    """Return the standard content block for a prompt part, or `None` if it has no form."""
    match part:
        case TextPart(text=text):
            return TextContentBlock(type="text", text=text)
        case MediaPart():
            return build_media_block(part)
        case _:
            assert_never(part)


def is_blank_text(part: PromptPart) -> bool:
    """Tell whether a part is text with no character but whitespace."""
    return isinstance(part, TextPart) and not part.text.strip()


def build_content_blocks(parts: tuple[PromptPart, ...]) -> list[ContentBlock] | None:
    """Return the blocks of the parts that are not blank text, or `None` if none can be sent.

    Some providers reject blank text, so it is left out. If a part has no
    form, or nothing is left, there is nothing to send.
    """
    blocks: list[ContentBlock] = []
    for part in parts:
        if is_blank_text(part):
            continue
        block = build_content_block(part)
        if block is None:
            return None
        blocks.append(block)
    return blocks or None


def build_messages(prompt: Prompt) -> list[BaseMessage] | None:
    """Return the prompt as a system message and a user message of content blocks.

    A blank system prompt is not sent. The result is `None` when there is
    nothing to send.
    """
    blocks = build_content_blocks(prompt.content)
    if blocks is None:
        return None
    user_message = HumanMessage(content_blocks=blocks)
    if not prompt.system.strip():
        return [user_message]
    return [SystemMessage(prompt.system), user_message]


def has_media_part(prompt: Prompt) -> bool:
    """Tell whether the prompt holds a media part."""
    return any(isinstance(part, MediaPart) for part in prompt.content)
