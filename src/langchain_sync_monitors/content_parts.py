"""The hyoshigi part each LangChain content block is: text, media, or opaque content.

Text is a text part, and an image, audio, video, file or plain-text document
is a media part by its bytes or its URL [@langchaincore2026; @hyoshigi2026].
Media with neither, such as a file stored with the provider, and a block of
any other type are opaque parts named by their type.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TypeGuard

from hyoshigi import ConfigurationError as HyoshigiConfigurationError
from hyoshigi import MediaPart, OpaquePart, Part, TextPart
from langchain_core.messages import (
    ContentBlock,
    DataContentBlock,
    NonStandardContentBlock,
    TextContentBlock,
)

from langchain_sync_monitors.errors import MessageFormatError
from langchain_sync_monitors.transcript import (
    read_unrecognised_block_name,
    read_unrecognised_block_value,
)

MEDIA_RANGES = {
    "image": "image/*",
    "audio": "audio/*",
    "video": "video/*",
    "file": "application/octet-stream",
    "text-plain": "text/plain",
}
"""The media type of each media block type, for a block that names none.

A file of unknown type is arbitrary binary data, `application/octet-stream`
(section 4.5.1) [@rfc2046].
"""


def read_content_part(block: ContentBlock, *, path: str) -> Part:
    """Read text or media into a part; a block of another type is opaque, named by its type."""
    if block["type"] == "text":
        return read_text(block, path=path)
    if is_media(block):
        return read_media(block, path=path)
    if block["type"] == "non_standard":
        return OpaquePart(kind=read_unmapped_kind(block))
    return OpaquePart(kind=block["type"])


def read_text(block: TextContentBlock, *, path: str) -> TextPart:
    """Read a text block, whose text must be a string."""
    text: object = block.get("text")
    if not isinstance(text, str):
        raise MessageFormatError(path=path, expected="a text block with its text")
    return TextPart(text=text)


def is_media(block: ContentBlock) -> TypeGuard[DataContentBlock]:
    """Tell whether a block is an image, audio, video, file or plain-text document."""
    return block["type"] in MEDIA_RANGES


def read_media(block: DataContentBlock, *, path: str) -> Part:
    """Read media by its bytes or its URL; media with neither, such as a file id, is opaque.

    A plain-text document that holds its text is text.
    """
    values: Mapping[str, object] = block
    match values:
        case {"type": "text-plain", "text": str(text)}:
            return TextPart(text=text)
        case {"base64": str(data)}:
            return build_media(block, data=decode_base64(data, path=path), path=path)
        case {"url": str(url)}:
            return build_media(block, url=url, path=path)
        case _:
            return OpaquePart(kind=block["type"])


def build_media(
    block: DataContentBlock,
    *,
    path: str,
    data: bytes | None = None,
    url: str | None = None,
) -> MediaPart:
    """Build a media part of the block's media type, or of its type's range when it names none."""
    media_type = block.get("mime_type") or MEDIA_RANGES[block["type"]]
    try:
        return MediaPart(media_type=media_type, data=data, url=url)
    except HyoshigiConfigurationError:
        raise MessageFormatError(path=path, expected="an IANA media type") from None


def decode_base64(data: str, *, path: str) -> bytes:
    """Decode base64 data, raising `MessageFormatError` without quoting it when it is not base64."""
    try:
        return base64.b64decode(data, validate=True)
    except ValueError:
        raise MessageFormatError(path=path, expected="base64 data") from None


def read_unmapped_kind(block: NonStandardContentBlock) -> str:
    """Return the type a block LangChain could not map gives itself, or `non_standard`."""
    return read_unrecognised_block_name(read_unrecognised_block_value(block)) or "non_standard"
