"""A redaction that masks one planted token, and the agents whose messages all hold it."""

from __future__ import annotations

from hyoshigi import MediaPart, Part, ToolCall
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

from langchain_sync_monitors import Redaction
from tests.support.chat_models import build_tool_call_message

SECRET = "SECRET-123"
MASK = "[REDACTED]"
TASK_WITH_SECRET = f"Summarise q3.md, key {SECRET}."
READ_RESULT_WITH_SECRET = f"contents of q3.md: {SECRET}"
FIRST_REPLY_WITH_SECRET = f"Reading with {SECRET}."
LAST_REPLY_WITH_SECRET = f"The key is {SECRET}."
MASKED_TASK = f"Summarise q3.md, key {MASK}."
MASKED_READ_RESULT = f"contents of q3.md: {MASK}"
MASKED_FIRST_REPLY = f"Reading with {MASK}."
MASKED_LAST_REPLY = f"The key is {MASK}."
READ_CALL_WITH_SECRET = ToolCall(
    id="call-read",
    name="read_file",
    arguments=f'{{"path": "q3.md", "note": "{SECRET}"}}',
)
MASKED_READ_CALL = ToolCall(
    id="call-read",
    name="read_file",
    arguments=f'{{"path": "q3.md", "note": "{MASK}"}}',
)
SUSPICION_BY_SECRET = {SECRET: 0.9}
"""A keyword monitor's scores for a reply that holds the token, whose reason names it."""


def mask_secret(text: str) -> str:
    """Return the text with the planted token masked."""
    return text.replace(SECRET, MASK)


def convert_to_capitals(text: str) -> str:
    """Return the text in capitals, which changes the sources "system" and "developer" too."""
    return text.upper()


def mask_secret_or_return_none(text: str) -> str:
    """Mask the token, but return no str for a monitor's reason or an error the middleware reports.

    The agent's own texts never begin that way, so hyoshigi never meets the None.
    """
    if text.startswith(("mentions", "RuntimeError")):
        return None  # ty: ignore[invalid-return-type]
    return mask_secret(text)


def keep_media(part: MediaPart) -> Part:
    """Return the media as it is."""
    return part


def append_exclamation_mark(text: str) -> str:
    """Return the text with one exclamation mark more, which each pass through it shows."""
    return f"{text}!"


MASK_SECRET = Redaction(text=mask_secret, media=keep_media)
"""Masks the planted token in every text; running it again changes nothing."""

RETURN_NONE_FOR_REASONS = Redaction(text=mask_secret_or_return_none, media=keep_media)
"""Masks the token, but its text function returns None for a reason or an error it is given."""

UPPER_CASE = Redaction(text=convert_to_capitals, media=keep_media)
"""Puts every text in capitals, a source included; running it again changes nothing."""

COUNT_PASSES = Redaction(text=append_exclamation_mark, media=keep_media)
"""Adds a mark to every text, so a text shows how many times it passed through."""


def build_secret_tools() -> list[BaseTool]:
    """Return a `read_file` tool whose result holds the planted token."""

    @tool
    def read_file(path: str, note: str = "") -> str:
        """Read a file from the workspace, with a note of why."""
        return f"contents of {path}: {SECRET}"

    return [read_file]


def build_secret_steps() -> list[AIMessage]:
    """Script two replies that hold the token: a read of q3.md, then the final answer."""
    return [
        build_tool_call_message(
            tool_name="read_file",
            call_id="call-read",
            arguments={"path": "q3.md", "note": SECRET},
            content=FIRST_REPLY_WITH_SECRET,
        ).model_copy(update={"id": "step-1"}),
        AIMessage(LAST_REPLY_WITH_SECRET, id="step-2"),
    ]
