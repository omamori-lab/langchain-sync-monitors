"""The key a trajectory reads a message under."""

from __future__ import annotations

import hashlib
import json

from langchain_core.messages import BaseMessage


def build_content_key(message: BaseMessage) -> str:
    """Return a hash of a message's type and content."""
    text = json.dumps([type(message).__name__, message.content], sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def build_message_key(message: BaseMessage) -> str:
    """Return a message's key: its id, or its content key when it has none."""
    return message.id or build_content_key(message)
