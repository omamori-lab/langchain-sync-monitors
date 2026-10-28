"""The actions and observations a LangChain message adds to a hyoshigi trajectory."""

from __future__ import annotations

from hyoshigi import Action, Event
from langchain_core.messages import AIMessage, BaseMessage

from langchain_sync_monitors.contracts import TaskAuthor


def read_events(message: BaseMessage, *, task_author: TaskAuthor) -> list[Event]:
    """Read a message that enters the history unrated into the events it adds."""
    del message, task_author
    return []


def read_proposed_action(message: AIMessage) -> Action:
    """Read a reply the monitor rates into the action it rates."""
    del message
    return Action()
