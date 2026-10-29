"""Errors raised by the library, and the warning about what a monitor cannot gate.

`ConfigurationError` is hyoshigi's, raised as it is. Every other error derives
from `MonitorError`, so a caller can catch the library's failures in one place
without catching unrelated exceptions.
"""

from hyoshigi import ConfigurationError


class MonitorError(Exception):
    """Base class for the errors this library raises, `ConfigurationError` excepted."""


class MissingExtraError(ConfigurationError, ImportError):
    """A feature needs an optional extra that is not installed.

    It is both hyoshigi's `ConfigurationError` and an `ImportError`, so code
    that catches either one sees it. The message names the extra, and the uv and
    pip commands that install it.
    """


def build_missing_extra_message(feature: str, *, extra: str) -> str:
    """Say that `feature` needs `extra`, and how to install it with uv or with pip.

    Every `MissingExtraError` the library raises takes its message from here,
    so the install commands are written once. The requirement is in double
    quotes, which POSIX shells, PowerShell and Windows `cmd` all accept.
    """
    requirement = f'"langchain-sync-monitors[{extra}]"'
    return (
        f"{feature} needs the {extra} extra. "
        f"Install it with: uv add {requirement} (or pip install {requirement})"
    )


class MessageFormatError(MonitorError, ValueError):
    """A LangChain message that the trajectory reader cannot read.

    It is raised for an AI message without an id, a message of a type the
    reader does not read, a block of a known type that lacks what its type
    requires, and an action that reuses an action id or a tool call id
    already in the monitor's trajectory. The message names where the
    problem is, such as `AIMessage.content_blocks[2]`, and what was expected
    there, and quotes nothing the message holds. It is both a `MonitorError` and a
    `ValueError`, so code that catches either one sees it.
    """

    def __init__(self, *, path: str, expected: str) -> None:
        """Name where in the message the problem is, and what was expected there."""
        super().__init__(f"{path}: expected {expected}")
        self.path = path


class RetriedCallError(MonitorError):
    """The stand-in for an error the library retries, naming its type and HTTP status alone.

    The library retries with stamina, which hands the error it retries on to
    every retry hook, and whose logging hook logs the error's repr
    [@schlawack2026stamina]. The error a call failed with can hold what it
    sent or received, so each retry site retries on this error in its place,
    as `retries` explains. Nothing is chained to it. It never leaves the
    library: after the last attempt, the error it stood in for is raised.
    """

    def __init__(self, *, error_type: str, http_status: int | None) -> None:
        """Name the type of the error stood in for and its HTTP status, and nothing it holds."""
        status = "no HTTP status" if http_status is None else f"HTTP {http_status}"
        super().__init__(f"a call failed with {error_type} ({status}) and is retried")
        self.error_type = error_type
        self.http_status = http_status

    def __repr__(self) -> str:
        """Name the error's type and HTTP status, as stamina's retry log shows them."""
        return f"RetriedCallError(error_type={self.error_type!r}, http_status={self.http_status!r})"


class ServerToolWarning(UserWarning):
    """The agent's model is given server tools, which the model provider runs itself.

    A provider runs its server-side tools, such as Anthropic's `web_fetch` or
    OpenAI's `web_search`, inside the model call, before the monitor judges
    the step, and again for every sample a protocol draws, so no monitor can
    stop them. The tools of an MCP server the application connects itself
    are not server tools: they run on the client, as the agent's own tools,
    and the monitor judges their calls before they run. The monitor
    middleware emits this warning once per middleware instance. It knows the
    server tools of Anthropic, OpenAI and Gemini that `server_tools` lists,
    the providers' MCP connectors among them, and reads only the tools of a
    model request: a server-side feature set on the model itself, such as
    OpenRouter's web plugin, runs without a warning.
    """
