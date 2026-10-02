"""Errors raised by the library, and the warning about what a monitor cannot gate.

Every error derives from `MonitorError`, so a caller can catch the library's
failures in one place without catching unrelated exceptions.
"""


class MonitorError(Exception):
    """Base class for every error this library raises."""


class ConfigurationError(MonitorError):
    """A monitor, protocol or middleware was built with settings that cannot work.

    Most are raised when the object is built. A few settings can only be
    checked during a run, so the same error is raised then: a
    `monitor_delegation` in an agent's input that is not a valid
    `Delegation`, and a guard model that returns no log-probabilities with
    alternatives under `GuardScoring.LOG_PROBABILITIES`, or whose adapter
    does not take the `logprobs` keyword under `GuardScoring.AUTO` or
    `GuardScoring.LOG_PROBABILITIES`.
    """


class MissingExtraError(ConfigurationError, ImportError):
    """A feature needs an optional extra that is not installed.

    It is both a `ConfigurationError` and an `ImportError`, so code that
    catches either one sees it. The message names the extra, and the uv and
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


class InvalidSuspicionError(MonitorError, ValueError):
    """A verdict's suspicion is NaN or lies outside [0, 1].

    It is both a `MonitorError` and a `ValueError`, so code that catches
    either one sees it.
    """


class SynchronousRunError(MonitorError):
    """A control protocol or a monitor needed an event loop during a synchronous `invoke()`.

    The message names which one: a protocol that awaited real asynchronous
    work or started asyncio work, or a monitor whose `evaluate_sync` started
    asyncio work. It is also raised when a pending step is used after its
    `invoke()` step ended.
    """


class RateLimitedCallError(MonitorError):
    """A chat monitor's model call that the provider answered with HTTP 429.

    A chat monitor retries a rate-limited call on this error in place of the
    provider's own. stamina hands the error it retries on to every retry
    hook, and its logging hook logs the error's repr [@schlawack2026stamina].
    A provider's error can hold its whole reply: OpenRouter's
    `TooManyRequestsResponseError` is a dataclass whose repr holds the reply's
    body, the account's `user_id` included, and its headers
    [@openrouterpythonsdk2026]. This error names the status and the type of the
    provider's error, nothing else, and has nothing chained to it. It never
    leaves the monitor: after the last attempt, the provider's own error is
    raised.
    """

    def __init__(self, *, error_type: str) -> None:
        """Name the type of the provider's error, and nothing it holds."""
        super().__init__(f"the provider answered HTTP 429 with {error_type}")
        self.error_type = error_type

    def __repr__(self) -> str:
        """Name the status and the type of the provider's error, as stamina's retry log shows it."""
        return f"RateLimitedCallError(status_code=429, error_type={self.error_type!r})"


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
