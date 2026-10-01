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
    alternatives under `GuardScoring.LOG_PROBABILITIES`, or rejects the
    request for them under `GuardScoring.AUTO` or
    `GuardScoring.LOG_PROBABILITIES`.
    """


class MissingExtraError(ConfigurationError, ImportError):
    """A feature needs an optional extra that is not installed.

    It is both a `ConfigurationError` and an `ImportError`, so code that
    catches either one sees it. The message names the extra to install.
    """


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
