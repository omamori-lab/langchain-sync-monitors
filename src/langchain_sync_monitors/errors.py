"""Errors raised by the library, and the warning about what a monitor cannot gate.

Every error derives from `MonitorError`, so a caller can catch the library's
failures in one place without catching unrelated exceptions.
"""


class MonitorError(Exception):
    """Base class for every error this library raises."""


class ConfigurationError(MonitorError):
    """A monitor, protocol or middleware was built with settings that cannot work."""


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
    """A control protocol awaited real asynchronous work during a synchronous `invoke()`."""


class ProviderToolWarning(UserWarning):
    """The agent's model is given tools that the model provider runs itself.

    LangChain passes every dictionary in an agent's tools to the provider as a
    built-in tool and never runs it itself [@langchain2026]. A provider runs
    its server-side tools, such as Anthropic's `web_fetch` or OpenAI's
    `web_search`, inside the model call, before the monitor judges the step,
    and again for every sample a protocol draws, so no monitor can stop them.
    The monitor middleware emits this warning once per middleware instance.
    """
