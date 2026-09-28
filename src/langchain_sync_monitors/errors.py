"""Errors raised by the library.

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


class SynchronousRunError(MonitorError):
    """A control protocol awaited real asynchronous work during a synchronous `invoke()`."""
