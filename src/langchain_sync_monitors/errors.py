"""Errors raised by the library.

Every error derives from `MonitorError`, so a caller can catch the library's
failures in one place without catching unrelated exceptions.
"""


class MonitorError(Exception):
    """Base class for every error this library raises."""


class ConfigurationError(MonitorError):
    """A monitor, protocol or middleware was built with settings that cannot work."""


class SynchronousRunError(MonitorError):
    """A control protocol awaited real asynchronous work during a synchronous `invoke()`."""
