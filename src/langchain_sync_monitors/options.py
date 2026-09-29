"""Checks on the enum options that public constructors take.

Options such as `Resampling` are `StrEnum`s, whose members compare equal to
their strings. The library tells options apart by identity, so a plain string
such as `"parallel"`, which configuration read from YAML or JSON produces,
would match no member and silently select another behaviour. Constructors
therefore reject any value that is not a member, and never convert one.
"""

from __future__ import annotations

from enum import Enum

from langchain_sync_monitors.errors import ConfigurationError


def check_enum_option(value: object, *, option_type: type[Enum], parameter_name: str) -> None:
    """Raise `ConfigurationError` unless `value` is a member of `option_type`.

    The message names every accepted member and how to convert a string.
    """
    if isinstance(value, option_type):
        return
    type_name = option_type.__name__
    accepted = ", ".join(f"{type_name}.{member.name}" for member in option_type)
    message = (
        f"{parameter_name} must be one of {accepted}, got {value!r}. "
        f"Convert a string with {type_name}(value)."
    )
    raise ConfigurationError(message)
