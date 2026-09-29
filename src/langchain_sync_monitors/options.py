"""Checks on the options that public constructors take.

Options such as `Resampling` are `StrEnum`s, whose members compare equal to
their strings. The library tells options apart by identity, so a plain string
such as `"parallel"`, which configuration read from YAML or JSON produces,
would match no member and silently select another behaviour. Constructors
therefore reject any value that is not a member, and never convert one.

Constructors check the type of their other options too. A limit given as a
float, or a protocol given where a fallback belongs, would otherwise fail only
at the first suspicious step, the one the protocol exists for.
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


def describe_option_value(value: object) -> str:
    """Name a refused value: a plain value as written, anything else by its type."""
    if value is None or isinstance(value, str | int | float):
        return repr(value)
    return f"a {type(value).__name__}"


def check_instance_option(
    value: object,
    *,
    option_type: type,
    parameter_name: str,
    hint: str = "",
) -> None:
    """Raise `ConfigurationError` unless `value` is an instance of `option_type`.

    The message names the value and adds `hint`, which can name the likely
    mix-up.
    """
    if isinstance(value, option_type):
        return
    message = (
        f"{parameter_name} must be a {option_type.__name__}, "
        f"got {describe_option_value(value)}. {hint}"
    )
    raise ConfigurationError(message.rstrip())


def check_count_option(value: object, *, parameter_name: str, minimum: int) -> None:
    """Raise `ConfigurationError` unless `value` is an `int` of at least `minimum`.

    A `bool` is refused although Python counts it as an `int`, and so is a
    float such as `2.0`, which would fail only when the protocol first counts
    with it.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        message = (
            f"{parameter_name} must be an int of at least {minimum}, "
            f"got {describe_option_value(value)}"
        )
        raise ConfigurationError(message)
    if value < minimum:
        message = f"{parameter_name} must be at least {minimum}, got {value}"
        raise ConfigurationError(message)
