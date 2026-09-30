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

import math
import numbers
import operator
from enum import Enum
from typing import TypeGuard

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
    """Name a refused value: a string, a number or None as written, anything else by its type.

    A type from outside Python's builtins and this library is named with its
    module, so numpy's `bool` does not read as Python's.
    """
    if value is None or isinstance(value, str | numbers.Number):
        return repr(value)
    value_type = type(value)
    module = value_type.__module__
    is_own = module == "builtins" or module.split(".")[0] == __name__.split(".")[0]
    type_name = value_type.__qualname__ if is_own else f"{module}.{value_type.__qualname__}"
    return f"an instance of {type_name}"


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


def is_whole_number(value: object) -> TypeGuard[numbers.Integral]:
    """Tell whether a value is an integral number other than a `bool`.

    numpy's integers count, since the protocols count with them through
    `__index__`. A `bool` does not, although Python counts it as an `int`, and
    nor does a float such as `2.0`, which would fail only when the protocol
    first counts with it.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def read_count_option(value: object, *, parameter_name: str, minimum: int) -> int:
    """Return a count option as an `int`, raising `ConfigurationError` unless it is one.

    The value must be a whole number of at least `minimum`.
    """
    if not is_whole_number(value):
        message = (
            f"{parameter_name} must be a whole number of at least {minimum}, "
            f"got {describe_option_value(value)}"
        )
        raise ConfigurationError(message)
    count = operator.index(value)
    if count < minimum:
        message = f"{parameter_name} must be at least {minimum}, got {count}"
        raise ConfigurationError(message)
    return count


def read_limit_option(value: object, *, parameter_name: str, minimum: int) -> int | float:
    """Return a limit option: a whole number of at least `minimum`, or `math.inf` for no limit.

    Anything else raises `ConfigurationError`.
    """
    if isinstance(value, float) and value == math.inf:
        return value
    if is_whole_number(value) and operator.index(value) >= minimum:
        return operator.index(value)
    message = (
        f"{parameter_name} must be a whole number of at least {minimum}, or math.inf for "
        f"no limit, got {describe_option_value(value)}"
    )
    raise ConfigurationError(message)
