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
from collections.abc import Set as AbstractSet
from decimal import Decimal
from enum import Enum
from typing import Final, TypeGuard

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
    type_name = option_type.__name__
    article = "an" if type_name[0] in "AEIOU" else "a"
    message = (
        f"{parameter_name} must be {article} {type_name}, "
        f"got {describe_option_value(value)}. {hint}"
    )
    raise ConfigurationError(message.rstrip())


def check_optional_instance_option(
    value: object,
    *,
    option_type: type,
    parameter_name: str,
    hint: str = "",
) -> None:
    """Raise `ConfigurationError` unless `value` is `None` or an instance of `option_type`."""
    if value is not None:
        check_instance_option(
            value,
            option_type=option_type,
            parameter_name=parameter_name,
            hint=hint,
        )


def check_string_set_option(value: object, *, parameter_name: str, example: str) -> None:
    """Raise `ConfigurationError` unless `value` is a set that holds only strings.

    A plain string is refused, although it is a collection of strings: its
    membership test would match any part of it. `example` shows the shape.
    """
    if not isinstance(value, AbstractSet):
        message = (
            f"{parameter_name} must be a set of strings, such as {example}, "
            f"got {describe_option_value(value)}"
        )
        raise ConfigurationError(message)
    for item in value:
        if not isinstance(item, str):
            message = f"{parameter_name} must hold only strings, got {describe_option_value(item)}"
            raise ConfigurationError(message)


RESERVED_NODE_NAME_CHARACTERS: Final = (":", "|")
"""The characters LangGraph refuses in a graph node's name [@langgraph2026]."""


def check_name_part_option(value: object, *, parameter_name: str) -> None:
    """Raise `ConfigurationError` unless `value` is a non-blank string that fits in a node name.

    A middleware's hooks become graph nodes named after it, such as
    `monitor[main].before_model`, so a part of the name that holds a
    character LangGraph refuses there would fail only when the agent is built.
    """
    if not isinstance(value, str) or not value.strip():
        message = f"{parameter_name} must be a non-blank string, got {describe_option_value(value)}"
        raise ConfigurationError(message)
    reserved = [character for character in RESERVED_NODE_NAME_CHARACTERS if character in value]
    if reserved:
        characters = " or ".join(map(repr, reserved))
        message = (
            f"{parameter_name} must not contain {characters}, which LangGraph refuses in the "
            f"names of the graph nodes the monitor's hooks become, got {value!r}"
        )
        raise ConfigurationError(message)


def is_whole_number(value: object) -> TypeGuard[numbers.Integral]:
    """Tell whether a value is an integral number other than a `bool`.

    numpy's integers count, since the protocols count with them through
    `__index__`. A `bool` does not, although Python counts it as an `int`, and
    nor does a float such as `2.0`, which would fail only when the protocol
    first counts with it.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def read_integer_option(value: object, *, parameter_name: str) -> int:
    """Return an integer option, such as the end of a score scale, as an `int`.

    Any whole number counts, negative ones and numpy's integers included; a
    `bool`, a float such as `2.0` or a string raises `ConfigurationError`.
    """
    if not is_whole_number(value):
        message = f"{parameter_name} must be an integer, got {describe_option_value(value)}"
        raise ConfigurationError(message)
    return operator.index(value)


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


def read_optional_count_option(
    value: object,
    *,
    parameter_name: str,
    minimum: int,
    none_means: str,
) -> int | None:
    """Return a count option that may be `None`: a whole number of at least `minimum`, or `None`.

    `none_means` says what `None` does, for the message. Anything else raises
    `ConfigurationError`.
    """
    if value is None:
        return None
    if not is_whole_number(value):
        message = (
            f"{parameter_name} must be a whole number of at least {minimum}, or None "
            f"{none_means}, got {describe_option_value(value)}"
        )
        raise ConfigurationError(message)
    count = operator.index(value)
    if count < minimum:
        message = f"{parameter_name} must be at least {minimum}, or None {none_means}, got {count}"
        raise ConfigurationError(message)
    return count


def read_positive_number_option(value: object, *, parameter_name: str) -> float:
    """Return a positive, finite number, such as a timeout in seconds, as a float.

    Any real number counts, and so does a `Decimal`, as for a threshold. A
    `bool`, a string, zero, a negative number, NaN, infinity and a number that
    a float would round to zero or infinity raise `ConfigurationError`.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real | Decimal):
        message = f"{parameter_name} must be a positive number, got {describe_option_value(value)}"
        raise ConfigurationError(message)
    try:
        number = float(value)
    except (ArithmeticError, ValueError):
        # A huge int raises `OverflowError`, and a signalling Decimal NaN `ValueError`.
        number = math.nan
    if not 0.0 < number < math.inf:
        message = f"{parameter_name} must be a positive, finite number, got {value!r}"
        raise ConfigurationError(message)
    return number
