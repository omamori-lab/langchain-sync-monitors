"""Stand-ins for numpy's scalars, which a monitor or a protocol may score and compare with.

numpy is not a dependency, so these small classes play its scalars:
`ArrayFloat64` is a `float` whose comparisons give numpy's `bool`, which is not
Python's, and `ArrayFloat32` is a real number that is not a `float`.
"""

from __future__ import annotations

import numbers
from typing import Any, cast


class ArrayBool:
    """Stands in for numpy's `bool`: it has a truth value, but is not Python's `bool`."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value


class ArrayFloat64(float):
    """Stands in for numpy's `float64`: a `float` whose comparisons give numpy's `bool`."""

    def __ge__(self, other: object) -> Any:
        return ArrayBool(float(self) >= cast("float", other))

    def __gt__(self, other: object) -> Any:
        return ArrayBool(float(self) > cast("float", other))

    def __le__(self, other: object) -> Any:
        return ArrayBool(float(self) <= cast("float", other))

    def __lt__(self, other: object) -> Any:
        return ArrayBool(float(self) < cast("float", other))


class ArrayFloat32:
    """Stands in for numpy's `float32`: a real number that is not a `float`."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    def __ge__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value >= cast("float", other))

    def __gt__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value > cast("float", other))

    def __le__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value <= cast("float", other))

    def __lt__(self, other: object) -> ArrayBool:
        return ArrayBool(self.value < cast("float", other))


numbers.Real.register(ArrayFloat32)
