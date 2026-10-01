"""Set a context variable for the length of one block.

The monitor keeps in context variables the values that every call inside a
block must see, such as the config LangChain's runs inherit, the labels of a
step's spans and the tool request the outermost monitor is checking.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


@contextmanager
def set_context_value[ValueT](variable: ContextVar[ValueT], *, value: ValueT) -> Iterator[None]:
    """Set the variable for the block, and give it back its earlier value however the block ends."""
    token = variable.set(value)
    try:
        yield
    finally:
        variable.reset(token)
