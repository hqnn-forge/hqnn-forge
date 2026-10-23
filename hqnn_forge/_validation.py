"""Argument checks shared across the library."""

from __future__ import annotations

import numbers
from typing import Any, Literal, overload


@overload
def check_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    allow_none: Literal[False] = False,
) -> int:
    ...


@overload
def check_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    allow_none: Literal[True],
) -> int | None:
    ...


def check_int(
    value: Any, name: str, *, minimum: int, allow_none: bool = False
) -> int | None:
    """``value`` as an ``int >= minimum`` (``None`` too if ``allow_none``).

    A bool, a float, a string or anything else that is not a
    :class:`numbers.Integral` is refused; NumPy integers are accepted
    and converted with ``int()``.

    Raises
    ------
    ValueError
        If ``value`` is not such an integer.
    """
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        suffix = " or None" if allow_none else ""
        raise ValueError(
            f"{name} must be an integer >= {minimum}{suffix}; got {value!r}."
        )
    return int(value)