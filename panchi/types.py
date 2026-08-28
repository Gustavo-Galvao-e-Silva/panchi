from __future__ import annotations

from fractions import Fraction

Scalar = int | float | Fraction
SCALAR_TYPES = (int, float, Fraction)


def parse_scalar(value) -> int | float | Fraction:
    """Parse a value into a scalar. Strings like '1/3' become Fraction."""
    if isinstance(value, bool):
        raise TypeError(
            "Cannot convert bool to a number. "
            "Booleans are not valid scalars; use 1 or 0 explicitly."
        )
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, str):
        try:
            if "/" in value:
                return Fraction(value)
            try:
                return int(value)
            except ValueError:
                return float(value)
        except (ValueError, ZeroDivisionError):
            raise TypeError(
                f"Cannot convert string '{value}' to a number. "
                f"Expected a numeric string or a fraction like '1/3'."
            )
    raise TypeError(
        f"Cannot convert {type(value).__name__} to a number. "
        f"Expected int, float, Fraction, or a string like '1/3'."
    )
