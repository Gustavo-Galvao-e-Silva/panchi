from __future__ import annotations

from fractions import Fraction

Scalar = int | float | Fraction
SCALAR_TYPES = (int, float, Fraction)


def _parse_numeric_string(value: str) -> Scalar:
    """Parse a numeric or fractional string into a scalar.

    Parameters
    ----------
    value : str
        The string to parse.

    Returns
    -------
    Scalar
        The parsed int, float, or Fraction.

    Raises
    ------
    TypeError
        If the string cannot be parsed into an int, float, or Fraction.
    """
    if "/" in value:
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise TypeError(
                f"Cannot convert string '{value}' to a number. "
                f"Expected a numeric string or a fraction like '1/3'."
            ) from None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        raise TypeError(
            f"Cannot convert string '{value}' to a number. "
            f"Expected a numeric string or a fraction like '1/3'."
        ) from None


def parse_scalar(value: object) -> Scalar:
    """Parse a value into a scalar number.

    Strings like '1/3' become Fraction, integer strings become int,
    and floating-point strings become float.

    Parameters
    ----------
    value : object
        The value to convert into a scalar number.

    Returns
    -------
    Scalar
        The parsed int, float, or Fraction.

    Raises
    ------
    TypeError
        If `value` is a boolean, an invalid numeric string, or an unsupported type.

    Examples
    --------
    >>> parse_scalar(42)
    42
    >>> parse_scalar("1/3")
    Fraction(1, 3)
    >>> parse_scalar("2.5")
    2.5
    """
    if isinstance(value, bool):
        raise TypeError(
            "Cannot convert bool to a number. "
            "Booleans are not valid scalars; use 1 or 0 explicitly."
        )
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, str):
        return _parse_numeric_string(value)
    raise TypeError(
        f"Cannot convert {type(value).__name__} to a number. "
        f"Expected int, float, Fraction, or a string like '1/3'."
    )
