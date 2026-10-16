from __future__ import annotations

from fractions import Fraction
from math import lcm

Rational = int | Fraction


def _squarefree_split(n: int) -> tuple[int, int]:
    """Split a non-zero integer n into (s, m) with n = s² · m and m squarefree."""
    sign = -1 if n < 0 else 1
    remaining = abs(n)
    square_root = 1
    factor = 2
    while factor * factor <= remaining:
        while remaining % (factor * factor) == 0:
            remaining //= factor * factor
            square_root *= factor
        factor += 1
    return square_root, sign * remaining


def _simplify(value: Fraction) -> Rational:
    """Return an int when the Fraction is whole, otherwise the Fraction."""
    if value.denominator == 1:
        return value.numerator
    return value


def _make(rational: Fraction, coefficient: Fraction, radicand: int) -> Scalar:
    """Build a + b√d, collapsing to an int or Fraction when b is zero."""
    if coefficient == 0:
        return _simplify(rational)
    return QuadraticNumber(rational, coefficient, radicand)


def sqrt(value: Rational) -> Rational | QuadraticNumber:
    """
    Return the exact square root of a rational number.

    Perfect squares give an int or Fraction. Anything else gives a
    QuadraticNumber in simplest radical form, so ``sqrt(12)`` is ``2√3``.
    Negative values give imaginary results, so ``sqrt(-4)`` is ``2i``.

    Parameters
    ----------
    value : int | Fraction
        The rational number whose square root to take.

    Returns
    -------
    int | Fraction | QuadraticNumber
        The exact square root.

    Raises
    ------
    TypeError
        If value is not an int or Fraction.

    Examples
    --------
    >>> sqrt(9)
    3
    >>> print(sqrt(8))
    2√2
    >>> print(sqrt(Fraction(1, 2)))
    √2/2
    >>> print(sqrt(-1))
    i
    """
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(
            f"sqrt() needs an exact rational number (int or Fraction), "
            f"but got {type(value).__name__}. "
            f"Floats cannot be square-rooted exactly."
        )
    value = Fraction(value)
    if value == 0:
        return 0
    denominator = value.denominator
    square_root, radicand = _squarefree_split(value.numerator * denominator)
    coefficient = Fraction(square_root, denominator)
    if radicand == 1:
        return _simplify(coefficient)
    return QuadraticNumber(0, coefficient, radicand)


class QuadraticNumber:
    """
    An exact number of the form a + b√d with rational a and b.

    The radicand d is a squarefree integer other than 0 and 1. A negative
    radicand makes the number complex: d = -1 is the imaginary unit i, so
    ``QuadraticNumber(3, -2, -1)`` is 3 - 2i. These numbers are closed under
    +, -, × and ÷ as long as every operand shares the same radicand, which
    is exactly what the eigenvalues of a 2×2 or 3×3 matrix need.

    Arithmetic results whose √d part cancels collapse back to a plain int
    or Fraction, so exact computations stay as simple as possible.

    Parameters
    ----------
    rational : int | Fraction
        The rational part a.
    coefficient : int | Fraction
        The coefficient b of √d.
    radicand : int
        The integer d under the square root. It is reduced to squarefree
        form, so ``QuadraticNumber(0, 1, 12)`` is stored as 2√3.

    Raises
    ------
    TypeError
        If a or b is not an int or Fraction, or d is not an int.
    ValueError
        If d is a perfect square (including 0 and 1), since √d would
        then be rational.

    Examples
    --------
    >>> golden = QuadraticNumber(Fraction(1, 2), Fraction(1, 2), 5)
    >>> print(golden)
    (1 + √5)/2
    >>> golden * golden == golden + 1
    True
    >>> print(QuadraticNumber(1, 1, -1) * QuadraticNumber(1, -1, -1))
    2
    """

    def __init__(
        self, rational: Rational, coefficient: Rational, radicand: int
    ) -> None:
        for name, part in (("rational", rational), ("coefficient", coefficient)):
            if isinstance(part, bool) or not isinstance(part, (int, Fraction)):
                raise TypeError(
                    f"The {name} part of a QuadraticNumber must be an int or "
                    f"Fraction, but got {type(part).__name__}."
                )
        if isinstance(radicand, bool) or not isinstance(radicand, int):
            raise TypeError(
                f"The radicand of a QuadraticNumber must be an int, "
                f"but got {type(radicand).__name__}."
            )
        if radicand == 0:
            raise ValueError("The radicand cannot be 0, since √0 = 0 is rational.")
        square_root, squarefree = _squarefree_split(radicand)
        if squarefree == 1:
            raise ValueError(
                f"The radicand {radicand} is a perfect square, so √{radicand} "
                f"= {square_root} is rational. Use a plain int or Fraction."
            )
        self.rational = Fraction(rational)
        self.coefficient = Fraction(coefficient) * square_root
        self.radicand = squarefree

    def _parts(self, other: object) -> tuple[Fraction, Fraction] | None:
        """Return other as (a, b) over this radicand, or None if unsupported."""
        if isinstance(other, float):
            raise TypeError(
                "Cannot combine an exact QuadraticNumber with a float. "
                "Floats would destroy exactness; use int or Fraction values."
            )
        if isinstance(other, bool):
            return None
        if isinstance(other, (int, Fraction)):
            return Fraction(other), Fraction(0)
        if isinstance(other, QuadraticNumber):
            if other.coefficient != 0 and other.radicand != self.radicand:
                raise ValueError(
                    f"Cannot combine numbers involving √{self.radicand} and "
                    f"√{other.radicand}. A QuadraticNumber holds a single "
                    f"square root, so both operands must share the same radicand."
                )
            return other.rational, other.coefficient
        return None

    def __add__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        return _make(self.rational + a, self.coefficient + b, self.radicand)

    __radd__ = __add__

    def __neg__(self) -> QuadraticNumber:
        return QuadraticNumber(-self.rational, -self.coefficient, self.radicand)

    def __pos__(self) -> QuadraticNumber:
        return self

    def __sub__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        return _make(self.rational - a, self.coefficient - b, self.radicand)

    def __rsub__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        return _make(a - self.rational, b - self.coefficient, self.radicand)

    def __mul__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        return _make(
            self.rational * a + self.coefficient * b * self.radicand,
            self.rational * b + self.coefficient * a,
            self.radicand,
        )

    __rmul__ = __mul__

    def norm(self) -> Fraction:
        """
        Return the field norm (a + b√d)(a - b√d) = a² - d·b².

        The norm is always rational. For a complex number it is |z|².
        """
        return self.rational**2 - self.radicand * self.coefficient**2

    def conjugate(self) -> QuadraticNumber:
        """
        Return the conjugate a - b√d.

        For a complex number this is the usual complex conjugate.
        """
        return QuadraticNumber(self.rational, -self.coefficient, self.radicand)

    def _reciprocal(self) -> Scalar:
        norm = self.norm()
        if norm == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return _make(self.rational / norm, -self.coefficient / norm, self.radicand)

    def __truediv__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        if b == 0:
            if a == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return _make(self.rational / a, self.coefficient / a, self.radicand)
        divisor = QuadraticNumber(a, b, self.radicand)
        return self * divisor._reciprocal()

    def __rtruediv__(self, other: object) -> Scalar:
        parts = self._parts(other)
        if parts is None:
            return NotImplemented
        a, b = parts
        return _make(a, b, self.radicand) * self._reciprocal()

    def __pow__(self, exponent: int) -> Scalar:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            return NotImplemented
        base: Scalar = self if exponent >= 0 else self._reciprocal()
        result: Scalar = 1
        for _ in range(abs(exponent)):
            result = result * base
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return False
        if isinstance(other, (int, Fraction)):
            return self.coefficient == 0 and self.rational == other
        if isinstance(other, QuadraticNumber):
            if self.coefficient == 0 and other.coefficient == 0:
                return self.rational == other.rational
            return (
                self.rational == other.rational
                and self.coefficient == other.coefficient
                and self.radicand == other.radicand
            )
        return NotImplemented

    def __hash__(self) -> int:
        if self.coefficient == 0:
            return hash(self.rational)
        return hash((self.rational, self.coefficient, self.radicand))

    def __bool__(self) -> bool:
        return self.rational != 0 or self.coefficient != 0

    @property
    def is_real(self) -> bool:
        """True when the number has no imaginary part."""
        return self.radicand > 0 or self.coefficient == 0

    @property
    def real(self) -> Scalar:
        """The real part, exactly."""
        if self.is_real:
            return self
        return _simplify(self.rational)

    @property
    def imag(self) -> Scalar:
        """The imaginary part, exactly (so the imaginary part of 1 + i√2 is √2)."""
        if self.is_real:
            return 0
        imaginary = self.coefficient * sqrt(-self.radicand)
        return _simplify(imaginary) if isinstance(imaginary, Fraction) else imaginary

    def _sign(self) -> int:
        """Return -1, 0 or 1 for a real number, using exact comparison."""
        a_sign = (self.rational > 0) - (self.rational < 0)
        b_sign = (self.coefficient > 0) - (self.coefficient < 0)
        if a_sign == b_sign or b_sign == 0:
            return a_sign
        if a_sign == 0:
            return b_sign
        rational_square = self.rational**2
        radical_square = self.coefficient**2 * self.radicand
        if rational_square > radical_square:
            return a_sign
        if rational_square < radical_square:
            return b_sign
        return 0

    def _compare(self, other: object) -> int | None:
        """Return the sign of self - other, or None if unsupported."""
        if not self.is_real:
            raise TypeError(
                f"Cannot order the complex number {self}. "
                f"Only real numbers can be compared with <, <=, > or >=."
            )
        if isinstance(other, float):
            if other == 0:
                return self._sign()
            approximation = float(self)
            return (approximation > other) - (approximation < other)
        difference = self - other
        if difference is NotImplemented:
            return None
        if isinstance(difference, QuadraticNumber):
            if not difference.is_real:
                raise TypeError(
                    f"Cannot order {self} against the complex number {other}."
                )
            return difference._sign()
        return (difference > 0) - (difference < 0)

    def __lt__(self, other: object) -> bool:
        sign = self._compare(other)
        return NotImplemented if sign is None else sign < 0

    def __le__(self, other: object) -> bool:
        sign = self._compare(other)
        return NotImplemented if sign is None else sign <= 0

    def __gt__(self, other: object) -> bool:
        sign = self._compare(other)
        return NotImplemented if sign is None else sign > 0

    def __ge__(self, other: object) -> bool:
        sign = self._compare(other)
        return NotImplemented if sign is None else sign >= 0

    def __abs__(self) -> Scalar:
        """
        Return the exact absolute value (the modulus, for complex numbers).

        For a complex number this is √(a² - d·b²), which may involve a
        different square root than the number itself.
        """
        if self.is_real:
            return -self if self._sign() < 0 else self
        return sqrt(self.norm())

    def __float__(self) -> float:
        if not self.is_real:
            raise TypeError(
                f"Cannot convert the complex number {self} to a float. "
                f"Use complex() instead."
            )
        return float(self.rational) + float(self.coefficient) * self.radicand**0.5

    def __complex__(self) -> complex:
        if self.is_real:
            return complex(float(self))
        return complex(
            float(self.rational),
            float(self.coefficient) * (-self.radicand) ** 0.5,
        )

    def _integer_parts(self) -> tuple[int, int, int]:
        """Return (A, B, L) with a + b√d = (A + B√d) / L over a common denominator."""
        denominator = lcm(self.rational.denominator, self.coefficient.denominator)
        return (
            int(self.rational * denominator),
            int(self.coefficient * denominator),
            denominator,
        )

    def _radical_term(self, magnitude: int, latex: bool) -> str:
        """Render |B|·√|d| (with i for complex numbers) for a positive magnitude."""
        prefix = "" if magnitude == 1 else str(magnitude)
        root = abs(self.radicand)
        imaginary = "i" if self.radicand < 0 else ""
        if root == 1:
            return f"{prefix}{imaginary}"
        radical = f"\\sqrt{{{root}}}" if latex else f"√{root}"
        return f"{prefix}{imaginary}{radical}"

    def _render(self, latex: bool) -> str:
        if self.coefficient == 0:
            return str(_simplify(self.rational))
        whole, radical_coefficient, denominator = self._integer_parts()
        radical = self._radical_term(abs(radical_coefficient), latex)
        if whole == 0:
            sign = "-" if radical_coefficient < 0 else ""
            if denominator == 1:
                return f"{sign}{radical}"
            if latex:
                return f"{sign}\\frac{{{radical}}}{{{denominator}}}"
            return f"{sign}{radical}/{denominator}"
        operator = "-" if radical_coefficient < 0 else "+"
        numerator = f"{whole} {operator} {radical}"
        if denominator == 1:
            return numerator
        if latex:
            return f"\\frac{{{numerator}}}{{{denominator}}}"
        return f"({numerator})/{denominator}"

    def __str__(self) -> str:
        """
        Return the number in standard mathematical form.

        Examples
        --------
        >>> print(QuadraticNumber(Fraction(1, 2), Fraction(1, 2), 5))
        (1 + √5)/2
        >>> print(QuadraticNumber(1, -2, -3))
        1 - 2i√3
        """
        return self._render(latex=False)

    def __repr__(self) -> str:
        return (
            f"QuadraticNumber({_simplify(self.rational)!r}, "
            f"{_simplify(self.coefficient)!r}, "
            f"{self.radicand})"
        )

    def _latex(self) -> str:
        """Render the number as a LaTeX fragment (no math delimiters)."""
        return self._render(latex=True)

    def _repr_latex_(self) -> str:
        return f"${self._latex()}$"


Scalar = int | Fraction | QuadraticNumber
