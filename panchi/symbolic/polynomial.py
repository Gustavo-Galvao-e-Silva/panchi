from __future__ import annotations

from fractions import Fraction
from math import isqrt, lcm

from panchi.symbolic.quadratic_number import (
    QuadraticNumber,
    Rational,
    Scalar,
    _simplify,
)
from panchi.utils.latex import scalar_to_latex

_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_LATEX_VARIABLES = {"λ": "\\lambda"}


def _divisors(n: int) -> list[int]:
    """Return the positive divisors of a non-zero integer."""
    n = abs(n)
    divisors = set()
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            divisors.add(d)
            divisors.add(n // d)
    return sorted(divisors)


class Polynomial:
    """
    A polynomial in one variable with exact coefficients.

    Coefficients are listed from the constant term upwards, so
    ``Polynomial([6, -5, 1])`` is λ² - 5λ + 6. Coefficients may be ints,
    Fractions or QuadraticNumbers, and trailing zero coefficients are
    dropped.

    Parameters
    ----------
    coefficients : list[int | Fraction | QuadraticNumber]
        The coefficients, constant term first.
    variable : str, optional
        The name of the variable used when printing. Defaults to "λ".

    Raises
    ------
    TypeError
        If any coefficient is not an exact number.

    Examples
    --------
    >>> p = Polynomial([6, -5, 1])
    >>> print(p)
    λ² - 5λ + 6
    >>> p(2)
    0
    """

    def __init__(self, coefficients: list[Scalar], variable: str = "λ") -> None:
        for c in coefficients:
            if isinstance(c, bool) or not isinstance(
                c, (int, Fraction, QuadraticNumber)
            ):
                raise TypeError(
                    f"Polynomial coefficients must be exact numbers (int, "
                    f"Fraction or QuadraticNumber), but got {type(c).__name__}."
                )
        trimmed = list(coefficients)
        while len(trimmed) > 1 and trimmed[-1] == 0:
            trimmed.pop()
        self.coefficients: list[Scalar] = trimmed or [0]
        self.variable = variable

    @property
    def degree(self) -> int:
        """The highest power with a non-zero coefficient (0 for constants)."""
        return len(self.coefficients) - 1

    @property
    def leading_coefficient(self) -> Scalar:
        """The coefficient of the highest power."""
        return self.coefficients[-1]

    def _coerce(self, other: object) -> Polynomial | None:
        if isinstance(other, Polynomial):
            return other
        if isinstance(other, bool):
            return None
        if isinstance(other, (int, Fraction, QuadraticNumber)):
            return Polynomial([other], self.variable)
        return None

    def __add__(self, other: object) -> Polynomial:
        other_poly = self._coerce(other)
        if other_poly is None:
            return NotImplemented
        size = max(len(self.coefficients), len(other_poly.coefficients))
        mine = self.coefficients + [0] * (size - len(self.coefficients))
        theirs = other_poly.coefficients + [0] * (size - len(other_poly.coefficients))
        return Polynomial(
            [a + b for a, b in zip(mine, theirs, strict=True)], self.variable
        )

    __radd__ = __add__

    def __neg__(self) -> Polynomial:
        return Polynomial([-c for c in self.coefficients], self.variable)

    def __sub__(self, other: object) -> Polynomial:
        other_poly = self._coerce(other)
        if other_poly is None:
            return NotImplemented
        return self + (-other_poly)

    def __rsub__(self, other: object) -> Polynomial:
        other_poly = self._coerce(other)
        if other_poly is None:
            return NotImplemented
        return other_poly + (-self)

    def __mul__(self, other: object) -> Polynomial:
        other_poly = self._coerce(other)
        if other_poly is None:
            return NotImplemented
        product: list[Scalar] = [0] * (
            len(self.coefficients) + len(other_poly.coefficients) - 1
        )
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other_poly.coefficients):
                product[i + j] = product[i + j] + a * b
        return Polynomial(product, self.variable)

    __rmul__ = __mul__

    def __call__(self, value: Scalar) -> Scalar:
        """
        Evaluate the polynomial at a value using Horner's method.

        Examples
        --------
        >>> Polynomial([6, -5, 1])(3)
        0
        """
        result: Scalar = 0
        for c in reversed(self.coefficients):
            result = result * value + c
        return result

    def __eq__(self, other: object) -> bool:
        other_poly = self._coerce(other)
        if other_poly is None:
            return NotImplemented
        return self.coefficients == other_poly.coefficients

    def __hash__(self) -> int:
        return hash(tuple(self.coefficients))

    def divide_by_root(self, root: Scalar) -> tuple[Polynomial, Scalar]:
        """
        Divide by (λ - root) using synthetic division.

        Parameters
        ----------
        root : int | Fraction | QuadraticNumber
            The value r in the divisor (λ - r).

        Returns
        -------
        tuple[Polynomial, Scalar]
            The quotient and the remainder. The remainder is zero exactly
            when root is a root of the polynomial.

        Examples
        --------
        >>> quotient, remainder = Polynomial([6, -5, 1]).divide_by_root(2)
        >>> print(quotient)
        λ - 3
        >>> remainder
        0
        """
        running: Scalar = 0
        quotient: list[Scalar] = []
        for c in reversed(self.coefficients):
            running = running * root + c
            quotient.append(running)
        remainder = quotient.pop()
        quotient.reverse()
        return Polynomial(quotient or [0], self.variable), remainder

    def rational_roots(self) -> list[Rational]:
        """
        Find every rational root using the rational root theorem.

        After clearing denominators, any rational root p/q in lowest terms
        has p dividing the constant term and q dividing the leading
        coefficient, so only finitely many candidates need checking.

        Returns
        -------
        list[int | Fraction]
            The distinct rational roots, in increasing order.

        Raises
        ------
        TypeError
            If any coefficient is irrational or complex.

        Examples
        --------
        >>> Polynomial([-6, 11, -6, 1]).rational_roots()
        [1, 2, 3]
        """
        if any(isinstance(c, QuadraticNumber) for c in self.coefficients):
            raise TypeError(
                "The rational root theorem only applies to polynomials with "
                "rational coefficients."
            )
        if self.degree == 0:
            return []
        scale = lcm(*(Fraction(c).denominator for c in self.coefficients))
        integers = [int(c * scale) for c in self.coefficients]

        roots: list[Rational] = []
        lowest = 0
        while integers[lowest] == 0:
            lowest += 1
        if lowest > 0:
            roots.append(0)
        integers = integers[lowest:]
        if len(integers) == 1:
            return roots

        candidates = set()
        for p in _divisors(integers[0]):
            for q in _divisors(integers[-1]):
                candidates.add(Fraction(p, q))
                candidates.add(Fraction(-p, q))
        shifted = Polynomial(integers, self.variable)
        for candidate in sorted(candidates):
            if shifted(candidate) == 0:
                roots.append(_simplify(candidate))
        return sorted(roots)

    def _term(self, power: int, latex: bool) -> str:
        variable = (
            _LATEX_VARIABLES.get(self.variable, self.variable)
            if latex
            else self.variable
        )
        if power == 0:
            return ""
        if power == 1:
            return variable
        if latex:
            return f"{variable}^{{{power}}}"
        return f"{variable}{str(power).translate(_SUPERSCRIPTS)}"

    def _coefficient_text(self, value: Scalar, latex: bool) -> str:
        if latex:
            return scalar_to_latex(value)
        return str(value)

    def _render(self, latex: bool) -> str:
        pieces: list[str] = []
        for power in range(self.degree, -1, -1):
            c = self.coefficients[power]
            if c == 0 and self.degree > 0:
                continue
            term = self._term(power, latex)
            if isinstance(c, QuadraticNumber):
                text = self._coefficient_text(c, latex)
                sign, body = "+", f"({text})" if term else text
            else:
                sign = "-" if c < 0 else "+"
                magnitude = abs(c)
                body = (
                    ""
                    if magnitude == 1 and term
                    else self._coefficient_text(magnitude, latex)
                )
                if term and not latex and "/" in body:
                    body = f"({body})"
            pieces.append(sign)
            pieces.append(f"{body}{term}")
        first_sign, first_body = pieces[0], pieces[1]
        text = ("-" if first_sign == "-" else "") + first_body
        for i in range(2, len(pieces), 2):
            text += f" {pieces[i]} {pieces[i + 1]}"
        return text

    def __str__(self) -> str:
        """
        Return the polynomial written from the highest power down.

        Examples
        --------
        >>> print(Polynomial([-2, 0, 0, 1]))
        λ³ - 2
        """
        return self._render(latex=False)

    def __repr__(self) -> str:
        return f"Polynomial({self.coefficients!r})"

    def _latex(self) -> str:
        """Render the polynomial as a LaTeX fragment (no math delimiters)."""
        return self._render(latex=True)

    def _repr_latex_(self) -> str:
        return f"${self._latex()}$"
