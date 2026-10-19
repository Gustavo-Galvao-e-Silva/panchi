from __future__ import annotations

from fractions import Fraction

from panchi.symbolic.polynomial import Polynomial
from panchi.symbolic.quadratic_number import Scalar, _simplify, sqrt


def _root_order(root: Scalar) -> tuple[bool, float, float]:
    """Sort key: real roots in increasing order, then complex ones."""
    value = complex(root)
    return (value.imag != 0, value.real, value.imag)


def quadratic_roots(polynomial: Polynomial) -> list[tuple[Scalar, int]]:
    """
    Solve a quadratic exactly with the quadratic formula.

    For aλ² + bλ + c the roots are (-b ± √(b² - 4ac)) / 2a. The
    discriminant b² - 4ac decides their kind: a perfect square gives
    rational roots, any other positive value gives irrational ones such
    as 1 ± √2, a negative value gives a complex conjugate pair, and zero
    gives a single repeated root.

    Parameters
    ----------
    polynomial : Polynomial
        A degree-2 polynomial with rational coefficients.

    Returns
    -------
    list[tuple[Scalar, int]]
        The distinct roots paired with their multiplicities.

    Raises
    ------
    ValueError
        If the polynomial is not quadratic.

    Examples
    --------
    >>> [(str(r), m) for r, m in quadratic_roots(Polynomial([1, -2, 1]))]
    [('1', 2)]
    >>> [str(r) for r, _ in quadratic_roots(Polynomial([2, -2, 1]))]
    ['1 - i', '1 + i']
    """
    if polynomial.degree != 2:
        raise ValueError(
            f"quadratic_roots() needs a degree-2 polynomial, but {polynomial} "
            f"has degree {polynomial.degree}."
        )
    c, b, a = polynomial.coefficients
    discriminant = b * b - 4 * a * c
    if discriminant == 0:
        return [(_simplify(-b / Fraction(2 * a)), 2)]
    root = sqrt(discriminant)
    roots = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
    roots = [_simplify(r) for r in roots]
    return [(r, 1) for r in sorted(roots, key=_root_order)]


def exact_roots(polynomial: Polynomial) -> list[tuple[Scalar, int]]:
    """
    Find every root of a polynomial exactly, with multiplicities.

    Rational roots are found with the rational root theorem and divided
    out by synthetic division, repeatedly, so repeated roots are counted.
    Whatever remains must have degree at most 2, which is then solved with
    the quadratic formula. This is the by-hand method for characteristic
    polynomials of 2×2 and 3×3 matrices.

    Parameters
    ----------
    polynomial : Polynomial
        A non-constant polynomial with rational coefficients.

    Returns
    -------
    list[tuple[Scalar, int]]
        The distinct roots paired with their multiplicities, real roots
        first in increasing order, then complex ones.

    Raises
    ------
    ValueError
        If the polynomial is constant, or if after removing rational roots
        a factor of degree 3 or more remains (for example λ³ - 2, whose
        roots involve cube roots and cannot be written as a + b√d).

    Examples
    --------
    >>> [(str(r), m) for r, m in exact_roots(Polynomial([-4, 8, -5, 1]))]
    [('1', 1), ('2', 2)]
    """
    if polynomial.degree < 1:
        raise ValueError(f"A constant polynomial ({polynomial}) has no roots to find.")
    multiplicities: dict[Scalar, int] = {}
    remaining = polynomial
    for root in polynomial.rational_roots():
        while remaining.degree > 0:
            quotient, remainder = remaining.divide_by_root(root)
            if remainder != 0:
                break
            multiplicities[root] = multiplicities.get(root, 0) + 1
            remaining = quotient

    if remaining.degree == 1:
        constant, slope = remaining.coefficients
        root = _simplify(Fraction(-constant) / slope)
        multiplicities[root] = multiplicities.get(root, 0) + 1
    elif remaining.degree == 2:
        for root, count in quadratic_roots(remaining):
            multiplicities[root] = multiplicities.get(root, 0) + count
    elif remaining.degree > 2:
        if remaining == polynomial:
            reason = "It has no rational roots"
        else:
            reason = (
                f"After removing its rational roots, the factor {remaining} "
                f"is left, which has no rational roots"
            )
        raise ValueError(
            f"Cannot find the roots of {polynomial} exactly. {reason}, so its "
            f"roots involve cube roots (or worse) and cannot be written in "
            f"the form a + b√d."
        )
    return sorted(multiplicities.items(), key=lambda pair: _root_order(pair[0]))
