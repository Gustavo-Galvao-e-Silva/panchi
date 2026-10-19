"""
Exact symbolic numbers and polynomials.

QuadraticNumber represents numbers of the form a + b√d exactly, including
complex numbers such as 1 - 2i. Polynomial represents polynomials with
exact coefficients, and the root finders solve them exactly whenever the
roots can be written as a + b√d, which covers the characteristic
polynomial of every 2×2 matrix and of most 3×3 matrices met in a course.
"""

from panchi.symbolic.polynomial import Polynomial
from panchi.symbolic.quadratic_number import QuadraticNumber, sqrt
from panchi.symbolic.roots import exact_roots, quadratic_roots

__all__ = [
    "QuadraticNumber",
    "sqrt",
    "Polynomial",
    "exact_roots",
    "quadratic_roots",
]
