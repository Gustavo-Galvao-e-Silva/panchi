from __future__ import annotations

from fractions import Fraction
from math import lcm

from panchi.algorithms.matrix_operations import determinant
from panchi.algorithms.results import EigenResult
from panchi.algorithms.vector_space_operations import null_space
from panchi.primitives.factories import exact_matrix, identity
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.symbolic.polynomial import Polynomial
from panchi.symbolic.quadratic_number import QuadraticNumber
from panchi.symbolic.roots import exact_roots
from panchi.utils.types import Scalar


def _validate_exact_small_square(matrix: Matrix, task: str) -> None:
    """Check that matrix is a 1×1, 2×2 or 3×3 Matrix with rational entries."""
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"Expected a Matrix, but got {type(matrix).__name__}. "
            f"{task} is only defined for Matrix objects."
        )
    if not matrix.is_square:
        raise ValueError(
            f"Cannot compute the {task.lower()} of a non-square matrix. "
            f"Your matrix is {matrix.rows}×{matrix.cols}, but eigenvalues are "
            f"only defined for square matrices (n×n)."
        )
    if matrix.rows > 3:
        raise ValueError(
            f"Exact eigenvalues are supported for 2×2 and 3×3 matrices, but "
            f"your matrix is {matrix.rows}×{matrix.cols}. For larger matrices "
            f"use eigen_qr(), which approximates the eigenvalues numerically."
        )
    for row in matrix.data:
        for entry in row:
            if isinstance(entry, float):
                raise TypeError(
                    f"Exact eigenvalues need exact entries, but the matrix "
                    f"contains the float {entry}. Use ints or Fractions (for "
                    f"example exact_matrix()), or eigen_qr() for a numerical "
                    f"approximation."
                )
            if isinstance(entry, QuadraticNumber):
                raise TypeError(
                    f"Exact eigenvalues need rational entries, but the matrix "
                    f"contains {entry}. Use ints or Fractions."
                )


def _principal_minor_sum(matrix: Matrix) -> Scalar:
    """Sum the determinants of the 2×2 submatrices left by deleting row and column i."""
    n = matrix.rows
    total: Scalar = 0
    for skip in range(n):
        kept = [i for i in range(n) if i != skip]
        minor = Matrix([[matrix[i][j] for j in kept] for i in kept])
        total += determinant(minor)
    return total


def characteristic_polynomial(matrix: Matrix) -> Polynomial:
    """
    Compute the characteristic polynomial det(A - λI) exactly.

    Uses the formulas taught for small matrices, written in terms of the
    trace and determinant:

    - 2×2: λ² - tr(A)·λ + det(A)
    - 3×3: -λ³ + tr(A)·λ² - (sum of principal 2×2 minors)·λ + det(A)

    The principal 2×2 minors are the determinants left after deleting row
    i and column i, for each i.

    Parameters
    ----------
    matrix : Matrix
        A 1×1, 2×2 or 3×3 matrix with int or Fraction entries.

    Returns
    -------
    Polynomial
        The characteristic polynomial in λ. Its leading coefficient is
        (-1)ⁿ, so it is λ² + … for a 2×2 matrix and -λ³ + … for a 3×3.

    Raises
    ------
    TypeError
        If matrix is not a Matrix, or has float or irrational entries.
    ValueError
        If matrix is not square or is larger than 3×3.

    Examples
    --------
    >>> print(characteristic_polynomial(Matrix([[2, 1], [1, 2]])))
    λ² - 4λ + 3
    >>> print(characteristic_polynomial(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]])))
    -λ³ + 14λ² - 35λ + 22
    """
    _validate_exact_small_square(matrix, "Characteristic polynomial")
    n = matrix.rows
    if n == 1:
        return Polynomial([matrix[0][0], -1])
    if n == 2:
        return Polynomial([determinant(matrix), -matrix.trace, 1])
    return Polynomial(
        [determinant(matrix), -_principal_minor_sum(matrix), matrix.trace, -1]
    )


def _simple(value: Scalar) -> Scalar:
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    return value


def _clear_denominators(vector: Vector) -> Vector:
    """Scale a vector by the smallest whole number that removes every fraction."""
    denominators = []
    for component in vector.data:
        if isinstance(component, QuadraticNumber):
            denominators.append(component.rational.denominator)
            denominators.append(component.coefficient.denominator)
        else:
            denominators.append(Fraction(component).denominator)
    scale = lcm(*denominators)
    return Vector([_simple(component * scale) for component in vector.data])


def eigen(matrix: Matrix) -> EigenResult:
    """
    Compute the exact eigenvalues and eigenvectors of a 2×2 or 3×3 matrix.

    Follows the method used by hand:

    1. Form the characteristic polynomial det(A - λI).
    2. Solve it exactly. Rational roots are found with the rational root
       theorem and divided out, and any quadratic left over is solved with
       the quadratic formula.
    3. For each eigenvalue λ, the eigenspace is the null space of A - λI,
       found by row reduction.

    Answers are exact: irrational eigenvalues such as (1 + √5)/2 and
    complex ones such as 1 - 2i are QuadraticNumbers, and the eigenvectors
    are scaled so they contain no fractions.

    Parameters
    ----------
    matrix : Matrix
        A 1×1, 2×2 or 3×3 matrix with int or Fraction entries.

    Returns
    -------
    EigenResult
        The characteristic polynomial, the distinct eigenvalues with their
        algebraic multiplicities, and the eigenspaces.

    Raises
    ------
    TypeError
        If matrix is not a Matrix, or has float or irrational entries.
    ValueError
        If matrix is not square, is larger than 3×3, or is a 3×3 matrix
        whose characteristic polynomial has no rational root (its
        eigenvalues then involve cube roots and cannot be written exactly
        as a + b√d).

    See Also
    --------
    eigen_qr : Numerical eigenvalues via the QR algorithm, for any size.

    Examples
    --------
    >>> result = eigen(Matrix([[2, 1], [1, 2]]))
    >>> result.eigenvalues
    [1, 3]
    >>> [str(v) for v in eigen(Matrix([[0, -1], [1, 0]])).eigenvalues]
    ['-i', 'i']
    """
    polynomial = characteristic_polynomial(matrix)
    try:
        roots = exact_roots(polynomial)
    except ValueError as error:
        raise ValueError(
            f"{error} Use eigen_qr() to approximate these eigenvalues numerically."
        ) from None

    n = matrix.rows
    exact = exact_matrix(matrix.to_list())
    eigenvalues = []
    multiplicities = []
    eigenspaces = []
    for value, multiplicity in roots:
        space = null_space(exact - value * identity(n))
        eigenvalues.append(value)
        multiplicities.append(multiplicity)
        eigenspaces.append(VectorSpace([_clear_denominators(v) for v in space]))
    return EigenResult(matrix, polynomial, eigenvalues, multiplicities, eigenspaces)
