from __future__ import annotations

from fractions import Fraction
from math import lcm

from panchi.algorithms.decompositions import qr_decomposition
from panchi.algorithms.matrix_operations import (
    _require_square_matrix,
    determinant,
    solve,
)
from panchi.algorithms.results import EigenResult, QREigenResult
from panchi.algorithms.vector_space_operations import null_space
from panchi.primitives.factories import exact_matrix, identity, zero_vector
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.symbolic.polynomial import Polynomial
from panchi.symbolic.quadratic_number import QuadraticNumber, _simplify
from panchi.symbolic.roots import exact_roots
from panchi.utils.types import Scalar


def _validate_exact_small_square(matrix: Matrix, task: str) -> None:
    """Check that matrix is a 1×1, 2×2 or 3×3 Matrix with rational entries."""
    _require_square_matrix(matrix, task.lower())
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
    return Vector([_simplify(component * scale) for component in vector.data])


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


def _below_diagonal_mass(matrix: Matrix) -> float:
    """
    Sum the absolute values of the strictly-below-diagonal entries.

    Used as the convergence measure for the QR algorithm: when this sum is
    small, the iterate is (near) upper triangular and its diagonal holds the
    eigenvalues.
    """
    n = matrix.rows
    return sum(abs(matrix[i][j]) for i in range(n) for j in range(i))


def _eigenvector(matrix: Matrix, eigenvalue: float, n: int) -> Vector | None:
    """
    Find an eigenvector for a known eigenvalue as the null space of A - λI.

    Solves the homogeneous system ``(A - λI) x = 0`` with a scale-relative
    tolerance, reusing solve(). Because a floating-point λ makes ``A - λI``
    only approximately singular, the tolerance is what lets solve() report an
    infinite solution set and expose the null space. Returns the first
    null-space basis vector, normalized, or None if the system is not
    detected as rank-deficient (e.g. for tightly clustered eigenvalues).
    """
    shifted = matrix - eigenvalue * identity(n)
    scale = max(1.0, max(abs(shifted[i][j]) for i in range(n) for j in range(n)))
    solution = solve(shifted, zero_vector(n), tolerance=1e-6 * scale)

    if solution.null_space is None:
        return None
    return solution.null_space[0].normalize()


def eigen_qr(
    matrix: Matrix,
    max_iterations: int = 1000,
    tolerance: float = 1e-12,
) -> QREigenResult:
    """
    Compute the eigenvalues of a square matrix using the QR algorithm.

    Starting from the original matrix, repeatedly computes a QR
    decomposition and reassembles the matrix as R @ Q. For matrices with
    real eigenvalues this sequence converges to an upper triangular matrix
    whose diagonal entries are the eigenvalues.

    The iteration stops once the strictly-below-diagonal entries sum to less
    than ``tolerance`` (recorded as converged), or once ``max_iterations`` is
    reached (recorded as not converged). Eigenvalues are the diagonal of the
    final iterate. Eigenvectors are computed separately and are only
    populated when the iteration converges.

    Only real eigenvalues are supported. Matrices with complex eigenvalues
    or eigenvalues of equal magnitude may fail to converge, in which case the
    returned result has ``converged`` set to False and no eigenvectors.

    Parameters
    ----------
    matrix : Matrix
        The square matrix whose eigenvalues will be computed.
    max_iterations : int, optional
        The maximum number of QR iterations to perform. Defaults to 1000.
    tolerance : float, optional
        The convergence threshold on the below-diagonal mass. Defaults to
        1e-12.

    Returns
    -------
    QREigenResult
        A result object containing the eigenvalues, eigenvectors, the number
        of iterations performed, whether the iteration converged, and the
        final (near) upper-triangular matrix.

    Raises
    ------
    TypeError
        If matrix is not a Matrix instance.
    ValueError
        If matrix is not square.

    Examples
    --------
    >>> result = eigen_qr(Matrix([[2, 1], [1, 2]]))
    >>> sorted(round(v, 6) for v in result.eigenvalues)
    [1.0, 3.0]
    """
    _require_square_matrix(matrix, "eigenvalues")

    n = matrix.rows
    current = matrix
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if _below_diagonal_mass(current) < tolerance:
            converged = True
            iterations -= 1
            break
        try:
            decomposition = qr_decomposition(current)
        except ZeroDivisionError:
            # A singular iterate cannot be orthonormalized by Gram-Schmidt;
            # treat this as failure to converge rather than crashing.
            break
        current = decomposition.r @ decomposition.q

    eigenvalues = [current[i][i] for i in range(n)]
    eigenvectors: list[Vector] = []
    if converged:
        for eigenvalue in eigenvalues:
            vector = _eigenvector(matrix, eigenvalue, n)
            if vector is not None:
                eigenvectors.append(vector)
    return QREigenResult(
        matrix, eigenvalues, eigenvectors, iterations, converged, current
    )
