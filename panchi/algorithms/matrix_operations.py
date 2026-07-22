from panchi.types import Scalar
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.factories import identity, zero_vector
from panchi.algorithms.row_operations import RowOperation, RowSwap
from panchi.algorithms.results import InverseResult, Reduction, Solution, EigenResult
from panchi.algorithms.reductions import rref
from panchi.algorithms.decompositions import lu, qr_decomposition


def _calculate_inverse(n: int, steps: list[RowOperation]) -> Matrix:
    """
    Reconstruct the inverse matrix by replaying row operations on the identity.

    Applies each step from the RREF reduction sequentially to an n×n identity
    matrix. The result is the inverse of the original matrix, since the same
    sequence of operations that reduces A to I also transforms I into A⁻¹.

    Parameters
    ----------
    n : int
        The size of the square matrix being inverted.
    steps : list[RowOperation]
        The ordered row operations produced by reducing the original matrix
        to RREF.

    Returns
    -------
    Matrix
        The n×n inverse matrix.
    """
    result = identity(n)
    for step in steps:
        result = step.apply(result)
    return result


def _main_diagonal_product(matrix: Matrix) -> Scalar:
    """
    Compute the product of all entries on the main diagonal.

    Parameters
    ----------
    matrix : Matrix
        A square matrix whose diagonal entries will be multiplied together.

    Returns
    -------
    float
        The product of entries matrix[0][0], matrix[1][1], ..., matrix[n-1][n-1].
    """
    result = 1
    n = matrix.rows
    for i in range(n):
        result *= matrix[i][i]
    return result


def _swap_parity(steps: list[RowOperation]) -> int:
    """
    Compute the sign contribution of all row swaps in a step sequence.

    Each row swap flips the sign of the determinant. This function counts
    the number of RowSwap operations and returns 1 if that count is even,
    -1 if it is odd.

    Parameters
    ----------
    steps : list[RowOperation]
        The ordered row operations from an LU decomposition.

    Returns
    -------
    int
        1 if the number of row swaps is even, -1 if it is odd.
    """
    swap_count = sum(1 for step in steps if isinstance(step, RowSwap))
    return (-1) ** swap_count


def _inconsistent_rows(
    matrix_rref: Reduction, applied_b: Vector, tol: float = 0.0
) -> list[int]:
    """
    Find row indices that make the system inconsistent.

    A row is inconsistent if it is a zero row in the RREF of A but the
    corresponding entry in the transformed b is non-zero. This represents
    a contradiction of the form 0 = c where c ≠ 0.

    Parameters
    ----------
    matrix_rref : Reduction
        The RREF reduction of the coefficient matrix A.
    applied_b : Vector
        The right-hand side vector b after the same row operations from
        matrix_rref have been applied to it.
    tol : float, optional
        Entries of the transformed b with magnitude at or below this value
        are treated as zero. The default of 0.0 means exact comparison.

    Returns
    -------
    list[int]
        The row indices where the system is contradicted. Empty if the
        system is consistent.
    """
    pivot_row_indices = {row for row, _ in matrix_rref.pivots}
    zero_row_indices = set(range(matrix_rref.result.rows)) - pivot_row_indices
    inconsistent_indices = []
    for i in zero_row_indices:
        if abs(applied_b[i]) > tol:
            inconsistent_indices.append(i)

    return inconsistent_indices


def inverse(matrix: Matrix) -> InverseResult:
    """
    Compute the inverse of a square, invertible matrix.

    Reduces the matrix to RREF using Gauss-Jordan elimination and replays
    the recorded row operations on the identity matrix to construct A⁻¹.
    The matrix must be square and have full rank; otherwise it is singular
    and no inverse exists.

    Parameters
    ----------
    matrix : Matrix
        The matrix to invert. Must be square and have full rank.

    Returns
    -------
    InverseResult
        An object containing the original matrix, the computed inverse, and
        the sequence of row operations used.

    Raises
    ------
    TypeError
        If matrix is not a Matrix instance.
    ValueError
        If matrix is not square, or if matrix is singular (rank < n).

    Examples
    --------
    >>> m = Matrix([[1, 2], [3, 4]])
    >>> result = inverse(m)
    >>> print(result.inverse)
    [[-2.0, 1.0],
     [1.5, -0.5]]
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"Expected a Matrix, but got {type(matrix).__name__}. "
            f"Inverse is only defined for Matrix objects."
        )
    if not matrix.is_square:
        raise ValueError(
            f"Cannot compute the inverse of a non-square matrix. "
            f"Your matrix is {matrix.rows}×{matrix.cols}. "
            f"Inverse is only defined for square matrices (n×n)."
        )
    n = matrix.rows
    matrix_rref = rref(matrix)
    if matrix_rref.rank != n:
        raise ValueError(
            f"Cannot compute the inverse of a singular matrix. "
            f"Your matrix has rank {matrix_rref.rank}, but must have rank {n}. "
            f"Only matrices with full rank are invertible."
        )
    steps = matrix_rref.steps
    inv = _calculate_inverse(n, steps)
    return InverseResult(matrix, inv, steps)


def determinant_lu(matrix: Matrix) -> float:
    """
    Compute the determinant of a square matrix using LU decomposition.

    Factors the matrix into P, L, and U using partial pivoting, then
    multiplies the main diagonal entries of U by the parity of the
    permutation. Each row swap in P contributes a factor of -1 to the
    determinant, so the sign is adjusted by counting the number of swaps
    performed during factorization.

    Parameters
    ----------
    matrix : Matrix
        The matrix whose determinant will be computed. Must be square.

    Returns
    -------
    float
        The determinant of the matrix.

    Raises
    ------
    TypeError
        If matrix is not a Matrix instance.
    ValueError
        If matrix is not square.

    Examples
    --------
    >>> determinant_lu(Matrix([[1, 2], [3, 4]]))
    -2.0
    >>> determinant_lu(Matrix([[0, 1], [1, 2]]))
    -1.0

    See Also
    --------
    Matrix.determinant : Determinant via cofactor expansion.
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"Expected a Matrix, but got {type(matrix).__name__}. "
            f"Determinant is only defined for Matrix objects."
        )
    if not matrix.is_square:
        raise ValueError(
            f"Cannot compute the determinant of a non-square matrix. "
            f"Your matrix is {matrix.rows}×{matrix.cols}. "
            f"Determinants are only defined for square matrices (n×n)."
        )
    matrix_lu = lu(matrix)
    parity = _swap_parity(matrix_lu.steps)
    upper_diagonal_product = _main_diagonal_product(matrix_lu.upper)
    return parity * upper_diagonal_product


def solve(A: Matrix, b: Vector, tol: float = 0.0) -> Solution:
    """
    Solve the linear system Ax = b.

    Reduces A to RREF and applies the same row operations to b. The
    system's status is determined by inspecting the reduced forms: an
    inconsistent row (zero row in A with a non-zero corresponding entry
    in b) means no solution exists; fewer pivots than variables means
    infinitely many solutions exist; otherwise a unique solution is
    extracted from the pivot rows of the transformed b.

    Parameters
    ----------
    A : Matrix
        The coefficient matrix in the system Ax = b.
    b : Vector
        The right-hand side vector in the system Ax = b.
    tol : float, optional
        Pivot and right-hand-side magnitudes at or below this value are
        treated as zero during reduction. The default of 0.0 performs an
        exact solve. A small positive tolerance lets the solver treat a
        matrix that is only approximately rank-deficient as singular, which
        is how eigenvectors are recovered as the null space of A - λI for a
        floating-point eigenvalue estimate λ.

    Returns
    -------
    Solution
        An object containing the original matrix and vector, the status
        ('unique', 'infinite', or 'inconsistent'), the solution vector
        if unique, and the row operations applied.

    Raises
    ------
    TypeError
        If A is not a Matrix instance, or b is not a Vector instance.
    ValueError
        If the number of rows in A does not match the length of b.

    Examples
    --------
    >>> A = Matrix([[2, 1], [5, 3]])
    >>> b = Vector([1, 2])
    >>> result = solve(A, b)
    >>> result.status
    'unique'
    >>> result.solution
    [1.0, -1.0]
    """
    if not isinstance(A, Matrix):
        raise TypeError(
            f"Expected a Matrix for A, but got {type(A).__name__}. "
            f"Solve is only defined for Matrix objects."
        )
    if not isinstance(b, Vector):
        raise TypeError(
            f"Expected a Vector for b, but got {type(b).__name__}. "
            f"Solve is only defined for Vector objects."
        )
    if A.rows != b.dims:
        raise ValueError(
            f"The number of rows in A must match the length of b. "
            f"A has {A.rows} rows but b has {b.dims} entries."
        )

    matrix_rref = rref(A, tol)
    steps = matrix_rref.steps

    applied_b = b
    for step in steps:
        applied_b = step.apply(applied_b)

    if _inconsistent_rows(matrix_rref, applied_b, tol):
        return Solution(A, b, "inconsistent", None, steps)

    if matrix_rref.nullity > 0:
        from panchi.primitives.vector_space import VectorSpace

        n = A.cols
        pivot_cols = {col for _, col in matrix_rref.pivots}
        pivot_map = {col: row for row, col in matrix_rref.pivots}
        free_cols = [j for j in range(n) if j not in pivot_cols]

        null_vectors = []
        for fc in free_cols:
            components: list[Scalar] = [0] * n
            components[fc] = 1
            for pc, pr in pivot_map.items():
                components[pc] = -matrix_rref.result[pr][fc]
            null_vectors.append(Vector(components))

        particular_components: list[Scalar] = [0] * n
        for pc, pr in pivot_map.items():
            particular_components[pc] = applied_b[pr]
        particular = Vector(particular_components)

        null_space = VectorSpace(null_vectors)
        return Solution(A, b, "infinite", None, steps, particular, null_space)

    pivot_row_indices = [row for row, _ in sorted(matrix_rref.pivots, key=lambda p: p[1])]
    solution = Vector([applied_b[row] for row in pivot_row_indices])
    return Solution(A, b, "unique", solution, steps)


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
    solution = solve(shifted, zero_vector(n), tol=1e-6 * scale)

    if solution.null_space is None:
        return None
    return solution.null_space.basis[0].normalize()


def eigen(
    matrix: Matrix,
    max_iterations: int = 1000,
    tol: float = 1e-12,
) -> EigenResult:
    """
    Compute the eigenvalues of a square matrix using the QR algorithm.

    Starting from the original matrix, repeatedly computes a QR
    decomposition and reassembles the matrix as R @ Q. For matrices with
    real eigenvalues this sequence converges to an upper triangular matrix
    whose diagonal entries are the eigenvalues.

    The iteration stops once the strictly-below-diagonal entries sum to less
    than ``tol`` (recorded as converged), or once ``max_iterations`` is
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
    tol : float, optional
        The convergence threshold on the below-diagonal mass. Defaults to
        1e-12.

    Returns
    -------
    EigenResult
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
    >>> result = eigen(Matrix([[2, 1], [1, 2]]))
    >>> sorted(round(v, 6) for v in result.eigenvalues)
    [1.0, 3.0]
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(
            f"Expected a Matrix, but got {type(matrix).__name__}. "
            f"Eigenvalues are only defined for Matrix objects."
        )
    if not matrix.is_square:
        raise ValueError(
            f"Cannot compute the eigenvalues of a non-square matrix. "
            f"Your matrix is {matrix.rows}×{matrix.cols}. "
            f"Eigenvalues are only defined for square matrices (n×n)."
        )

    n = matrix.rows
    current = matrix
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if _below_diagonal_mass(current) < tol:
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
    return EigenResult(matrix, eigenvalues, eigenvectors, iterations, converged, current)
