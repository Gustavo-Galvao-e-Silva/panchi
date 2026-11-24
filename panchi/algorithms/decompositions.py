from __future__ import annotations

import math

from panchi.algorithms.jacobi import _columns_to_matrix, _one_sided_jacobi
from panchi.algorithms.reductions import ref
from panchi.algorithms.results import LUDecomposition, QRDecomposition, SVDResult
from panchi.algorithms.row_operations import RowAdd, RowOperation, RowSwap
from panchi.algorithms.vector_operations import _gram_schmidt_steps
from panchi.primitives.factories import identity, vector_column_matrix
from panchi.primitives.matrix import Matrix


def _calculate_l(n: int, steps: list[RowOperation]) -> Matrix:
    """
    Build the lower triangular matrix L from a sequence of row operations.

    Constructs L by starting from the n×n identity and placing the negated
    scalar of each RowAdd operation at position [target, source]. This
    encodes the elimination multipliers used during Gaussian elimination,
    which is the standard way to read L from the steps of REF.

    A RowSwap that happens after elimination has started also swaps the
    multipliers already recorded for those two rows, so each multiplier
    stays with the row it was computed for. The swap itself is captured
    in P.

    Parameters
    ----------
    n : int
        Size of the square matrix L to construct.
    steps : list[RowOperation]
        The ordered sequence of row operations produced by REF.

    Returns
    -------
    Matrix
        An n×n lower triangular matrix with ones on the diagonal.
    """
    l = identity(n)
    for step in steps:
        if isinstance(step, RowSwap):
            for column in range(min(step.a, step.b)):
                multiplier_a = l[step.a, column]
                l[step.a, column] = l[step.b, column]
                l[step.b, column] = multiplier_a
        elif isinstance(step, RowAdd):
            l[step.target, step.source] = -step.scalar

    return l


def _calculate_p(n: int, steps: list[RowOperation]) -> Matrix:
    """
    Build the permutation matrix P from a sequence of row operations.

    Constructs P by applying each RowSwap in the step sequence to the
    n×n identity matrix in order. The result encodes all row swaps made
    during elimination, satisfying P @ A == L @ U.

    RowAdd and RowScale operations are ignored here — only swaps affect P.

    Parameters
    ----------
    n : int
        Size of the square permutation matrix to construct.
    steps : list[RowOperation]
        The ordered sequence of row operations produced by REF.

    Returns
    -------
    Matrix
        An n×n permutation matrix.
    """
    p = identity(n)
    for step in steps:
        if isinstance(step, RowSwap):
            p = step.apply(p)

    return p


def lu(matrix: Matrix) -> LUDecomposition:
    """
    Compute the LU decomposition of a square matrix, swapping rows as needed.

    Factors the matrix into a lower triangular matrix L, an upper triangular
    matrix U, and a permutation matrix P such that P @ matrix == L @ U.

    A row swap happens only when the current pivot is zero, using the first
    row below with a non-zero entry in the pivot column. This is not partial
    pivoting: a small non-zero pivot is kept. The swaps are recorded in P so
    the factorisation relationship holds exactly.

    L is lower triangular with ones on the diagonal. Its off-diagonal
    entries are the elimination multipliers used during Gaussian elimination.
    U is the row echelon form of P @ matrix.

    Parameters
    ----------
    matrix : Matrix
        The square matrix to decompose.

    Returns
    -------
    LUDecomposition
        A result object containing the original matrix, L, U, P, and the
        ordered list of row operations applied during elimination.

    Examples
    --------
    >>> A = Matrix([[2, 1], [4, 3]])
    >>> decomp = lu(A)
    >>> decomp.permutation @ A == decomp.lower @ decomp.upper
    True
    >>> print(decomp.lower)
    [[1, 0],
     [2.0, 1]]
    >>> print(decomp.upper)
    [[2, 1],
     [0.0, 1.0]]
    """
    matrix_ref = ref(matrix)
    n = matrix.rows
    steps = matrix_ref.steps
    l = _calculate_l(n, steps)
    u = matrix_ref.result
    p = _calculate_p(n, steps)
    return LUDecomposition(matrix, l, u, p, steps)


def qr_decomposition(matrix: Matrix) -> QRDecomposition:
    """
    Compute the (thin) QR decomposition of a matrix.

    Factors the matrix into a matrix Q with orthonormal columns and an
    upper triangular matrix R such that matrix == Q @ R. Q is built by
    applying Gram-Schmidt to the columns of the matrix, and R is recovered
    as Q.T @ matrix.

    This is the reduced ("thin") QR decomposition and assumes the columns
    of the matrix are linearly independent. Dependent columns produce a zero
    orthogonal vector during Gram-Schmidt, which cannot be normalized and
    raises ZeroDivisionError.

    Parameters
    ----------
    matrix : Matrix
        The matrix to decompose. Its columns are expected to be linearly
        independent.

    Returns
    -------
    QRDecomposition
        A result object containing the original matrix, Q, R, and the
        ordered list of Gram-Schmidt steps used to build Q.

    Raises
    ------
    ZeroDivisionError
        If the columns of the matrix are linearly dependent.

    Examples
    --------
    >>> A = Matrix([[1, 0], [0, 1]])
    >>> decomp = qr_decomposition(A)
    >>> decomp.q @ decomp.r == A
    True
    """
    steps = _gram_schmidt_steps(matrix.col_vectors)
    orthonormal_vectors = [step.orthonormal for step in steps]
    q = vector_column_matrix(orthonormal_vectors)
    r = q.T @ matrix
    return QRDecomposition(matrix, q, r, steps)


_RECORD_LEVELS = ("none", "sweeps", "rotations")


def _float_columns(matrix: Matrix) -> list[list[float]]:
    """
    Convert a matrix to a list of float columns, rejecting NaN and infinity.
    """
    columns: list[list[float]] = [[] for _ in range(matrix.cols)]
    for i, row in enumerate(matrix.to_list()):
        for j, entry in enumerate(row):
            try:
                value = float(entry)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"svd() needs entries that convert to float. The entry at "
                    f"row {i}, column {j} is {entry!r} "
                    f"({type(entry).__name__})."
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"svd() needs finite entries. The entry at row {i}, "
                    f"column {j} is {value}."
                )
            columns[j].append(value)
    return columns


def _complete_orthonormal(
    columns: list[list[float]], size: int, count: int
) -> list[list[float]]:
    """
    Return count unit vectors that extend the orthonormal columns, built by
    Gram-Schmidt on the standard basis vectors e_0, e_1, ... of R^size.
    """
    basis = [list(column) for column in columns]
    added: list[list[float]] = []
    for index in range(size):
        if len(added) == count:
            break
        candidate = [1.0 if i == index else 0.0 for i in range(size)]
        for _ in range(2):
            for column in basis:
                projection = sum(a * b for a, b in zip(candidate, column, strict=True))
                candidate = [
                    a - projection * b for a, b in zip(candidate, column, strict=True)
                ]
        length = math.sqrt(sum(a * a for a in candidate))
        if length > 1e-8:
            unit = [a / length for a in candidate]
            basis.append(unit)
            added.append(unit)
    return added


def svd(
    matrix: Matrix,
    *,
    tolerance: float = 1e-15,
    max_sweeps: int = 60,
    record: str = "sweeps",
) -> SVDResult:
    """
    Compute the (thin) singular value decomposition A = U Σ Vᵀ.

    Uses one-sided Jacobi: pairs of columns of A are rotated until every
    pair is orthogonal. The rotations, collected into V, give A V = W with
    orthogonal columns wᵢ. Each singular value is σᵢ = ‖wᵢ‖ and each left
    singular vector is uᵢ = wᵢ / σᵢ. For a wide matrix (m < n) the algorithm
    runs on Aᵀ and swaps U and V.

    With k = min(m, n), U is m×k with orthonormal columns, Σ is the k×k
    diagonal matrix of singular values in descending order, and V is n×k
    with orthonormal columns. The first entry of each column of V with
    absolute value above 1e-12 is positive. Columns of U for zero singular
    values are completed so that Uᵀ U = I always holds.

    Parameters
    ----------
    matrix : Matrix
        The matrix to decompose. Entries are converted with float().
    tolerance : float, optional
        A column pair is rotated while |cos(angle between columns)| exceeds
        this value. Also sets the numerical rank threshold. Defaults to 1e-15.
    max_sweeps : int, optional
        The maximum number of sweeps over all column pairs. Defaults to 60.
    record : str, optional
        How much history to keep in ``steps``: "none", "sweeps" (one summary
        per sweep, the default) or "rotations" (every rotation, plus the
        working matrix after each sweep).

    Returns
    -------
    SVDResult
        The factors U, σ, V, the sweep history and helpers for rank-k
        approximation. If the sweeps do not converge, ``converged`` is False
        and the best factors found are returned.

    Raises
    ------
    TypeError
        If matrix is not a Matrix, or an entry cannot be converted to float.
    ValueError
        If an entry is NaN or infinite, record is not a valid level,
        max_sweeps < 1, or tolerance <= 0.

    See Also
    --------
    qr_decomposition : Factor a matrix into orthonormal Q and triangular R.
    eigen_qr : Numerical eigenvalues via the QR algorithm.

    Examples
    --------
    >>> result = svd(Matrix([[1, 1], [1, 1]]))
    >>> [round(value, 10) for value in result.sigma]
    [2.0, 0.0]
    >>> result.rank
    1
    >>> product = result.u @ result.s @ result.vt
    >>> [[round(x, 10) for x in row] for row in product.to_list()]
    [[1.0, 1.0], [1.0, 1.0]]
    """
    if not isinstance(matrix, Matrix):
        raise TypeError(f"svd() requires a Matrix. Got {type(matrix).__name__}.")
    if record not in _RECORD_LEVELS:
        raise ValueError(
            f"record must be one of {', '.join(repr(r) for r in _RECORD_LEVELS)}. "
            f"Got {record!r}."
        )
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int):
        raise TypeError(f"max_sweeps must be an int. Got {type(max_sweeps).__name__}.")
    if max_sweeps < 1:
        raise ValueError(f"max_sweeps must be at least 1. Got {max_sweeps}.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError(f"tolerance must be a number. Got {type(tolerance).__name__}.")
    if not tolerance > 0:
        raise ValueError(f"tolerance must be positive. Got {tolerance}.")

    columns = _float_columns(matrix)
    transposed = matrix.rows < matrix.cols
    if transposed:
        columns = [list(row) for row in zip(*columns, strict=True)]

    work, v_columns, sweeps, converged, steps = _one_sided_jacobi(
        columns, tolerance, max_sweeps, record
    )

    norms = [math.sqrt(sum(x * x for x in column)) for column in work]
    order = sorted(range(len(work)), key=lambda j: norms[j], reverse=True)
    sigma = [norms[j] for j in order]
    rows = len(work[0])
    threshold = tolerance * sigma[0] * max(matrix.rows, matrix.cols)

    u_columns: list[list[float]] = []
    for position, j in enumerate(order):
        if sigma[position] > threshold:
            u_columns.append([x / sigma[position] for x in work[j]])
    u_columns += _complete_orthonormal(u_columns, rows, len(order) - len(u_columns))
    v_sorted = [v_columns[j] for j in order]

    if transposed:
        u_columns, v_sorted = v_sorted, u_columns

    for position, v_column in enumerate(v_sorted):
        leading = next((x for x in v_column if abs(x) > 1e-12), 0.0)
        if leading < 0:
            v_sorted[position] = [-x for x in v_column]
            u_columns[position] = [-x for x in u_columns[position]]

    u = _columns_to_matrix(u_columns)
    v = _columns_to_matrix(v_sorted)
    return SVDResult(matrix, u, sigma, v, sweeps, converged, steps, tolerance)
