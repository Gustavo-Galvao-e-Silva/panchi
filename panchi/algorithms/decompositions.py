from __future__ import annotations

from panchi.primitives.matrix import Matrix
from panchi.primitives.factories import identity
from panchi.algorithms.reductions import ref
from panchi.algorithms.row_operations import RowOperation, RowAdd, RowSwap
from panchi.algorithms.results import LUDecomposition


def _calculate_l(n: int, steps: list[RowOperation]) -> Matrix:
    """
    Build the lower triangular matrix L from a sequence of row operations.

    Constructs L by starting from the n×n identity and placing the negated
    scalar of each RowAdd operation at position [target, source]. This
    encodes the elimination multipliers used during Gaussian elimination,
    which is the standard way to read L from the steps of REF.

    RowSwap operations are ignored here — they are captured in P instead.

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
        if isinstance(step, RowAdd):
            l[step.target][step.source] = -step.scalar

    return l


def _calculate_p(n: int, steps: list[RowOperation]) -> Matrix:
    """
    Build the permutation matrix P from a sequence of row operations.

    Constructs P by applying each RowSwap in the step sequence to the
    n×n identity matrix in order. The result encodes all row swaps made
    during partial pivoting, satisfying P @ A == L @ U.

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
    Compute the LU decomposition of a square matrix with partial pivoting.

    Factors the matrix into a lower triangular matrix L, an upper triangular
    matrix U, and a permutation matrix P such that P @ matrix == L @ U.

    Partial pivoting swaps rows before each elimination step to place the
    largest available entry in the pivot column at the pivot position. This
    improves numerical stability and avoids division by zero or near-zero
    values. The swaps are recorded in P so the factorisation relationship
    holds exactly.

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
