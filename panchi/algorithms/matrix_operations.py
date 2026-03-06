from panchi.primitives.matrix import Matrix
from panchi.primitives.factories import identity
from panchi.algorithms.row_operations import RowOperation, RowSwap
from panchi.algorithms.results import InverseResult
from panchi.algorithms.reductions import rref
from panchi.algorithms.decompositions import lu


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


def _main_diagonal_product(matrix: Matrix) -> float:
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
    return parity * _main_diagonal_product(matrix_lu.upper)
