from __future__ import annotations

from mathrix.primitives.matrix import Matrix
from mathrix.algorithms.results import Reduction
from mathrix.algorithms.row_operations import (
    RowOperation,
    RowAdd,
    RowScale,
    RowSwap,
)


def _find_first_non_zero_row(
    starting_row_num: int, col_num: int, matrix: Matrix
) -> int | None:
    """
    Find the first row at or below starting_row_num with a non-zero entry in col_num.

    Parameters
    ----------
    starting_row_num : int
        The row index to begin searching from (inclusive).
    col_num : int
        The column index to inspect in each row.
    matrix : Matrix
        The matrix to search.

    Returns
    -------
    int or None
        The index of the first row with a non-zero entry in col_num,
        or None if no such row exists.
    """
    for i in range(starting_row_num, matrix.rows):
        if matrix[i][col_num] != 0:
            return i

    return None


def _swap_pivot(
    pivot_row: int, pivot_col: int, matrix: Matrix
) -> tuple[Matrix, list[RowOperation]]:
    """
    Swap the pivot row with the first row below it that has a non-zero entry in pivot_col.

    If the entry at [pivot_row][pivot_col] is already non-zero, no swap is performed
    and the matrix is returned unchanged.

    Parameters
    ----------
    pivot_row : int
        The row index of the current pivot position.
    pivot_col : int
        The column index of the current pivot position.
    matrix : Matrix
        The matrix to operate on.

    Returns
    -------
    tuple[Matrix, list[RowOperation]]
        The updated matrix and a list containing the RowSwap applied, or an
        empty list if no swap was needed.
    """
    result = matrix.copy()
    new_operations = []
    if result[pivot_row][pivot_col] == 0:
        target_row = _find_first_non_zero_row(pivot_row + 1, pivot_col, result)
        if target_row is not None:
            op = RowSwap(pivot_row, target_row)
            result = op.apply(result)
            new_operations.append(op)

    return result, new_operations


def _add_below_pivot(
    pivot_row: int, pivot_col: int, matrix: Matrix
) -> tuple[Matrix, list[RowOperation]]:
    """
    Eliminate all non-zero entries below the pivot using row addition.

    For each row below pivot_row with a non-zero entry in pivot_col, adds
    a scalar multiple of pivot_row to that row so the entry becomes zero.
    This is the forward elimination step of Gaussian elimination.

    Parameters
    ----------
    pivot_row : int
        The row index of the current pivot.
    pivot_col : int
        The column index of the current pivot.
    matrix : Matrix
        The matrix to operate on. The entry at [pivot_row][pivot_col] must
        be non-zero.

    Returns
    -------
    tuple[Matrix, list[RowOperation]]
        The updated matrix and the list of RowAdd operations applied.
    """
    pivot = matrix[pivot_row][pivot_col]
    result = matrix.copy()
    new_operations = []
    for i in range(pivot_row + 1, result.rows):
        val = result[i][pivot_col]
        if val == 0:
            continue
        op = RowAdd(i, pivot_row, -(val / pivot))
        result = op.apply(result)
        new_operations.append(op)

    return result, new_operations


def _add_above_pivot(
    pivot_row: int, pivot_col: int, matrix: Matrix
) -> tuple[Matrix, list[RowOperation]]:
    """
    Eliminate all non-zero entries above the pivot using row addition.

    For each row above pivot_row with a non-zero entry in pivot_col, adds
    a scalar multiple of pivot_row to that row so the entry becomes zero.
    This is the back-substitution step of Gauss-Jordan elimination. The
    pivot at [pivot_row][pivot_col] is assumed to equal 1 before this
    function is called.

    Parameters
    ----------
    pivot_row : int
        The row index of the current pivot.
    pivot_col : int
        The column index of the current pivot.
    matrix : Matrix
        The matrix to operate on. The entry at [pivot_row][pivot_col] must
        equal 1.

    Returns
    -------
    tuple[Matrix, list[RowOperation]]
        The updated matrix and the list of RowAdd operations applied.
    """
    pivot = matrix[pivot_row][pivot_col]
    result = matrix.copy()
    new_operations = []
    for i in range(pivot_row - 1, -1, -1):
        val = result[i][pivot_col]
        if val == 0:
            continue
        op = RowAdd(i, pivot_row, -val)
        result = op.apply(result)
        new_operations.append(op)

    return result, new_operations


def _scale_pivot(
    pivot_row: int, pivot_col: int, matrix: Matrix
) -> tuple[Matrix, list[RowOperation]]:
    """
    Scale the pivot row so that the pivot entry equals 1.

    If the pivot is already 1, no operation is applied and the matrix is
    returned unchanged.

    Parameters
    ----------
    pivot_row : int
        The row index of the current pivot.
    pivot_col : int
        The column index of the current pivot.
    matrix : Matrix
        The matrix to operate on. The entry at [pivot_row][pivot_col] must
        be non-zero.

    Returns
    -------
    tuple[Matrix, list[RowOperation]]
        The updated matrix and a list containing the RowScale applied, or an
        empty list if no scaling was needed.
    """
    pivot = matrix[pivot_row][pivot_col]
    result = matrix.copy()
    new_operations = []
    if pivot != 1:
        op = RowScale(pivot_row, (1 / pivot))
        result = op.apply(result)
        new_operations.append(op)

    return result, new_operations


def ref(matrix: Matrix) -> Reduction:
    """
    Reduce a matrix to row echelon form using Gaussian elimination.

    Applies a sequence of elementary row operations to produce an upper
    triangular form where each pivot is to the right of the pivot in the
    row above it, and all entries below each pivot are zero. The pivot
    values are not normalised to 1.

    Parameters
    ----------
    matrix : Matrix
        The matrix to reduce. Not modified by this function.

    Returns
    -------
    Reduction
        A Reduction object containing the original matrix, the REF result,
        the ordered list of row operations applied, the pivot positions as
        (row, col) tuples, and the form label 'REF'.

    Examples
    --------
    >>> m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
    >>> reduction = ref(m)
    >>> print(reduction.result)
    [[1, 2, 3],
     [0, 1, 1],
     [0, 0, 1]]
    >>> reduction.rank
    3
    >>> reduction.pivots
    [(0, 0), (1, 1), (2, 2)]
    """
    result = matrix.copy()
    operations = []
    pivots = []
    i = 0
    for j in range(min(matrix.cols, matrix.rows)):
        result, swap_operations = _swap_pivot(i, j, result)
        operations += swap_operations
        if result[i][j] == 0:
            continue

        result, addition_operations = _add_below_pivot(i, j, result)
        operations += addition_operations
        pivots.append((i, j))
        i += 1

    return Reduction(matrix, result, operations, pivots, "REF")


def rref(matrix: Matrix) -> Reduction:
    """
    Reduce a matrix to reduced row echelon form using Gauss-Jordan elimination.

    First reduces to REF via Gaussian elimination, then applies back-substitution
    to clear all entries above each pivot and scales each pivot row so that the
    pivot value equals 1. The result is unique for any given matrix.

    Parameters
    ----------
    matrix : Matrix
        The matrix to reduce. Not modified by this function.

    Returns
    -------
    Reduction
        A Reduction object containing the original matrix, the RREF result,
        the complete ordered list of row operations applied (including those
        from the initial REF step), the pivot positions as (row, col) tuples,
        and the form label 'RREF'.

    Examples
    --------
    >>> m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
    >>> reduction = rref(m)
    >>> print(reduction.result)
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    >>> reduction.rank
    3
    >>> reduction.nullity
    0
    """
    gaussian_step = ref(matrix)
    result = gaussian_step.result
    operations = gaussian_step.steps
    pivots = gaussian_step.pivots
    for i, j in pivots:
        result, scale_operations = _scale_pivot(i, j, result)
        operations += scale_operations
        result, addition_operations = _add_above_pivot(i, j, result)
        operations += addition_operations

    return Reduction(matrix, result, operations, pivots, "RREF")
