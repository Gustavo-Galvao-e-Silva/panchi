from panchi.primitives.matrix import Matrix
from panchi.primitives.factories import identity
from panchi.algorithms.row_operations import RowOperation
from panchi.algorithms.results import InverseResult
from panchi.algorithms.reductions import rref
from panchi.algorithms.decompositions import lu


def _calculate_inverse(n: int, steps: list[RowOperation]) -> Matrix:
    result = identity(n)
    for step in steps:
        result = step.apply(result)

    return result

def _main_diagonal_product(matrix: Matrix) -> float:
    result = 1
    n = matrix.rows
    for i in range(n):
        result *= matrix[i][i]

    return result

def inverse(matrix: Matrix) -> InverseResult:
    if not isinstance(matrix, Matrix):
        raise TypeError(f"Inverse calculation expects a matrix. Got: {type(matrix).__name__}")

    if not matrix.is_square:
        raise ValueError("")

    n = matrix.rows
    matrix_rref = rref(matrix)
    if matrix_rref.rank != n:
        raise ValueError("")

    steps = matrix_rref.steps
    inv = _calculate_inverse(n, steps)

    return InverseResult(matrix, inv, steps)


def determinant_lu(matrix: Matrix) -> float:
    if not isinstance(matrix, Matrix):
        raise TypeError(f"Determinant calculation expects a matrix. Got: {type(matrix).__name__}")

    if not matrix.is_square:
        raise ValueError("")

    matrix_lu = lu(matrix)
    u = matrix_lu.upper
    return _main_diagonal_product(u)
