from __future__ import annotations

from panchi.primitives.matrix import Matrix
from panchi.primitives.factories import identity
from panchi.algorithms.reductions import ref
from panchi.algorithms.row_operations import RowOperation, RowAdd, RowSwap
from panchi.algorithms.results import LUDecomposition


def _calculate_l(n: int, steps: list[RowOperation]) -> Matrix:
    l = identity(n)
    for step in steps:
        if isinstance(step, RowAdd):
            l[step.target][step.source] = -step.scalar

    return l


def _calculate_p(n: int, steps: list[RowOperation]) -> Matrix:
    p = identity(n)
    for step in steps:
        if isinstance(step, RowSwap):
            p = step.apply(p)

    return p


def lu(matrix: Matrix) -> LUDecomposition:
    matrix_ref = ref(matrix)
    n = matrix.rows
    steps = matrix_ref.steps
    l = _calculate_l(n, steps)
    u = matrix_ref.result
    p = _calculate_p(n, steps)
    return LUDecomposition(matrix, l, u, p, steps)
