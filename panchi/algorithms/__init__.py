"""
Matrix algorithms: row operations, reductions, decompositions, and solvers.

This package provides the building blocks for operating on and reducing
matrices. Row operations represent the three elementary transformations
used in Gaussian elimination, and reductions apply those operations
systematically to bring a matrix into REF or RREF.

Decompositions factor matrices into structured components, and solvers
find solutions to linear systems by reducing augmented matrices.
"""

from panchi.algorithms.row_operations import RowSwap, RowScale, RowAdd
from panchi.algorithms.reductions import Reduction, ref, rref
from panchi.algorithms.decompositions import lu, qr_decomposition
from panchi.algorithms.results import (
    LUDecomposition,
    QRDecomposition,
    InverseResult,
    Solution,
    EigenResult,
)
from panchi.algorithms.matrix_operations import (
    inverse,
    solve,
    determinant,
    determinant_lu,
    eigen,
)
from panchi.algorithms.vector_operations import (
    dot,
    cross,
    vector_projection,
    gram_schmidt,
)
from panchi.algorithms.vector_space_operations import (
    basis,
    rank,
    is_full_rank,
    contains,
    same_subspace,
    orthogonal_complement,
)

__all__ = [
    "RowSwap",
    "RowScale",
    "RowAdd",
    "Reduction",
    "ref",
    "rref",
    "lu",
    "qr_decomposition",
    "LUDecomposition",
    "QRDecomposition",
    "InverseResult",
    "Solution",
    "EigenResult",
    "inverse",
    "solve",
    "determinant",
    "determinant_lu",
    "eigen",
    "dot",
    "cross",
    "vector_projection",
    "gram_schmidt",
    "basis",
    "rank",
    "is_full_rank",
    "contains",
    "same_subspace",
    "orthogonal_complement",
]
