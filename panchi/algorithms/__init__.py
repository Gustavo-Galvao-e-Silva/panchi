"""
Matrix algorithms: row operations, reductions, decompositions, and solvers.

This package provides the building blocks for operating on and reducing
matrices. Row operations represent the three elementary transformations
used in Gaussian elimination, and reductions apply those operations
systematically to bring a matrix into REF or RREF.

Decompositions factor matrices into structured components, and solvers
find solutions to linear systems by reducing augmented matrices.
"""

from panchi.algorithms.decompositions import lu, qr_decomposition
from panchi.algorithms.eigen import characteristic_polynomial, eigen, eigen_qr
from panchi.algorithms.matrix_operations import (
    determinant,
    determinant_lu,
    inverse,
    is_invertible,
    is_symmetric,
    nullity,
    rank,
    solve,
)
from panchi.algorithms.reductions import Reduction, ref, rref
from panchi.algorithms.results import (
    EigenResult,
    InverseResult,
    LUDecomposition,
    QRDecomposition,
    QREigenResult,
    Solution,
)
from panchi.algorithms.row_operations import RowAdd, RowScale, RowSwap
from panchi.algorithms.vector_operations import (
    cross,
    dot,
    gram_schmidt,
    vector_projection,
)
from panchi.algorithms.vector_space_operations import (
    basis,
    column_space,
    contains,
    is_full_rank,
    null_space,
    orthogonal_complement,
    row_space,
    same_subspace,
    span,
    standard_basis,
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
    "QREigenResult",
    "EigenResult",
    "inverse",
    "solve",
    "determinant",
    "determinant_lu",
    "eigen_qr",
    "eigen",
    "characteristic_polynomial",
    "rank",
    "nullity",
    "is_invertible",
    "is_symmetric",
    "dot",
    "cross",
    "vector_projection",
    "gram_schmidt",
    "basis",
    "is_full_rank",
    "contains",
    "same_subspace",
    "orthogonal_complement",
    "span",
    "standard_basis",
    "column_space",
    "row_space",
    "null_space",
]
