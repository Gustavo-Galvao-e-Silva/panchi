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
from panchi.algorithms.matrix_operations import determinant_lu, eigen, inverse, solve
from panchi.algorithms.reductions import Reduction, ref, rref
from panchi.algorithms.results import (
    EigenResult,
    InverseResult,
    LUDecomposition,
    QRDecomposition,
    Solution,
)
from panchi.algorithms.row_operations import RowAdd, RowScale, RowSwap
from panchi.algorithms.vector_operations import (
    cross,
    dot,
    gram_schmidt,
    orthogonal_complement,
    vector_projection,
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
    "determinant_lu",
    "eigen",
    "dot",
    "cross",
    "orthogonal_complement",
    "vector_projection",
    "gram_schmidt",
]
