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
from panchi.algorithms.decompositions import lu
from panchi.algorithms.results import LUDecomposition, InverseResult, Solution

__all__ = [
    "RowSwap",
    "RowScale",
    "RowAdd",
    "Reduction",
    "ref",
    "rref",
    "lu",
    "LUDecomposition",
    "InverseResult",
    "Solution",
]
