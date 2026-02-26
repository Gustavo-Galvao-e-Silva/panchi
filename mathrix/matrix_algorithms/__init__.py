"""
Matrix algorithms: row operations and row reductions.

This package provides the building blocks for operating on and reducing
matrices. Row operations represent the three elementary transformations
used in Gaussian elimination, and reductions apply those operations
systematically to bring a matrix into REF or RREF.
"""

from mathrix.matrix_algorithms.row_operations import RowSwap, RowScale, RowAdd
from mathrix.matrix_algorithms.reductions import Reduction, ref, rref

__all__ = [
    "RowSwap",
    "RowScale",
    "RowAdd",
    "Reduction",
    "ref",
    "rref",
]
