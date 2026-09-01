"""
panchi - A Python-native linear algebra library for learning and experimentation.

panchi prioritizes clarity and understanding over performance, making it ideal
for students, educators, and anyone who wants to see how linear algebra really works.
"""

__version__ = "2.0.0"

from fractions import Fraction

from panchi.algorithms.decompositions import qr_decomposition
from panchi.algorithms.matrix_operations import (
    determinant,
    determinant_lu,
    eigen,
    inverse,
    is_invertible,
    is_symmetric,
    nullity,
    rank,
    solve,
)
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
from panchi.primitives.factories import (
    diagonal,
    exact_matrix,
    exact_vector,
    identity,
    one_matrix,
    one_vector,
    random_matrix,
    random_vector,
    rotation_matrix_2d,
    rotation_matrix_3d,
    unit_vector,
    vector_column_matrix,
    zero_matrix,
    zero_vector,
)
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.types import Scalar

__all__ = [
    "Fraction",
    "Scalar",
    "Vector",
    "Matrix",
    "VectorSpace",
    "identity",
    "zero_matrix",
    "one_matrix",
    "zero_vector",
    "one_vector",
    "unit_vector",
    "diagonal",
    "random_vector",
    "random_matrix",
    "rotation_matrix_2d",
    "rotation_matrix_3d",
    "vector_column_matrix",
    "exact_vector",
    "exact_matrix",
    "dot",
    "cross",
    "vector_projection",
    "gram_schmidt",
    "basis",
    "rank",
    "nullity",
    "is_invertible",
    "is_symmetric",
    "is_full_rank",
    "contains",
    "same_subspace",
    "orthogonal_complement",
    "span",
    "standard_basis",
    "column_space",
    "row_space",
    "null_space",
    "inverse",
    "solve",
    "determinant",
    "determinant_lu",
    "qr_decomposition",
    "eigen",
]
