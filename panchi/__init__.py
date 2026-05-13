"""
panchi - A Python-native linear algebra library for learning and experimentation.

panchi prioritizes clarity and understanding over performance, making it ideal
for students, educators, and anyone who wants to see how linear algebra really works.
"""

__version__ = "0.3.0b1"

from panchi.primitives.vector import Vector
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector_space import VectorSpace
from panchi.primitives.factories import (
    identity,
    zero_matrix,
    one_matrix,
    zero_vector,
    one_vector,
    unit_vector,
    diagonal,
    random_vector,
    random_matrix,
    rotation_matrix_2d,
    rotation_matrix_3d,
)
from panchi.primitives.factories import vector_column_matrix
from panchi.algorithms.vector_operations import dot, cross, orthogonal_complement
from panchi.algorithms.matrix_operations import inverse, solve, determinant_lu

__all__ = [
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
    "dot",
    "cross",
    "orthogonal_complement",
    "inverse",
    "solve",
    "determinant_lu",
]
