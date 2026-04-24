"""
Primitive linear algebra objects: Vector, Matrix, and factory functions.
"""

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
    vector_column_matrix,
)

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
]
