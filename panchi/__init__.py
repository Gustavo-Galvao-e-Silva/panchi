"""
panchi - A Python-native linear algebra library for learning and experimentation.

panchi prioritizes clarity and understanding over performance, making it ideal
for students, educators, and anyone who wants to see how linear algebra really works.
"""

__version__ = "0.1.0a1"

from panchi.primitives.vector import Vector
from panchi.primitives.matrix import Matrix
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
from panchi.primitives.vector_operations import (
    dot,
    cross,
)

__all__ = [
    "Vector",
    "Matrix",
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
    "dot",
    "cross",
]
