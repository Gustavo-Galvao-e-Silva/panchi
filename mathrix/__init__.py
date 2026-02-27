"""
Mathrix - A Python-native linear algebra library for learning and experimentation.

Mathrix prioritizes clarity and understanding over performance, making it ideal
for students, educators, and anyone who wants to see how linear algebra really works.
"""

__version__ = "0.1.0"

from mathrix.primitives.vector import Vector
from mathrix.primitives.matrix import Matrix
from mathrix.primitives.factories import (
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
from mathrix.primitives.vector_operations import (
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
