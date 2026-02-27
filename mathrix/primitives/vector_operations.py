from __future__ import annotations

from mathrix.primitives.vector import Vector


def dot(vector_1: Vector, vector_2: Vector) -> float:
    """
    Compute the dot product (inner product) of two vectors.

    The dot product is the sum of the products of corresponding components.
    It measures the extent to which two vectors point in the same direction.

    Parameters
    ----------
    vector_1 : Vector
        The first vector.
    vector_2 : Vector
        The second vector.

    Returns
    -------
    float
        The dot product of the two vectors.

    Raises
    ------
    ValueError
        If the vectors have different dimensions.

    Examples
    --------
    >>> v1 = Vector([1, 2, 3])
    >>> v2 = Vector([4, 5, 6])
    >>> dot(v1, v2)
    32
    """
    if vector_1.dims != vector_2.dims:
        raise ValueError(
            f"Vector dimensions must match for dot product. Got vector_1: {vector_1.dims}, vector_2: {vector_2.dims}."
        )

    n = vector_1.dims
    return sum(vector_1[i] * vector_2[i] for i in range(n))


def cross(vector_1: Vector, vector_2: Vector) -> Vector:
    """
    Compute the cross product of two 3D vectors.

    The cross product produces a vector perpendicular to both input vectors.
    Its magnitude equals the area of the parallelogram formed by the vectors.

    Parameters
    ----------
    vector_1 : Vector
        The first 3D vector.
    vector_2 : Vector
        The second 3D vector.

    Returns
    -------
    Vector
        The cross product vector, perpendicular to both inputs.

    Raises
    ------
    ValueError
        If either vector is not 3-dimensional.

    Examples
    --------
    >>> v1 = Vector([1, 0, 0])
    >>> v2 = Vector([0, 1, 0])
    >>> v3 = cross(v1, v2)
    >>> print(v3)
    [0, 0, 1]
    """
    if not (vector_1.dims == 3 and vector_2.dims == 3):
        raise ValueError(
            f"Both vectors must be 3D. Got vector_1: {vector_1.dims}, vector_2: {vector_2.dims}."
        )

    x = (vector_1[1] * vector_2[2]) - (vector_1[2] * vector_2[1])
    y = (vector_1[2] * vector_2[0]) - (vector_1[0] * vector_2[2])
    z = (vector_1[0] * vector_2[1]) - (vector_1[1] * vector_2[0])

    return Vector([x, y, z])
