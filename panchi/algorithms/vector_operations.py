from __future__ import annotations

from panchi.types import Scalar
from panchi.primitives.vector import Vector


def dot(vector_1: Vector, vector_2: Vector) -> Scalar:
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
    int | float | Fraction
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


def orthogonal_complement(space: VectorSpace) -> VectorSpace:
    """
    Compute the orthogonal complement of a vector space.

    The orthogonal complement of a subspace W of R^n is the set of all
    vectors in R^n that are orthogonal to every vector in W. It is computed
    as the null space of the matrix whose rows are the basis vectors of W.

    Parameters
    ----------
    space : VectorSpace
        The subspace whose orthogonal complement is to be computed.

    Returns
    -------
    VectorSpace
        A VectorSpace representing the orthogonal complement. If the input
        space spans all of R^n, returns a VectorSpace containing only the
        zero vector (dimension 0).

    Raises
    ------
    TypeError
        If space is not a VectorSpace.

    Examples
    --------
    >>> v1 = Vector([1, 0, 0])
    >>> v2 = Vector([0, 1, 0])
    >>> vs = VectorSpace([v1, v2])
    >>> comp = orthogonal_complement(vs)
    >>> comp.dims
    1
    >>> comp.basis[0]
    Vector([0, 0, 1])
    """
    from panchi.primitives.vector_space import VectorSpace
    from panchi.primitives.matrix import Matrix
    from panchi.algorithms.reductions import rref

    if not isinstance(space, VectorSpace):
        raise TypeError(
            f"orthogonal_complement() requires a VectorSpace. "
            f"Got {type(space).__name__}."
        )

    basis_vectors = space.basis
    n = space.ambient_dims

    if not basis_vectors:
        return VectorSpace([Vector([1 if i == j else 0 for j in range(n)]) for i in range(n)])

    row_matrix = Matrix([v.to_list() for v in basis_vectors])
    reduction = rref(row_matrix)

    pivot_cols = {col for _, col in reduction.pivots}
    free_cols = [j for j in range(n) if j not in pivot_cols]

    if not free_cols:
        return VectorSpace([Vector([0] * n)])

    null_vectors = []
    for j in free_cols:
        components = [0] * n
        components[j] = 1
        for row, col in reduction.pivots:
            components[col] = -reduction.result[row][j]
        null_vectors.append(Vector(components))

    return VectorSpace(null_vectors)


def vector_projection(projected_vector: Vector, axis_vector: Vector) -> Vector:
    scalar_projection = dot(projected_vector, axis_vector) / dot(axis_vector, axis_vector)
    return scalar_projection * axis_vector


# TODO: Decide if this is a smart file to keep gram schmidt in
def gram_schmidt_algorithm(vectors: list[Vector]) -> list[Vector]:
    if not vectors:
        raise ValueError

    n = len(vectors)
    orthogonal_vectors = []
    for i in range(n):
        orthogonal_vector = vectors[i]
        for j in range(i):
            orthogonal_vector -= vector_projection(vectors[i], orthogonal_vectors[j])

        orthogonal_vectors.append(orthogonal_vector)

    return [v.normalize() for v in orthogonal_vectors]
