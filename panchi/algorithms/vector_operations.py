from __future__ import annotations

from typing import TYPE_CHECKING

from panchi.primitives.vector import Vector
from panchi.types import Scalar

if TYPE_CHECKING:
    from panchi.primitives.vector_space import VectorSpace


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
    from panchi.algorithms.reductions import rref
    from panchi.primitives.matrix import Matrix
    from panchi.primitives.vector_space import VectorSpace

    if not isinstance(space, VectorSpace):
        raise TypeError(
            f"orthogonal_complement() requires a VectorSpace. "
            f"Got {type(space).__name__}."
        )

    basis_vectors = space.basis
    n = space.ambient_dims

    if not basis_vectors:
        return VectorSpace(
            [Vector([1 if i == j else 0 for j in range(n)]) for i in range(n)]
        )

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
    """
    Project one vector orthogonally onto the line spanned by another.

    The projection of v onto a is the component of v that lies along a,
    computed as (v . a) / (a . a) * a. It is the closest point to v on the
    line through the origin in the direction of a.

    Parameters
    ----------
    projected_vector : Vector
        The vector being projected (v).
    axis_vector : Vector
        The vector defining the direction to project onto (a). Must be
        non-zero.

    Returns
    -------
    Vector
        The projection of projected_vector onto axis_vector.

    Raises
    ------
    ValueError
        If the vectors have different dimensions.
    ZeroDivisionError
        If axis_vector is the zero vector.

    Examples
    --------
    >>> v = Vector([2, 3])
    >>> a = Vector([1, 0])
    >>> print(vector_projection(v, a))
    [2.0, 0.0]
    """
    scalar_projection = dot(projected_vector, axis_vector) / dot(
        axis_vector, axis_vector
    )
    return scalar_projection * axis_vector


class GramSchmidtStep:
    """
    The derivation of a single orthonormal vector in Gram-Schmidt.

    Records, for one input vector, the projections that were removed to make
    it orthogonal to the previously produced directions, the resulting
    orthogonal vector before normalization, and the final unit vector.

    Parameters
    ----------
    index : int
        The position of this vector in the input list (0-based).
    original : Vector
        The input vector this step started from.
    projections : list[Vector]
        The projections onto each previously produced orthogonal vector,
        subtracted in order, that were removed from original.
    orthogonal : Vector
        The orthogonalized vector, before normalization.
    orthonormal : Vector
        The final unit vector (orthogonal normalized to length 1).
    """

    def __init__(
        self,
        index: int,
        original: Vector,
        projections: list[Vector],
        orthogonal: Vector,
        orthonormal: Vector,
    ) -> None:
        self.index = index
        self.original = original
        self.projections = projections
        self.orthogonal = orthogonal
        self.orthonormal = orthonormal

    def __str__(self) -> str:
        lines = [f"Step {self.index + 1}: v{self.index} = {self.original}"]
        for j, projection in enumerate(self.projections):
            lines.append(f"  remove projection onto q{j}: {projection}")
        lines.append(f"  orthogonal: {self.orthogonal}")
        lines.append(f"  normalize -> q{self.index}: {self.orthonormal}")
        return "\n".join(lines)


def _gram_schmidt_steps(vectors: list[Vector]) -> list[GramSchmidtStep]:
    """
    Run Gram-Schmidt, recording the derivation of each orthonormal vector.

    Shared internal implementation behind gram_schmidt (which keeps only the
    final orthonormal vectors) and qr_decomposition (which keeps the full
    step-by-step walkthrough).
    """
    if not vectors:
        raise ValueError("gram_schmidt() requires at least one vector.")

    steps: list[GramSchmidtStep] = []
    orthogonal_vectors: list[Vector] = []
    for i, vector in enumerate(vectors):
        orthogonal_vector = vector
        projections: list[Vector] = []
        for previous in orthogonal_vectors:
            projection = vector_projection(vector, previous)
            projections.append(projection)
            orthogonal_vector -= projection

        orthogonal_vectors.append(orthogonal_vector)
        orthonormal = orthogonal_vector.normalize()
        steps.append(
            GramSchmidtStep(i, vector, projections, orthogonal_vector, orthonormal)
        )

    return steps


def gram_schmidt(vectors: list[Vector]) -> list[Vector]:
    """
    Orthonormalize a list of vectors using the Gram-Schmidt process.

    Given a list of linearly independent vectors, produces an orthonormal
    list spanning the same subspace: each output vector has unit length and
    is orthogonal to all the others. Each input vector has its components
    along the previously produced directions removed, and the remainder is
    normalized.

    Parameters
    ----------
    vectors : list[Vector]
        The vectors to orthonormalize. Must be non-empty; the vectors are
        expected to be linearly independent.

    Returns
    -------
    list[Vector]
        An orthonormal list of vectors spanning the same subspace, in the
        same order as the inputs.

    Raises
    ------
    ValueError
        If vectors is empty.
    ZeroDivisionError
        If the vectors are linearly dependent (a vector reduces to the zero
        vector and cannot be normalized).

    Examples
    --------
    >>> a = Vector([1, 1, 0])
    >>> b = Vector([1, 0, 1])
    >>> q = gram_schmidt([a, b])
    >>> round(dot(q[0], q[1]), 10)
    0.0
    >>> round(q[0].magnitude, 10)
    1.0
    """
    return [step.orthonormal for step in _gram_schmidt_steps(vectors)]
