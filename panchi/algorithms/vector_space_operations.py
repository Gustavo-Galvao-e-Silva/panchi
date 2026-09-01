from __future__ import annotations

from panchi.algorithms.matrix_operations import _null_space_basis, rank, solve
from panchi.algorithms.reductions import ref, rref
from panchi.primitives.factories import unit_vector, zero_vector
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace


def basis(space: VectorSpace) -> list[Vector]:
    """
    Return a basis for the span of the vectors in a space.

    Computes a maximal linearly independent subset of the spanning set by
    reducing the matrix whose columns are the vectors to row echelon form.
    Each pivot column corresponds to a vector that is linearly independent
    of all preceding pivot vectors, so the original vectors at those column
    indices form a basis.

    Parameters
    ----------
    space : VectorSpace
        The space whose basis to compute.

    Returns
    -------
    list[Vector]
        A list of vectors from the original spanning set that form a basis.
        The order follows the original input order.

    Examples
    --------
    >>> v1 = Vector([1, 2, 3])
    >>> v2 = Vector([4, 5, 6])
    >>> v3 = Vector([7, 8, 9])  # linearly dependent on v1 and v2
    >>> vs = VectorSpace([v1, v2, v3])
    >>> len(basis(vs))
    2
    """
    vector_col_list = [v.to_list() for v in space.data]
    vector_col_matrix = Matrix(vector_col_list).T
    matrix_ref = ref(vector_col_matrix)
    result = []
    for _, pivot_col in matrix_ref.pivots:
        result.append(space.data[pivot_col])

    return result


def is_full_rank(space: VectorSpace) -> bool:
    """
    Return True if a space spans its entire ambient space.

    A space is full rank when its rank equals the ambient dimension — i.e.
    when the basis vectors span all of R^n.

    Parameters
    ----------
    space : VectorSpace
        The space to test.

    Returns
    -------
    bool
        True if ``rank(space) == space.ambient_dims``, False otherwise.

    Examples
    --------
    >>> is_full_rank(VectorSpace([Vector([1, 0]), Vector([0, 1])]))
    True
    >>> is_full_rank(VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])]))
    False
    """
    return rank(space) == space.ambient_dims


def contains(space: VectorSpace, v: Vector) -> bool:
    """
    Return True if vector v lies in a space.

    Checks membership by attempting to solve the linear system ``Bx = v``,
    where B is the matrix whose columns are the basis vectors. The vector v
    is in the span if and only if the system is consistent.

    Parameters
    ----------
    space : VectorSpace
        The space to test membership against.
    v : Vector
        The vector to test for membership.

    Returns
    -------
    bool
        True if v is in the span of the space, False otherwise.

    Raises
    ------
    TypeError
        If v is not a Vector.
    ValueError
        If v has a different number of components than the vectors in the
        space.

    Examples
    --------
    >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
    >>> contains(vs, Vector([3, 4]))
    True
    >>> vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
    >>> contains(vs2, Vector([0, 0, 1]))
    False
    """
    if not isinstance(v, Vector):
        raise TypeError(f"contains() requires a Vector. Got {type(v).__name__}.")

    if v.dims != space.ambient_dims:
        raise ValueError(
            f"Vector has {v.dims} component(s), but this space lives in "
            f"R^{space.ambient_dims}. Dimensions must match."
        )

    basis_matrix = Matrix([b.to_list() for b in basis(space)]).T
    return solve(basis_matrix, v).status != "inconsistent"


def same_subspace(a: VectorSpace, b: VectorSpace) -> bool:
    """
    Check if two spaces span the same subspace.

    Two subspaces are equal if and only if they have the same rank and every
    basis vector of one is contained in the other. This is a mathematical
    comparison — unlike ``VectorSpace.__eq__``, which checks whether the
    generating sets contain the same vectors, this determines whether the two
    spaces cover the same region of R^n.

    Parameters
    ----------
    a : VectorSpace
        The first space.
    b : VectorSpace
        The second space.

    Returns
    -------
    bool
        True if both spaces span the same subspace, False otherwise.

    Raises
    ------
    TypeError
        If b is not a VectorSpace.
    ValueError
        If the two spaces live in different ambient dimensions.

    Examples
    --------
    >>> v1, v2 = Vector([1, 0]), Vector([0, 1])
    >>> a = VectorSpace([v1, v2])
    >>> b = VectorSpace([v1, v1 + v2])
    >>> same_subspace(a, b)
    True
    >>> a == b
    False
    """
    if not isinstance(b, VectorSpace):
        raise TypeError(
            f"same_subspace() requires a VectorSpace. Got {type(b).__name__}."
        )

    if a.ambient_dims != b.ambient_dims:
        raise ValueError(
            f"Cannot compare subspaces of different ambient dimensions. "
            f"The first lives in R^{a.ambient_dims}, "
            f"but the second lives in R^{b.ambient_dims}."
        )

    if rank(a) != rank(b):
        return False

    return all(contains(b, v) for v in basis(a))


def span(vectors: list[Vector]) -> VectorSpace:
    """
    Construct the vector space spanned by a list of vectors.

    A readable alias for ``VectorSpace(vectors)`` — the span is the set of all
    linear combinations of the given vectors.

    Parameters
    ----------
    vectors : list[Vector]
        The spanning vectors.

    Returns
    -------
    VectorSpace
        The space these vectors span.

    Examples
    --------
    >>> span([Vector([1, 0]), Vector([0, 1])])
    VectorSpace(ambient=2, generators=2)
    """
    return VectorSpace(vectors)


def standard_basis(n: int) -> VectorSpace:
    """
    Construct the standard basis of R^n as a vector space.

    The standard basis is ``{e_1, ..., e_n}``, where ``e_i`` has a 1 in
    position i and 0 elsewhere. The resulting space is all of R^n.

    Parameters
    ----------
    n : int
        The dimension of the ambient space.

    Returns
    -------
    VectorSpace
        The full space R^n, spanned by the standard basis vectors.

    Examples
    --------
    >>> rank(standard_basis(3))
    3
    """
    return VectorSpace([unit_vector(n, i) for i in range(n)])


def column_space(matrix: Matrix) -> VectorSpace:
    """
    Construct the column space of a matrix.

    The column space is the span of the matrix's columns — the set of all
    vectors reachable as ``matrix @ x``. Its dimension equals ``rank(matrix)``.

    Parameters
    ----------
    matrix : Matrix
        The matrix whose column space to build.

    Returns
    -------
    VectorSpace
        The span of the matrix's columns.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 6]])
    >>> rank(column_space(A))
    1
    """
    return VectorSpace(matrix.col_vectors)


def row_space(matrix: Matrix) -> VectorSpace:
    """
    Construct the row space of a matrix.

    The row space is the span of the matrix's rows, equivalently the column
    space of its transpose. Its dimension equals ``rank(matrix)``.

    Parameters
    ----------
    matrix : Matrix
        The matrix whose row space to build.

    Returns
    -------
    VectorSpace
        The span of the matrix's rows.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 6]])
    >>> rank(row_space(A))
    1
    """
    return column_space(matrix.transpose())


def null_space(matrix: Matrix) -> VectorSpace:
    """
    Construct the null space (kernel) of a matrix.

    The null space is the set of all vectors x with ``matrix @ x == 0``. Its
    basis is extracted parametrically from the reduced row echelon form: one
    basis vector per free column. When the matrix has full column rank the
    null space is trivial and is represented by the zero vector (rank 0).

    Parameters
    ----------
    matrix : Matrix
        The matrix whose null space to build.

    Returns
    -------
    VectorSpace
        The kernel of the matrix. ``rank(null_space(A)) == nullity(A)``.

    Examples
    --------
    >>> A = Matrix([[1, 1], [1, 1]])
    >>> rank(null_space(A))
    1
    """
    null_vectors = _null_space_basis(rref(matrix))
    if not null_vectors:
        return VectorSpace([zero_vector(matrix.cols)])
    return VectorSpace(null_vectors)


def orthogonal_complement(space: VectorSpace) -> VectorSpace:
    """
    Compute the orthogonal complement of a vector space.

    The orthogonal complement of a subspace W of R^n is the set of all
    vectors in R^n that are orthogonal to every vector in W. It is exactly
    the null space of the matrix whose rows are the basis vectors of W.

    Parameters
    ----------
    space : VectorSpace
        The subspace whose orthogonal complement is to be computed.

    Returns
    -------
    VectorSpace
        A VectorSpace representing the orthogonal complement. If the input
        space spans all of R^n, returns a VectorSpace containing only the
        zero vector (rank 0).

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
    >>> rank(comp)
    1
    >>> basis(comp)[0]
    Vector([0, 0, 1])
    """
    if not isinstance(space, VectorSpace):
        raise TypeError(
            f"orthogonal_complement() requires a VectorSpace. "
            f"Got {type(space).__name__}."
        )

    basis_vectors = basis(space)
    if not basis_vectors:
        return standard_basis(space.ambient_dims)

    row_matrix = Matrix([v.to_list() for v in basis_vectors])
    return null_space(row_matrix)
