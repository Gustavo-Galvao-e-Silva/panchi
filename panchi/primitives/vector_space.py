from __future__ import annotations

from panchi.primitives.vector import Vector
from panchi.primitives.matrix import Matrix
from panchi.algorithms.reductions import ref


class VectorSpace:
    """
    A subspace of R^n spanned by a set of vectors.

    A VectorSpace represents the span of a given list of vectors — the set of
    all linear combinations of those vectors. It provides access to a basis
    (a maximal linearly independent subset) and the dimension of the subspace.

    Parameters
    ----------
    vectors : list[Vector]
        A non-empty list of vectors that span the space. All vectors must
        have the same number of components.

    Attributes
    ----------
    data : list[Vector]
        The original spanning set as provided.

    Raises
    ------
    TypeError
        If vectors is not a list, or if any element is not a Vector instance.
    ValueError
        If the list is empty, or if the vectors have inconsistent dimensions.

    Examples
    --------
    >>> v1 = Vector([1, 0, 0])
    >>> v2 = Vector([0, 1, 0])
    >>> v3 = Vector([1, 1, 0])  # linearly dependent
    >>> vs = VectorSpace([v1, v2, v3])
    >>> len(vs.basis)
    2
    >>> vs.dims
    2
    """

    def __init__(self, vectors: list[Vector]) -> None:
        if not isinstance(vectors, list):
            raise TypeError(
                f"VectorSpace requires a list of vectors. "
                f"Got {type(vectors).__name__}."
            )

        if not vectors:
            raise ValueError(
                "VectorSpace requires at least one vector. "
                "An empty spanning set defines no subspace."
            )

        bad = [type(v).__name__ for v in vectors if not isinstance(v, Vector)]
        if bad:
            raise TypeError(
                f"All elements must be Vector instances. "
                f"Found: {', '.join(set(bad))}."
            )

        dims = vectors[0].dims
        for i, v in enumerate(vectors[1:], start=1):
            if v.dims != dims:
                raise ValueError(
                    f"All vectors must have the same number of components. "
                    f"Vector 0 has {dims} component(s), "
                    f"but vector {i} has {v.dims}."
                )

        self.data = vectors

    @property
    def basis(self) -> list[Vector]:
        """
        Return a basis for the span of the vectors in this space.

        Computes a maximal linearly independent subset of the spanning set by
        reducing the matrix whose columns are the vectors to row echelon form.
        Each pivot column corresponds to a vector that is linearly independent
        of all preceding pivot vectors, so the original vectors at those column
        indices form a basis.

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
        >>> len(vs.basis)
        2
        """
        vector_col_list = [v.to_list() for v in self.data]
        vector_col_matrix = Matrix(vector_col_list).T
        matrix_ref = ref(vector_col_matrix)
        result = []
        for _, pivot_col in matrix_ref.pivots:
            result.append(self.data[pivot_col])

        return result

    @property
    def dims(self) -> int:
        """
        Return the dimension of the subspace.

        The dimension equals the number of vectors in a basis, which is the
        rank of the matrix formed by the spanning vectors. This is always less
        than or equal to the number of vectors in the original spanning set.

        Returns
        -------
        int
            The number of linearly independent vectors in the basis.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
        >>> vs.dims
        2
        """
        return len(self.basis)
