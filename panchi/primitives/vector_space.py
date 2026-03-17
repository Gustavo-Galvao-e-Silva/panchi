from __future__ import annotations

from typing import Iterator

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

    def __getitem__(self, index: int) -> Vector:
        """
        Access a spanning vector by index.

        Parameters
        ----------
        index : int
            The index (0-based) of the vector to access. Negative indices
            are supported.

        Returns
        -------
        Vector
            The spanning vector at the specified index.

        Raises
        ------
        TypeError
            If index is not an integer.
        IndexError
            If index is out of range (raised by Python).

        Examples
        --------
        >>> v1 = Vector([1, 0])
        >>> v2 = Vector([0, 1])
        >>> vs = VectorSpace([v1, v2])
        >>> vs[0]
        Vector([1, 0])
        >>> vs[-1]
        Vector([0, 1])
        """
        if not isinstance(index, int):
            raise TypeError(
                f"VectorSpace indices must be integers. Got {type(index).__name__}."
            )

        return self.data[index]

    def __setitem__(self, index: int, new_vector: Vector) -> None:
        """
        Replace a spanning vector at a given index.

        The replacement vector must be a Vector with the same number of
        components as the other vectors in this space.

        Parameters
        ----------
        index : int
            The index (0-based) of the vector to replace. Negative indices
            are supported.
        new_vector : Vector
            The new vector to assign. Must be a Vector with the same
            number of components as the existing vectors.

        Raises
        ------
        TypeError
            If index is not an integer, or new_vector is not a Vector.
        ValueError
            If new_vector has a different number of components than the
            existing vectors in this space.
        IndexError
            If index is out of range (raised by Python).

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> vs[0] = Vector([2, 0])
        >>> vs[0]
        Vector([2, 0])
        """
        if not isinstance(index, int):
            raise TypeError(
                f"VectorSpace indices must be integers. Got {type(index).__name__}."
            )

        if not isinstance(new_vector, Vector):
            raise TypeError(
                f"VectorSpace can only hold Vector instances. "
                f"Got {type(new_vector).__name__}."
            )

        ambient = self.data[0].dims
        if new_vector.dims != ambient:
            raise ValueError(
                f"Cannot assign a {new_vector.dims}-dimensional vector to a space "
                f"whose vectors have {ambient} component(s). "
                f"All vectors in this space must have {ambient} component(s)."
            )

        self.data[index] = new_vector

    def __len__(self) -> int:
        """
        Get the number of spanning vectors in this space.

        Returns
        -------
        int
            The size of the generating set (not necessarily the dimension).

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
        >>> len(vs)
        3
        """
        return len(self.data)

    def __iter__(self) -> Iterator:
        """
        Iterate over the spanning vectors.

        Returns
        -------
        Iterator
            Iterator over the vectors in the generating set.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> for v in vs:
        ...     print(v)
        [1, 0]
        [0, 1]
        """
        return iter(self.data)

    def __eq__(self, other: object) -> bool:
        """
        Check if two VectorSpaces are equal.

        Two VectorSpaces are equal if their generating sets contain the
        same vectors in the same order.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if both spaces have identical generating sets, False otherwise.

        Examples
        --------
        >>> v1, v2 = Vector([1, 0]), Vector([0, 1])
        >>> VectorSpace([v1, v2]) == VectorSpace([v1, v2])
        True
        >>> VectorSpace([v1, v2]) == VectorSpace([v2, v1])
        False
        """
        if not isinstance(other, VectorSpace):
            return NotImplemented

        if len(self.data) != len(other.data):
            return False

        return all(a == b for a, b in zip(self.data, other.data))

    def __str__(self) -> str:
        """
        Return a human-readable representation of the vector space.

        Shows the ambient dimension, the computed dimension, and lists
        the basis vectors.

        Returns
        -------
        str
            A multi-line string showing the space summary and its basis.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> print(vs)
        VectorSpace in R^2, dimension 2
        Basis:
          [1, 0]
          [0, 1]
        """
        ambient = self.data[0].dims
        basis_lines = "\n".join(f"  {v}" for v in self.basis)
        return f"VectorSpace in R^{ambient}, dimension {self.dims}\nBasis:\n{basis_lines}"

    def __repr__(self) -> str:
        """
        Return a constructor-style string for data inspection.

        Returns
        -------
        str
            A compact summary showing ambient dimension, computed dimension,
            and number of generators, such as
            'VectorSpace(ambient=3, dim=2, generators=4)'.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
        >>> repr(vs)
        'VectorSpace(ambient=3, dim=2, generators=3)'
        """
        return (
            f"VectorSpace("
            f"ambient={self.data[0].dims}, "
            f"dim={self.dims}, "
            f"generators={len(self.data)})"
        )

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
