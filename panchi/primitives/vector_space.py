from __future__ import annotations

from collections.abc import Iterator

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector


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
        same vectors, regardless of order.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if both spaces have the same generating sets, False otherwise.

        Examples
        --------
        >>> v1, v2 = Vector([1, 0]), Vector([0, 1])
        >>> VectorSpace([v1, v2]) == VectorSpace([v1, v2])
        True
        >>> VectorSpace([v1, v2]) == VectorSpace([v2, v1])
        True
        """
        if not isinstance(other, VectorSpace):
            return NotImplemented

        if len(self.data) != len(other.data):
            return False

        return sorted(v.to_list() for v in self.data) == sorted(
            v.to_list() for v in other.data
        )

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
        return (
            f"VectorSpace in R^{ambient}, dimension {self.dims}\nBasis:\n{basis_lines}"
        )

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
        from panchi.algorithms.reductions import ref

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

    @property
    def ambient_dims(self) -> int:
        """
        Return the dimension of the ambient space R^n.

        This is the number of components in each vector, i.e. the n in R^n
        that this subspace lives inside.

        Returns
        -------
        int
            The number of components of each vector in this space.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        >>> vs.ambient_dims
        3
        """
        return self.data[0].dims

    @property
    def is_full_rank(self) -> bool:
        """
        Return True if the subspace spans the entire ambient space.

        A VectorSpace is full rank when its dimension equals the ambient
        dimension — i.e. when the basis vectors span all of R^n.

        Returns
        -------
        bool
            True if ``dims == ambient_dims``, False otherwise.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> vs.is_full_rank
        True
        >>> vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        >>> vs2.is_full_rank
        False
        """
        return self.dims == self.ambient_dims

    def contains(self, v: Vector) -> bool:
        """
        Return True if vector v lies in this subspace.

        Checks membership by attempting to solve the linear system
        ``Bx = v``, where B is the matrix whose columns are the basis
        vectors. The vector v is in the span if and only if the system
        is consistent.

        Parameters
        ----------
        v : Vector
            The vector to test for membership.

        Returns
        -------
        bool
            True if v is in the span of this space, False otherwise.

        Raises
        ------
        TypeError
            If v is not a Vector.
        ValueError
            If v has a different number of components than the vectors
            in this space.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> vs.contains(Vector([3, 4]))
        True
        >>> vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        >>> vs2.contains(Vector([0, 0, 1]))
        False
        """
        if not isinstance(v, Vector):
            raise TypeError(f"contains() requires a Vector. Got {type(v).__name__}.")

        if v.dims != self.ambient_dims:
            raise ValueError(
                f"Vector has {v.dims} component(s), but this space lives in "
                f"R^{self.ambient_dims}. Dimensions must match."
            )

        from panchi.algorithms.matrix_operations import solve

        basis_matrix = Matrix([b.to_list() for b in self.basis]).T
        return solve(basis_matrix, v).status != "inconsistent"

    def same_subspace(self, other: VectorSpace) -> bool:
        """
        Check if two VectorSpaces span the same subspace.

        Two subspaces are equal if and only if they have the same dimension
        and every basis vector of one is contained in the other. This is a
        mathematical comparison — unlike ``__eq__``, which checks whether the
        generating sets contain the same vectors, this method determines
        whether the two spaces cover the same region of R^n.

        Parameters
        ----------
        other : VectorSpace
            The VectorSpace to compare with.

        Returns
        -------
        bool
            True if both spaces span the same subspace, False otherwise.

        Raises
        ------
        TypeError
            If other is not a VectorSpace.
        ValueError
            If the two spaces live in different ambient dimensions.

        Examples
        --------
        >>> v1, v2 = Vector([1, 0]), Vector([0, 1])
        >>> vs1 = VectorSpace([v1, v2])
        >>> vs2 = VectorSpace([v1, v1 + v2])
        >>> vs1.same_subspace(vs2)
        True
        >>> vs1 == vs2
        False
        """
        if not isinstance(other, VectorSpace):
            raise TypeError(
                f"same_subspace() requires a VectorSpace. "
                f"Got {type(other).__name__}."
            )

        if self.ambient_dims != other.ambient_dims:
            raise ValueError(
                f"Cannot compare subspaces of different ambient dimensions. "
                f"This space lives in R^{self.ambient_dims}, "
                f"but the other lives in R^{other.ambient_dims}."
            )

        if self.dims != other.dims:
            return False

        return all(other.contains(v) for v in self.basis)
