from __future__ import annotations

from typing import Iterator

from panchi.primitives.vector import Vector


class VectorSpace:
    """
    A subspace of R^n spanned by a set of vectors.

    A VectorSpace represents the span of a given list of vectors — the set of
    all linear combinations of those vectors. It is a slim container over the
    spanning set; the derived structure (``basis``, ``rank``, ``contains``,
    ``is_full_rank``, ``same_subspace``, ``orthogonal_complement``) is computed
    on demand by the free functions in
    ``panchi.algorithms.vector_space_operations``.

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
    >>> from panchi.algorithms import basis, rank
    >>> v1 = Vector([1, 0, 0])
    >>> v2 = Vector([0, 1, 0])
    >>> v3 = Vector([1, 1, 0])  # linearly dependent
    >>> vs = VectorSpace([v1, v2, v3])
    >>> len(basis(vs))
    2
    >>> rank(vs)
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

        Shows the ambient dimension and lists the spanning vectors the space
        was constructed from. The basis and rank are computed on demand by the
        ``basis`` and ``rank`` functions, not stored on the container.

        Returns
        -------
        str
            A multi-line string showing the ambient dimension and generators.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        >>> print(vs)
        VectorSpace in R^2, spanned by 2 vectors:
          [1, 0]
          [0, 1]
        """
        ambient = self.data[0].dims
        count = len(self.data)
        noun = "vector" if count == 1 else "vectors"
        generator_lines = "\n".join(f"  {v}" for v in self.data)
        return (
            f"VectorSpace in R^{ambient}, spanned by {count} {noun}:\n"
            f"{generator_lines}"
        )

    def __repr__(self) -> str:
        """
        Return a constructor-style string for data inspection.

        Returns
        -------
        str
            A compact summary showing the ambient dimension and number of
            generators, such as 'VectorSpace(ambient=3, generators=4)'.

        Examples
        --------
        >>> vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
        >>> repr(vs)
        'VectorSpace(ambient=3, generators=3)'
        """
        return (
            f"VectorSpace("
            f"ambient={self.data[0].dims}, "
            f"generators={len(self.data)})"
        )

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
