from __future__ import annotations

from abc import ABC, abstractmethod
from typing import overload

from panchi.primitives.factories import identity
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.utils.latex import scalar_to_latex
from panchi.utils.types import SCALAR_TYPES, Scalar


class RowOperation(ABC):
    """
    Abstract base class for elementary row operations.

    An elementary row operation transforms a matrix by manipulating its rows.
    Every operation can be represented as left-multiplication by an elementary
    matrix E, such that the following invariant holds for any valid matrix M:

        apply(M) == elementary_matrix(M.rows) @ M

    The same operation can also be applied directly to a Vector, treating it
    as a single-column structure. This is used when solving linear systems by
    applying the reduction steps from A to the right-hand side vector b.

    Subclasses
    ----------
    RowSwap
        Swap two rows: R_a <-> R_b.
    RowScale
        Multiply a row by a non-zero scalar: R_i -> c * R_i.
    RowAdd
        Add a scalar multiple of one row to another: R_t -> R_t + c * R_s.
    """

    @overload
    def apply(self, target: Matrix) -> Matrix: ...

    @overload
    def apply(self, target: Vector) -> Vector: ...

    def apply(self, target: Matrix | Vector) -> Matrix | Vector:
        """
        Apply this operation to a matrix or vector, returning the result.

        Does not modify the input. When applied to a Matrix, the result
        satisfies:

            apply(M) == elementary_matrix(M.rows) @ M

        When applied to a Vector, the operation acts on the vector's
        components using the same row indices, treating the vector as a
        single-column matrix.

        Parameters
        ----------
        target : Matrix | Vector
            The matrix or vector to operate on.

        Returns
        -------
        Matrix | Vector
            A new Matrix or Vector with the operation applied. The return
            type matches the input type.

        Raises
        ------
        TypeError
            If target is not a Matrix or Vector instance.
        ValueError
            If the operation's row indices are out of range for this target.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
        >>> print(RowSwap(0, 2).apply(m))
        [[5, 6],
         [3, 4],
         [1, 2]]
        >>> print(RowAdd(target=1, source=0, scalar=-3).apply(Vector([2, 6])))
        [2, 0]
        """
        self._validate_target(target)
        n = target.dims if isinstance(target, Vector) else target.rows
        self._validate_indices(n)

        if isinstance(target, Vector):
            return self._apply_vector(target)

        return self.elementary_matrix(n) @ target

    @abstractmethod
    def _apply_vector(self, vector: Vector) -> Vector:
        """
        Apply this operation to the components of a vector.

        Called by apply after the row indices have been validated.

        Parameters
        ----------
        vector : Vector
            The vector to operate on. It is not modified.

        Returns
        -------
        Vector
            A new vector with the operation applied.
        """
        pass

    @abstractmethod
    def _validate_indices(self, n: int) -> None:
        """
        Validate that this operation's row indices are in range for size n.

        Parameters
        ----------
        n : int
            Number of rows in the target matrix or vector.

        Raises
        ------
        ValueError
            If a row index is outside [0, n - 1].
        """
        pass

    @abstractmethod
    def elementary_matrix(self, n: int) -> Matrix:
        """
        Return the n×n elementary matrix for this operation.

        The elementary matrix E is constructed by applying this operation
        to the n×n identity matrix. Left-multiplying any matrix M with n rows
        by E is equivalent to applying this operation directly:

            elementary_matrix(n) @ M == apply(M)

        Parameters
        ----------
        n : int
            The size of the elementary matrix. Must be greater than 1 and
            large enough to contain all row indices used by this operation.

        Returns
        -------
        Matrix
            The n×n elementary matrix for this operation.

        Raises
        ------
        TypeError
            If n is not an integer.
        ValueError
            If n is less than 2, or if a row index is out of range for n.
        """
        pass

    @abstractmethod
    def inverse(self) -> RowOperation:
        """
        Return the inverse of this operation.

        The inverse undoes the effect of applying this operation. For any
        matrix M, the following holds:

            inverse().apply(apply(M)) == M

        Returns
        -------
        RowOperation
            A new RowOperation instance that undoes this one.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the standard row operation notation for this operation.

        Returns
        -------
        str
            A string such as 'R0 <-> R2', 'R1 -> 3 * R1', or
            'R2 -> R2 + (-3) * R0'.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a constructor-style string for data inspection.

        Returns
        -------
        str
            A string showing the class name and all arguments needed to
            recreate this operation, such as 'RowSwap(row_a=0, row_b=2)'.
        """
        pass

    @abstractmethod
    def _latex(self) -> str:
        """
        Return the standard row operation notation as LaTeX (0-based rows).

        Returns
        -------
        str
            A LaTeX fragment such as 'R_{0} \\leftrightarrow R_{2}'.
        """
        pass

    def _validate_target(self, target: Matrix | Vector) -> None:
        """
        Validate that the argument is a Matrix or Vector instance.

        Parameters
        ----------
        target : Matrix | Vector
            The object to validate.

        Raises
        ------
        TypeError
            If target is not a Matrix or Vector instance.
        """
        if not isinstance(target, (Matrix, Vector)):
            raise TypeError(
                f"Expected a Matrix or Vector, but got {type(target).__name__}. "
                f"Row operations can only be applied to Matrix or Vector objects."
            )

    def _validate_n(self, n: int) -> None:
        """
        Validate the size argument for elementary_matrix.

        Parameters
        ----------
        n : int
            The matrix size to validate.

        Raises
        ------
        TypeError
            If n is not an integer.
        ValueError
            If n is less than 2.
        """
        if not isinstance(n, int):
            raise TypeError(
                f"Matrix size n must be an integer. Got {type(n).__name__}."
            )

        if n < 2:
            raise ValueError(
                f"Matrix size n must be at least 2. Got {n}. "
                f"Row operations require at least two rows to be meaningful."
            )

    @staticmethod
    def _validate_index_type(index: int, name: str | None = None) -> None:
        """
        Validate that a row index is an integer.

        Parameters
        ----------
        index : int
            The row index to validate.
        name : str, optional
            The argument name to mention in the error message.

        Raises
        ------
        TypeError
            If index is not an integer.
        """
        if not isinstance(index, int):
            where = f" for {name}" if name else ""
            raise TypeError(
                f"Row index must be an integer. Got {type(index).__name__}{where}."
            )

    @staticmethod
    def _validate_in_range(index: int, n: int, label: str = "Row") -> None:
        """
        Validate that a row index is in range for a matrix or vector of size n.

        Parameters
        ----------
        index : int
            The row index to validate.
        n : int
            Number of rows in the target matrix or vector.
        label : str, optional
            How the index is described in the error message. Defaults to "Row".

        Raises
        ------
        ValueError
            If index is outside [0, n - 1].
        """
        if not (0 <= index < n):
            raise ValueError(
                f"{label} index {index} is out of range for a matrix with {n} rows. "
                f"Valid indices are 0 to {n - 1}."
            )

    @staticmethod
    def _validate_scalar_type(scalar: Scalar) -> None:
        """
        Validate that a scalar is a number.

        Parameters
        ----------
        scalar : Scalar
            The value to validate.

        Raises
        ------
        TypeError
            If scalar is not an int, float, Fraction, or QuadraticNumber.
        """
        if not isinstance(scalar, SCALAR_TYPES):
            raise TypeError(
                f"Scalar must be a number (int, float, or Fraction). "
                f"Got {type(scalar).__name__}."
            )


class RowSwap(RowOperation):
    """
    Elementary row operation: swap two rows.

    Represents the operation R_a <-> R_b. The corresponding elementary
    matrix is the identity matrix with rows a and b exchanged.

    Applying this operation twice returns the original matrix.

    Parameters
    ----------
    row_a : int
        Index of the first row to swap (0-based).
    row_b : int
        Index of the second row to swap (0-based).

    Examples
    --------
    >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
    >>> op = RowSwap(0, 2)
    >>> print(op.apply(m))
    [[5, 6],
     [3, 4],
     [1, 2]]
    >>> v = Vector([1, 2, 3])
    >>> print(op.apply(v))
    [3, 2, 1]
    >>> print(op.elementary_matrix(3))
    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]]
    >>> print(op)
    R0 <-> R2
    >>> repr(op)
    'RowSwap(row_a=0, row_b=2)'
    """

    def __init__(self, row_a: int, row_b: int) -> None:
        self.a = row_a
        self.b = row_b
        self._validate_index_type(self.a, "row_a")
        self._validate_index_type(self.b, "row_b")

    def _validate_indices(self, n: int) -> None:
        self._validate_in_range(self.a, n)
        self._validate_in_range(self.b, n)

    def elementary_matrix(self, n: int) -> Matrix:
        """
        Return the n×n elementary matrix for this row swap.

        Constructed by swapping rows a and b in the n×n identity matrix.
        This matrix has determinant -1, reflecting that row swaps reverse
        the orientation of the row space.

        Parameters
        ----------
        n : int
            Size of the elementary matrix. Must be at least 2, and large
            enough so that both row indices are in range.

        Returns
        -------
        Matrix
            An n×n matrix identical to the identity except that rows a
            and b are exchanged.

        Raises
        ------
        TypeError
            If n is not an integer.
        ValueError
            If n < 2 or either row index is out of range for n.

        Examples
        --------
        >>> op = RowSwap(0, 1)
        >>> print(op.elementary_matrix(3))
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1]]
        """
        self._validate_n(n)
        self._validate_indices(n)

        grid: Matrix = identity(n)
        row_a: list[Scalar] = grid[self.a].copy()
        row_b: list[Scalar] = grid[self.b].copy()
        grid[self.a] = row_b
        grid[self.b] = row_a

        return grid

    def _apply_vector(self, vector: Vector) -> Vector:
        result = vector.copy()
        result[self.a], result[self.b] = vector[self.b], vector[self.a]
        return result

    def inverse(self) -> RowSwap:
        """
        Return the inverse of this row swap.

        A row swap is its own inverse: swapping the same two rows a second
        time restores the original matrix.

        Returns
        -------
        RowSwap
            A new RowSwap with the same row indices.

        Examples
        --------
        >>> op = RowSwap(0, 2)
        >>> op.inverse()
        RowSwap(row_a=0, row_b=2)
        """
        return RowSwap(self.a, self.b)

    def __str__(self) -> str:
        return f"R{self.a} <-> R{self.b}"

    def __repr__(self) -> str:
        return f"RowSwap(row_a={self.a}, row_b={self.b})"

    def _latex(self) -> str:
        return f"R_{{{self.a}}} \\leftrightarrow R_{{{self.b}}}"


class RowScale(RowOperation):
    """
    Elementary row operation: multiply a row by a non-zero scalar.

    Represents the operation R_i -> scalar * R_i. The corresponding
    elementary matrix is the identity matrix with the diagonal entry
    at position [i, i] replaced by scalar.

    Scaling a row by scalar multiplies the determinant of the matrix
    by scalar. To invert this operation, scale by 1 / scalar.

    Parameters
    ----------
    row : int
        Index of the row to scale (0-based).
    scalar : int | float | Fraction
        The non-zero value to multiply the row by.

    Examples
    --------
    >>> m = Matrix([[1, 2], [3, 4]])
    >>> op = RowScale(1, 3)
    >>> print(op.apply(m))
    [[1, 2],
     [9, 12]]
    >>> v = Vector([1, 2])
    >>> print(op.apply(v))
    [1, 6]
    >>> print(op.elementary_matrix(2))
    [[1, 0],
     [0, 3]]
    >>> print(op)
    R1 -> 3 * R1
    >>> repr(op)
    'RowScale(row=1, scalar=3)'
    """

    def __init__(self, row: int, scalar: Scalar) -> None:
        self.row = row
        self.scalar = scalar
        self._validate_index_type(self.row)
        self._validate_scalar_type(self.scalar)
        if self.scalar == 0:
            raise ValueError(
                "Scalar must be non-zero. Scaling a row by zero would make "
                "the matrix singular and the operation non-invertible."
            )

    def _validate_indices(self, n: int) -> None:
        self._validate_in_range(self.row, n)

    def elementary_matrix(self, n: int) -> Matrix:
        """
        Return the n×n elementary matrix for this row scale.

        Constructed from the identity matrix with the diagonal entry at
        position [row, row] replaced by scalar.

        Parameters
        ----------
        n : int
            Size of the elementary matrix. Must be at least 2, and large
            enough so that the row index is in range.

        Returns
        -------
        Matrix
            An n×n matrix identical to the identity except that entry
            [row, row] equals scalar.

        Raises
        ------
        TypeError
            If n is not an integer.
        ValueError
            If n < 2 or the row index is out of range.

        Examples
        --------
        >>> op = RowScale(0, 5)
        >>> print(op.elementary_matrix(3))
        [[5, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
        """
        self._validate_n(n)
        self._validate_indices(n)

        grid: Matrix = identity(n)
        grid[self.row, self.row] = self.scalar

        return grid

    def _apply_vector(self, vector: Vector) -> Vector:
        result = vector.copy()
        result[self.row] = vector[self.row] * self.scalar
        return result

    def inverse(self) -> RowScale:
        """
        Return the inverse of this row scale.

        The inverse scales the same row by 1 / scalar, which restores
        the original values.

        Returns
        -------
        RowScale
            A new RowScale on the same row with scalar 1 / self.scalar.

        Examples
        --------
        >>> op = RowScale(1, 3)
        >>> op.inverse()
        RowScale(row=1, scalar=0.3333333333333333)
        """
        return RowScale(self.row, 1 / self.scalar)

    def __str__(self) -> str:
        return f"R{self.row} -> {self.scalar} * R{self.row}"

    def __repr__(self) -> str:
        return f"RowScale(row={self.row}, scalar={self.scalar})"

    def _latex(self) -> str:
        return (
            f"R_{{{self.row}}} \\to {scalar_to_latex(self.scalar)}\\,R_{{{self.row}}}"
        )


class RowAdd(RowOperation):
    """
    Elementary row operation: add a scalar multiple of one row to another.

    Represents the operation R_target -> R_target + scalar * R_source.
    The corresponding elementary matrix is the identity with scalar placed
    at position [target, source].

    This is the core operation of Gaussian elimination. When scalar is
    chosen to eliminate an entry, the result is a zero in position
    [target, source_col] of the transformed matrix.

    The inverse of this operation is RowAdd(target, source, -scalar).

    Parameters
    ----------
    target : int
        Index of the row being modified (0-based).
    source : int
        Index of the row being added (0-based). Must differ from target.
    scalar : int | float | Fraction
        The value to multiply the source row by before adding.

    Examples
    --------
    >>> m = Matrix([[1, 2], [3, 4]])
    >>> op = RowAdd(target=1, source=0, scalar=-3)
    >>> print(op.apply(m))
    [[1,  2],
     [0, -2]]
    >>> v = Vector([1, 2])
    >>> print(op.apply(v))
    [1, -1]
    >>> print(op.elementary_matrix(2))
    [[1,  0],
     [-3, 1]]
    >>> print(op)
    R1 -> R1 + (-3) * R0
    >>> repr(op)
    'RowAdd(target=1, source=0, scalar=-3)'
    """

    def __init__(self, target: int, source: int, scalar: Scalar) -> None:
        self.target = target
        self.source = source
        self.scalar = scalar
        self._validate_index_type(self.target, "target")
        self._validate_index_type(self.source, "source")
        self._validate_scalar_type(self.scalar)

    def _validate_indices(self, n: int) -> None:
        self._validate_in_range(self.target, n, "Target row")
        self._validate_in_range(self.source, n, "Source row")

        if self.target == self.source:
            raise ValueError(
                f"Target and source rows must be different. Both are row {self.target}. "
                f"To scale a row, use RowScale instead."
            )

    def elementary_matrix(self, n: int) -> Matrix:
        """
        Return the n×n elementary matrix for this row addition.

        Constructed from the identity matrix with scalar placed at position
        [target, source]. This encodes the fact that left-multiplying by E
        replaces row target with row target plus scalar times row source.

        Parameters
        ----------
        n : int
            Size of the elementary matrix. Must be at least 2 and large
            enough so that both row indices are in range.

        Returns
        -------
        Matrix
            An n×n matrix identical to the identity except that entry
            [target, source] equals scalar.

        Raises
        ------
        TypeError
            If n is not an integer.
        ValueError
            If n < 2, indices are out of range, or target equals source.

        Examples
        --------
        >>> op = RowAdd(target=2, source=0, scalar=4)
        >>> print(op.elementary_matrix(3))
        [[1, 0, 0],
         [0, 1, 0],
         [4, 0, 1]]
        """
        self._validate_n(n)
        self._validate_indices(n)

        grid: Matrix = identity(n)
        grid[self.target, self.source] = self.scalar

        return grid

    def _apply_vector(self, vector: Vector) -> Vector:
        result = vector.copy()
        result[self.target] = vector[self.target] + self.scalar * vector[self.source]
        return result

    def inverse(self) -> RowAdd:
        """
        Return the inverse of this row addition.

        The inverse subtracts the same scalar multiple of the source row
        from the target row, which restores the original values.

        Returns
        -------
        RowAdd
            A new RowAdd with the same rows and negated scalar.

        Examples
        --------
        >>> op = RowAdd(target=1, source=0, scalar=-3)
        >>> op.inverse()
        RowAdd(target=1, source=0, scalar=3)
        """
        return RowAdd(self.target, self.source, -self.scalar)

    def __str__(self) -> str:
        return f"R{self.target} -> R{self.target} + ({self.scalar}) * R{self.source}"

    def __repr__(self) -> str:
        return (
            f"RowAdd(target={self.target}, source={self.source}, scalar={self.scalar})"
        )

    def _latex(self) -> str:
        return (
            f"R_{{{self.target}}} \\to R_{{{self.target}}} "
            f"+ ({scalar_to_latex(self.scalar)})\\,R_{{{self.source}}}"
        )
