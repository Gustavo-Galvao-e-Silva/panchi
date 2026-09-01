from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import overload

from panchi.primitives.vector import Vector
from panchi.types import SCALAR_TYPES, Scalar, parse_scalar


class Matrix:
    """
    A mathematical matrix for linear algebra operations.

    A matrix is a rectangular array of numbers arranged in rows and columns.
    Matrices are fundamental objects in linear algebra used to represent
    linear transformations, systems of equations, and more.

    Parameters
    ----------
    data : list[list[int | float | Fraction | str]]
        A 2D list representing the matrix. All rows must have the same length.
        Strings like '1/3' are automatically parsed as Fraction values.

    Attributes
    ----------
    data : list[list[int | float | Fraction]]
        The underlying 2D list of matrix elements.
    shape : tuple[int, int]
        The dimensions of the matrix as (rows, columns).

    Raises
    ------
    TypeError
        If data is not a list of lists containing only numbers.
    ValueError
        If rows have inconsistent lengths.

    Examples
    --------
    >>> m = Matrix([[1, 2], [3, 4]])
    >>> print(m.shape)
    (2, 2)
    >>> print(m)
    [[1, 2],
     [3, 4]]
    """

    def __init__(self, data: list[list[int | float | Fraction | str]]) -> None:
        if not isinstance(data, list):
            raise TypeError(
                f"Matrix data must be a list. Got {type(data).__name__} instead."
            )

        if not all(isinstance(row, list) for row in data):
            bad_types = [
                type(row).__name__ for row in data if not isinstance(row, list)
            ]
            raise TypeError(
                f"All rows must be lists. Found invalid row type(s): {', '.join(set(bad_types))}"
            )

        parsed_data = []
        if data:
            for i, row in enumerate(data):
                parsed_row = []
                for j, elem in enumerate(row):
                    try:
                        parsed_row.append(parse_scalar(elem))
                    except TypeError:
                        raise TypeError(
                            f"All matrix elements must be numbers (int, float, Fraction, "
                            f"or string like '1/3'). "
                            f"Found {type(elem).__name__} at position [{i}][{j}]."
                        ) from None
                parsed_data.append(parsed_row)

        self.data = parsed_data
        self.shape = (len(self.data), len(self.data[0])) if parsed_data else (0, 0)

        if self.data:
            expected_cols = self.cols
            for i, row in enumerate(self.data):
                if len(row) != expected_cols:
                    raise ValueError(
                        f"All rows must have the same number of columns. "
                        f"Expected {expected_cols} columns, but row {i} has {len(row)} columns."
                    )

    def _apply_transformation(self, other: Vector | Matrix) -> Vector | Matrix:
        """
        Internal method to apply matrix multiplication transformation.

        Handles both matrix-vector and matrix-matrix multiplication.

        Parameters
        ----------
        other : Vector | Matrix
            The vector or matrix to multiply with.

        Returns
        -------
        Vector | Matrix
            Result of the multiplication. Returns Vector if other is Vector,
            Matrix if other is Matrix.
        """
        if isinstance(other, Vector):
            other_data = [[x] for x in other]
            return_vector = True
        else:
            other_data = other.data
            return_vector = False

        result = []
        for i in range(self.rows):
            new_row = []
            for j in range(len(other_data[0])):
                val = 0
                for k in range(self.cols):
                    val += self.data[i][k] * other_data[k][j]
                new_row.append(val)
            result.append(new_row)

        if return_vector:
            flattened = [item for row in result for item in row]
            return Vector(flattened)
        else:
            return Matrix(result)

    def _validate_element_index(self, index: tuple) -> tuple[int, int]:
        """Validate an (i, j) element index and return it as a pair of ints."""
        if len(index) != 2 or not all(isinstance(k, int) for k in index):
            raise TypeError(
                f"Element indices must be an (int, int) pair. Got {index!r}."
            )
        return index

    def __getitem__(self, index: int | tuple[int, int]) -> list[Scalar] | Scalar:
        """
        Access a row (``m[i]``) or a single element (``m[i, j]``).

        Row access returns a *copy* of the row, so mutating it never changes
        the matrix — assign single elements through ``m[i, j] = value``, which
        is validated exactly like the constructor.

        Parameters
        ----------
        index : int or tuple[int, int]
            A row index ``i``, or an ``(i, j)`` pair selecting one element.
            Negative indices are supported.

        Returns
        -------
        list[int | float | Fraction] or int | float | Fraction
            A copy of the row for ``m[i]``, or the element for ``m[i, j]``.

        Raises
        ------
        TypeError
            If the index is not an int or an (int, int) pair.
        IndexError
            If a row or column index is out of range (raised by Python).

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m[0]
        [1, 2]
        >>> m[1, 1]
        4
        >>> m[-1]
        [3, 4]
        """
        if isinstance(index, tuple):
            i, j = self._validate_element_index(index)
            return self.data[i][j]

        if not isinstance(index, int):
            raise TypeError(
                f"Matrix indices must be an integer or an (int, int) pair. "
                f"Got {type(index).__name__}."
            )

        return list(self.data[index])

    def __setitem__(
        self, index: int | tuple[int, int], value: list[Scalar] | Scalar
    ) -> None:
        """
        Replace a whole row (``m[i] = [...]``) or one element (``m[i, j] = x``).

        Both paths validate their input like the constructor: elements are
        parsed with ``parse_scalar`` (so string fractions such as ``"1/2"``
        become ``Fraction``) and must be numbers. ``m[i, j] = x`` is the way to
        set a single element — ``m[i]`` returns a copy, so ``m[i][j] = x`` would
        not change the matrix.

        Parameters
        ----------
        index : int or tuple[int, int]
            A row index ``i`` for whole-row assignment, or an ``(i, j)`` pair
            for a single element. Negative indices are supported.
        value : list[int | float | Fraction | str] or int | float | Fraction | str
            A row (list) for ``m[i]``, or a single scalar for ``m[i, j]``.

        Raises
        ------
        TypeError
            If the index is malformed, a row is not a list, or an element is
            not a number.
        ValueError
            If a replacement row has a different length than the matrix width.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m[0] = [5, 6]
        >>> m[1, 0] = "1/2"
        >>> print(m)
        [[5, 6],
         [1/2, 4]]
        """
        if isinstance(index, tuple):
            i, j = self._validate_element_index(index)
            try:
                self.data[i][j] = parse_scalar(value)
            except TypeError:
                raise TypeError(
                    f"Matrix elements must be numbers (int, float, or Fraction). "
                    f"Got {type(value).__name__}."
                ) from None
            return

        if not isinstance(index, int):
            raise TypeError(
                f"Matrix indices must be an integer or an (int, int) pair. "
                f"Got {type(index).__name__}."
            )

        new_row = value
        if not isinstance(new_row, list):
            raise TypeError(f"Row must be a list. Got {type(new_row).__name__}.")

        for j, elem in enumerate(new_row):
            try:
                new_row[j] = parse_scalar(elem)
            except TypeError:
                raise TypeError(
                    f"All row elements must be numbers (int, float, or Fraction). "
                    f"Found {type(elem).__name__} at position [{j}]."
                ) from None

        if len(new_row) != self.cols:
            raise ValueError(
                f"Cannot assign row of length {len(new_row)} to a matrix with "
                f"{self.cols} columns. The replacement row must have exactly "
                f"{self.cols} elements."
            )

        self.data[index] = new_row

    def __len__(self) -> int:
        """
        Get the number of rows in the matrix.

        Returns
        -------
        int
            Number of rows.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
        >>> len(m)
        3
        """
        return self.rows

    def __iter__(self) -> Iterator:
        """
        Iterate over the rows of the matrix.

        Returns
        -------
        Iterator
            Iterator over matrix rows.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> for row in m:
        ...     print(row)
        [1, 2]
        [3, 4]
        """
        return iter([list(row) for row in self.data])

    def __add__(self, other: Matrix) -> Matrix:
        """
        Add two matrices element-wise.

        Matrix addition requires both matrices to have the same dimensions.

        Parameters
        ----------
        other : Matrix
            The matrix to add.

        Returns
        -------
        Matrix
            The sum of the two matrices.

        Raises
        ------
        TypeError
            If other is not a Matrix.
        ValueError
            If matrices have different dimensions.

        Examples
        --------
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[5, 6], [7, 8]])
        >>> m3 = m1 + m2
        >>> print(m3)
        [[6, 8],
         [10, 12]]
        """
        if not isinstance(other, Matrix):
            raise TypeError(
                f"Cannot add Matrix and {type(other).__name__}. "
                f"Both operands must be matrices."
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Cannot add matrices with different dimensions. "
                f"First matrix is {self.rows}×{self.cols}, "
                f"second matrix is {other.rows}×{other.cols}. "
                f"Both matrices must have the same shape."
            )

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self.data[i][j] + other.data[i][j])
            result.append(new_row)

        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        """
        Subtract one matrix from another element-wise.

        Matrix subtraction requires both matrices to have the same dimensions.

        Parameters
        ----------
        other : Matrix
            The matrix to subtract.

        Returns
        -------
        Matrix
            The difference of the two matrices.

        Raises
        ------
        TypeError
            If other is not a Matrix.
        ValueError
            If matrices have different dimensions.

        Examples
        --------
        >>> m1 = Matrix([[5, 6], [7, 8]])
        >>> m2 = Matrix([[1, 2], [3, 4]])
        >>> m3 = m1 - m2
        >>> print(m3)
        [[4, 4],
         [4, 4]]
        """
        if not isinstance(other, Matrix):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from Matrix. "
                f"Both operands must be matrices."
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Cannot subtract matrices with different dimensions. "
                f"First matrix is {self.rows}×{self.cols}, "
                f"second matrix is {other.rows}×{other.cols}. "
                f"Both matrices must have the same shape."
            )

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self.data[i][j] - other.data[i][j])
            result.append(new_row)

        return Matrix(result)

    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...

    @overload
    def __matmul__(self, other: Vector) -> Vector: ...

    def __matmul__(self, other: Vector | Matrix) -> Vector | Matrix:
        """
        Multiply matrix by a vector or another matrix using @ operator.

        For matrix-vector multiplication, the number of columns in the matrix
        must equal the dimension of the vector.

        For matrix-matrix multiplication, the number of columns in the first
        matrix must equal the number of rows in the second matrix.

        Parameters
        ----------
        other : Vector | Matrix
            The vector or matrix to multiply with.

        Returns
        -------
        Vector | Matrix
            The result of the multiplication. Returns a Vector when multiplying
            by a Vector, and a Matrix when multiplying by a Matrix.

        Raises
        ------
        TypeError
            If other is not a Vector or Matrix.
        ValueError
            If dimensions are incompatible for multiplication.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> v = Vector([5, 6])
        >>> result = m @ v
        >>> print(result)
        [17, 39]

        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[5, 6], [7, 8]])
        >>> m3 = m1 @ m2
        >>> print(m3)
        [[19, 22],
         [43, 50]]
        """
        if not isinstance(other, (Matrix, Vector)):
            raise TypeError(
                f"Cannot multiply Matrix by {type(other).__name__}. "
                f"Can only multiply by Vector or Matrix."
            )

        other_rows = other.rows if isinstance(other, Matrix) else other.dims

        if self.cols != other_rows:
            if isinstance(other, Vector):
                raise ValueError(
                    f"Cannot multiply {self.rows}×{self.cols} matrix by {other.dims}-dimensional vector. "
                    f"Matrix columns ({self.cols}) must equal vector dimension ({other.dims})."
                )
            else:
                raise ValueError(
                    f"Cannot multiply {self.rows}×{self.cols} matrix by {other.rows}×{other.cols} matrix. "
                    f"First matrix columns ({self.cols}) must equal second matrix rows ({other.rows})."
                )

        return self._apply_transformation(other)

    def __rmul__(self, other: Scalar) -> Matrix:
        """
        Multiply matrix by a scalar (from the left).

        Allows scalar multiplication in the form: scalar * matrix

        Parameters
        ----------
        other : int | float | Fraction
            The scalar to multiply by.

        Returns
        -------
        Matrix
            The result of scalar multiplication.

        Raises
        ------
        TypeError
            If other is not a number.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> result = 3 * m
        >>> print(result)
        [[3, 6],
         [9, 12]]
        """
        if not isinstance(other, SCALAR_TYPES):
            raise TypeError(
                f"Cannot multiply Matrix by {type(other).__name__}. "
                f"Scalar must be a number (int, float, or Fraction)."
            )

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self.data[i][j] * other)
            result.append(new_row)

        return Matrix(result)

    def __mul__(self, other: Scalar) -> Matrix:
        """
        Multiply matrix by a scalar (from the right).

        Allows scalar multiplication in the form: matrix * scalar

        Parameters
        ----------
        other : int | float | Fraction
            The scalar to multiply by.

        Returns
        -------
        Matrix
            The result of scalar multiplication.

        Raises
        ------
        TypeError
            If other is not a number.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> result = m * 3
        >>> print(result)
        [[3, 6],
         [9, 12]]
        """
        return self.__rmul__(other)

    def __pow__(self, exponent: int) -> Matrix:
        """
        Raise matrix to an integer power.

        Matrix exponentiation is only defined for square matrices.
        M^n means multiplying the matrix by itself n times.
        M^0 returns the identity matrix.

        Parameters
        ----------
        exponent : int
            The non-negative integer exponent.

        Returns
        -------
        Matrix
            The matrix raised to the given power.

        Raises
        ------
        TypeError
            If exponent is not an integer.
        ValueError
            If exponent is negative or matrix is not square.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m2 = m ** 2
        >>> print(m2)
        [[7, 10],
         [15, 22]]
        """
        if not isinstance(exponent, int):
            raise TypeError(
                f"Matrix exponent must be an integer. Got {type(exponent).__name__}."
            )

        if exponent < 0:
            raise ValueError(
                f"Matrix exponent cannot be negative. Got {exponent}. "
                f"For matrix inversion, use inverse() from panchi.algorithms instead."
            )

        if not self.is_square:
            raise ValueError(
                f"Cannot raise non-square matrix to a power. "
                f"Matrix exponentiation requires a square matrix, "
                f"but your matrix is {self.rows}×{self.cols}."
            )

        if exponent == 0:
            return self.left_identity

        if exponent == 1:
            return self.copy()

        result = self.copy()
        for _ in range(exponent - 1):
            result = result @ self

        return result

    def __neg__(self) -> Matrix:
        """
        Negate the matrix (multiply by -1).

        Returns
        -------
        Matrix
            The negated matrix.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> neg_m = -m
        >>> print(neg_m)
        [[-1, -2],
         [-3, -4]]
        """
        return -1 * self

    def __eq__(self, other: object) -> bool:
        """
        Check if two matrices are equal.

        Matrices are equal if they have the same shape and all corresponding
        elements are equal.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if matrices are equal, False otherwise.

        Examples
        --------
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[1, 2], [3, 4]])
        >>> m3 = Matrix([[5, 6], [7, 8]])
        >>> m1 == m2
        True
        >>> m1 == m3
        False
        """
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.shape != other.shape:
            return False

        row_num, col_num = self.shape
        for i in range(row_num):
            for j in range(col_num):
                if self.data[i][j] != other.data[i][j]:
                    return False

        return True

    __hash__ = None

    def __str__(self) -> str:
        """
        Return a string representation of the matrix.

        Returns
        -------
        str
            String representation showing the matrix structure.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> print(m)
        [[1, 2],
         [3, 4]]
        """
        formatted_rows = []
        for row in self.data:
            formatted_rows.append("[" + ", ".join(str(x) for x in row) + "]")
        rows = ",\n ".join(formatted_rows)
        return f"[{rows}]"

    def __repr__(self) -> str:
        """
        Return a constructor-style string for data inspection.

        Returns
        -------
        str
            A string showing the class name and data needed to recreate
            this matrix, such as 'Matrix([[1, 2], [3, 4]])'.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> repr(m)
        'Matrix([[1, 2], [3, 4]])'
        """
        return f"Matrix({self.data})"

    @property
    def T(self) -> Matrix:
        """
        Get the transpose of the matrix.

        Shorthand property for transpose() method.

        Returns
        -------
        Matrix
            The transposed matrix.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> print(m.T)
        [[1, 4],
         [2, 5],
         [3, 6]]
        """
        return self.transpose()

    @property
    def rows(self) -> int:
        """
        Get the number of rows in the matrix.

        Returns
        -------
        int
            Number of rows.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
        >>> m.rows
        3
        """
        return self.shape[0]

    @property
    def cols(self) -> int:
        """
        Get the number of columns in the matrix.

        Returns
        -------
        int
            Number of columns.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.cols
        3
        """
        return self.shape[1]

    @property
    def is_square(self) -> bool:
        """
        Check if the matrix is square.

        A matrix is square if it has the same number of rows and columns.
        Square matrices have special properties like determinants and traces.

        Returns
        -------
        bool
            True if matrix is square, False otherwise.

        Examples
        --------
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m1.is_square
        True
        >>> m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m2.is_square
        False
        """
        return self.rows == self.cols

    @property
    def left_identity(self) -> Matrix:
        """
        Get the left identity matrix for this matrix.

        The left identity is a square matrix with the same number of rows
        as this matrix. For any matrix A, I_left × A = A.

        Returns
        -------
        Matrix
            The left identity matrix.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> I = m.left_identity
        >>> print(I)
        [[1, 0],
         [0, 1]]
        """
        n = self.rows
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @property
    def right_identity(self) -> Matrix:
        """
        Get the right identity matrix for this matrix.

        The right identity is a square matrix with the same number of columns
        as this matrix. For any matrix A, A × I_right = A.

        Returns
        -------
        Matrix
            The right identity matrix.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> I = m.right_identity
        >>> print(I)
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
        """
        n = self.cols
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @property
    def trace(self) -> Scalar:
        """
        Calculate the trace of the matrix.

        The trace is the sum of the diagonal elements. Only defined
        for square matrices. The trace has many useful properties in
        linear algebra, including being invariant under change of basis.

        Returns
        -------
        int | float | Fraction
            The sum of the diagonal elements.

        Raises
        ------
        ValueError
            If the matrix is not square.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> m.trace
        15
        """
        if not self.is_square:
            raise ValueError(
                f"Cannot calculate trace of non-square matrix. "
                f"Your matrix is {self.rows}×{self.cols}. "
                f"Trace is only defined for square matrices (n×n)."
            )

        n = self.rows
        return sum(self.data[i][i] for i in range(n))

    @property
    def row_vectors(self) -> list[Vector]:
        """
        Return all rows of the matrix as a list of Vectors.

        Corresponds to the mathematical operator Row(A), which extracts
        the row vectors of a matrix. Each row becomes an independent
        Vector object; modifying the result does not affect this matrix.

        Returns
        -------
        list[Vector]
            A list of Vector objects, one per row, in order.

        Raises
        ------
        ValueError
            If the matrix is empty (has no rows).

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.row_vectors
        [Vector([1, 2, 3]), Vector([4, 5, 6])]
        >>> m.row_vectors[0]
        Vector([1, 2, 3])
        """
        if self.rows == 0:
            raise ValueError("Cannot extract row vectors from an empty matrix.")

        return [Vector(row.copy()) for row in self.data]

    @property
    def col_vectors(self) -> list[Vector]:
        """
        Return all columns of the matrix as a list of Vectors.

        Corresponds to the mathematical operator Col(A), which extracts
        the column vectors of a matrix. Each column becomes an independent
        Vector object; modifying the result does not affect this matrix.

        Returns
        -------
        list[Vector]
            A list of Vector objects, one per column, in order.

        Raises
        ------
        ValueError
            If the matrix is empty (has no columns).

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.col_vectors
        [Vector([1, 4]), Vector([2, 5]), Vector([3, 6])]
        >>> m.col_vectors[1]
        Vector([2, 5])
        """
        if self.cols == 0:
            raise ValueError("Cannot extract column vectors from an empty matrix.")

        return [
            Vector([self.data[i][j] for i in range(self.rows)])
            for j in range(self.cols)
        ]

    def transform(self, vec: Vector) -> Vector:
        """
        Apply this matrix as a linear transformation to a vector.

        This is the fundamental purpose of a matrix: transforming vectors
        in space. Equivalent to matrix-vector multiplication.

        Parameters
        ----------
        vec : Vector
            The vector to transform.

        Returns
        -------
        Vector
            The transformed vector.

        Raises
        ------
        TypeError
            If vec is not a Vector.
        ValueError
            If the vector dimension doesn't match the number of columns.

        Examples
        --------
        >>> m = Matrix([[2, 0], [0, 3]])
        >>> v = Vector([1, 1])
        >>> transformed = m.transform(v)
        >>> print(transformed)
        [2, 3]
        """
        if not isinstance(vec, Vector):
            raise TypeError(
                f"Can only transform Vector objects. Got {type(vec).__name__}."
            )

        return self @ vec

    def transpose(self) -> Matrix:
        """
        Compute the transpose of the matrix.

        The transpose flips the matrix over its diagonal, converting
        rows to columns and vice versa. For matrix A, (A^T)_ij = A_ji.

        Returns
        -------
        Matrix
            The transposed matrix.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> mt = m.transpose()
        >>> print(mt)
        [[1, 4],
         [2, 5],
         [3, 6]]
        """
        result = []
        row_num, col_num = self.shape
        for j in range(col_num):
            new_row = []
            for i in range(row_num):
                new_row.append(self.data[i][j])
            result.append(new_row)

        return Matrix(result)

    def to_list(self) -> list[list[Scalar]]:
        """
        Convert the matrix to a 2D list.

        Returns
        -------
        list[list[int | float | Fraction]]
            A copy of the matrix data as a 2D list.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.to_list()
        [[1, 2], [3, 4]]
        """
        return [row.copy() for row in self.data]

    def copy(self) -> Matrix:
        """
        Create a deep copy of the matrix.

        Returns
        -------
        Matrix
            A new Matrix object with the same data.

        Examples
        --------
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = m1.copy()
        >>> m2[0, 0] = 99
        >>> print(m1[0, 0])
        1
        """
        return Matrix([row.copy() for row in self.data])
