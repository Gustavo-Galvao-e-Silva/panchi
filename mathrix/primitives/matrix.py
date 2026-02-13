from typing import Iterator

from mathrix.primitives.vector import Vector


class Matrix:
    """
    A mathematical matrix for linear algebra operations.

    A matrix is a rectangular array of numbers arranged in rows and columns.
    Matrices are fundamental objects in linear algebra used to represent
    linear transformations, systems of equations, and more.

    Parameters
    ----------
    data : list[list[int | float]]
        A 2D list representing the matrix. All rows must have the same length.

    Attributes
    ----------
    data : list[list[int | float]]
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

    def __init__(self, data: list[list[int | float]]) -> None:
        if not isinstance(data, list):
            raise TypeError(
                f"Matrix data must be a list. Got {type(data).__name__} instead."
            )

        if not all(isinstance(row, list) for row in data):
            bad_types = [type(row).__name__ for row in data if not isinstance(row, list)]
            raise TypeError(
                f"All rows must be lists. Found invalid row type(s): {', '.join(set(bad_types))}"
            )

        if data:  # Check elements only if data is not empty
            for i, row in enumerate(data):
                for j, elem in enumerate(row):
                    if not isinstance(elem, (int, float)):
                        raise TypeError(
                            f"All matrix elements must be numbers (int or float). "
                            f"Found {type(elem).__name__} at position [{i}][{j}]."
                        )

        self.data = data
        self.shape = (len(self.data), len(self.data[0])) if data else (0, 0)

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
            other_data = other.copy()
            return_vector = False

        result = []
        for i in range(self.rows):
            new_row = []
            for j in range(len(other_data[0])):
                val = 0
                for k in range(self.cols):
                    val += self[i][k] * other_data[k][j]
                new_row.append(val)
            result.append(new_row)

        if return_vector:
            flattened = [item for row in result for item in row]
            return Vector(flattened)
        else:
            return Matrix(result)

    def __getitem__(self, index: int) -> list[int | float]:
        """
        Access a row of the matrix by index.

        Parameters
        ----------
        index : int
            The row index (0-based).

        Returns
        -------
        list[int | float]
            The row at the specified index.

        Raises
        ------
        TypeError
            If index is not an integer.
        IndexError
            If index is out of range.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m[0]
        [1, 2]
        >>> m[1][1]
        4
        """
        if not isinstance(index, int):
            raise TypeError(
                f"Matrix row indices must be integers. Got {type(index).__name__}."
            )

        if index < 0 or index >= self.rows:
            raise IndexError(
                f"Row index {index} is out of range. Matrix has {self.rows} row(s), "
                f"so valid indices are 0 to {self.rows - 1}."
            )

        return self[index]

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
        return iter(self.data)

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
                new_row.append(self[i][j] + other[i][j])
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
                new_row.append(self[i][j] - other[i][j])
            result.append(new_row)

        return Matrix(result)

    def __mul__(self, other: Vector | Matrix) -> Vector | Matrix:
        """
        Multiply matrix by a vector or another matrix.

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
        >>> result = m * v
        >>> print(result)
        [17, 39]

        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[5, 6], [7, 8]])
        >>> m3 = m1 * m2
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

    def __rmul__(self, other: int | float) -> Matrix:
        """
        Multiply matrix by a scalar (from the left).

        Allows scalar multiplication in the form: scalar * matrix

        Parameters
        ----------
        other : int | float
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
        if not isinstance(other, (int, float)):
            raise TypeError(
                f"Cannot multiply Matrix by {type(other).__name__}. "
                f"Scalar must be a number (int or float)."
            )

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self[i][j] * other)
            result.append(new_row)

        return Matrix(result)

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
                f"For matrix inversion, use the inverse() method instead."
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
            result = result * self

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
                if self[i][j] != other[i][j]:
                    return False

        return True

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
        rows = ",\n ".join(str(row) for row in self.data)
        return f"[{rows}]"

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
    def determinant(self) -> float:
        """
        Calculate the determinant of the matrix.

        The determinant is a scalar value that encodes certain properties
        of the matrix, including whether it's invertible. Only defined
        for square matrices.

        Returns
        -------
        float
            The determinant of the matrix.

        Raises
        ------
        ValueError
            If the matrix is not square.

        Notes
        -----
        This method is not yet implemented.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.determinant
        -2.0
        """
        if not self.is_square:
            raise ValueError(
                f"Cannot calculate determinant of non-square matrix. "
                f"Your matrix is {self.rows}×{self.cols}. "
                f"Determinants are only defined for square matrices (n×n)."
            )

        pass

    @property
    def trace(self) -> int | float:
        """
        Calculate the trace of the matrix.

        The trace is the sum of the diagonal elements. Only defined
        for square matrices. The trace has many useful properties in
        linear algebra, including being invariant under change of basis.

        Returns
        -------
        int | float
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
        return sum(self[i][i] for i in range(n))

    def get_row(self, index: int) -> list[int | float]:
        """
        Get a copy of a specific row.

        Parameters
        ----------
        index : int
            The row index (0-based).

        Returns
        -------
        list[int | float]
            A copy of the row at the specified index.

        Raises
        ------
        TypeError
            If index is not an integer.
        IndexError
            If index is out of range.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
        >>> m.get_row(1)
        [3, 4]
        """
        if not isinstance(index, int):
            raise TypeError(
                f"Row index must be an integer. Got {type(index).__name__}."
            )

        if index < 0 or index >= self.rows:
            raise IndexError(
                f"Row index {index} is out of range. Matrix has {self.rows} row(s), "
                f"so valid indices are 0 to {self.rows - 1}."
            )

        return self[index].copy()

    def get_col(self, index: int) -> list[int | float]:
        """
        Get a specific column as a list.

        Parameters
        ----------
        index : int
            The column index (0-based).

        Returns
        -------
        list[int | float]
            The column at the specified index.

        Raises
        ------
        TypeError
            If index is not an integer.
        IndexError
            If index is out of range.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.get_col(1)
        [2, 5]
        """
        if not isinstance(index, int):
            raise TypeError(
                f"Column index must be an integer. Got {type(index).__name__}."
            )

        if index < 0 or index >= self.cols:
            raise IndexError(
                f"Column index {index} is out of range. Matrix has {self.cols} column(s), "
                f"so valid indices are 0 to {self.cols - 1}."
            )

        return [self[i][index] for i in range(self.rows)]

    def get_rows(self) -> list[list[int | float]]:
        """
        Get all rows as a list of lists.

        Returns
        -------
        list[list[int | float]]
            A copy of all matrix rows.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.get_rows()
        [[1, 2], [3, 4]]
        """
        return [row.copy() for row in self]

    def get_cols(self, to_vector: bool = False) -> list[list[int | float]] | list[Vector]:
        """
        Get all columns as a list.

        Parameters
        ----------
        to_vector : bool, optional
            If True, return columns as Vector objects.
            If False, return columns as lists. Default is False.

        Returns
        -------
        list[list[int | float]] | list[Vector]
            List of columns, either as lists or Vector objects.

        Raises
        ------
        TypeError
            If to_vector is not a boolean.

        Examples
        --------
        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.get_cols()
        [[1, 4], [2, 5], [3, 6]]
        >>> m.get_cols(to_vector=True)
        [Vector([1, 4]), Vector([2, 5]), Vector([3, 6])]
        """
        if not isinstance(to_vector, bool):
            raise TypeError(
                f"Parameter 'to_vector' must be a boolean (True or False). "
                f"Got {type(to_vector).__name__}."
            )

        result = [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        if to_vector:
            return [Vector(col) for col in result]

        return result

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
        >>> m = Matrix([[2, 0], [0, 3]])  # Scaling transformation
        >>> v = Vector([1, 1])
        >>> transformed = m.transform(v)
        >>> print(transformed)
        [2, 3]
        """
        if not isinstance(vec, Vector):
            raise TypeError(
                f"Can only transform Vector objects. Got {type(vec).__name__}."
            )

        return self * vec

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

    def to_list(self) -> list[list[int | float]]:
        """
        Convert the matrix to a 2D list.

        Returns
        -------
        list[list[int | float]]
            A copy of the matrix data as a 2D list.

        Examples
        --------
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.to_list()
        [[1, 2], [3, 4]]
        """
        return self.get_rows()

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
        >>> m2[0][0] = 99
        >>> print(m1[0][0])  # Original unchanged
        1
        """
        return Matrix(self.get_rows())
