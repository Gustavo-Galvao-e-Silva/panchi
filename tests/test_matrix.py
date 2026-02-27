import pytest

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector

# ==================== MATRIX TESTS ====================


class TestMatrixInitialization:
    """Test cases for Matrix object initialization and validation."""

    def test_valid_matrix_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ Matrix([[1,2],[3,4]]) → data={m.data}, shape={m.shape}")
        assert m.data == [[1, 2], [3, 4]]
        assert m.shape == (2, 2)

    def test_valid_matrix_3x2(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        print(f"\n✓ Matrix([[1,2],[3,4],[5,6]]) → shape={m.shape} (expected (3,2))")
        assert m.shape == (3, 2)

    def test_valid_matrix_floats(self):
        m = Matrix([[1.5, 2.5], [3.5, 4.5]])
        print(f"\n✓ Matrix with floats → data={m.data}")
        assert m.data == [[1.5, 2.5], [3.5, 4.5]]

    def test_empty_matrix(self):
        m = Matrix([])
        print(f"\n✓ Matrix([]) → data={m.data}, shape={m.shape}")
        assert m.data == []
        assert m.shape == (0, 0)

    def test_single_element_matrix(self):
        m = Matrix([[5]])
        print(f"\n✓ Matrix([[5]]) → data={m.data}, shape={m.shape}")
        assert m.data == [[5]]
        assert m.shape == (1, 1)

    def test_row_vector_matrix(self):
        m = Matrix([[1, 2, 3, 4]])
        print(f"\n✓ Row vector matrix → shape={m.shape} (expected (1,4))")
        assert m.shape == (1, 4)

    def test_column_vector_matrix(self):
        m = Matrix([[1], [2], [3]])
        print(f"\n✓ Column vector matrix → shape={m.shape} (expected (3,1))")
        assert m.shape == (3, 1)

    def test_invalid_jagged_matrix(self):
        print(f"\n✓ Jagged matrix [[1,2],[3,4,5]] → raises ValueError")
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3, 4, 5]])

    def test_invalid_type_not_list(self):
        print(f"\n✓ Matrix('not a matrix') → raises TypeError")
        with pytest.raises(TypeError):
            Matrix("not a matrix")

    def test_invalid_type_in_matrix(self):
        print(f"\n✓ Matrix with strings → raises TypeError")
        with pytest.raises(TypeError):
            Matrix([[1, 2], ["a", "b"]])


class TestMatrixIndexing:
    """Test cases for Matrix indexing operations."""

    def test_valid_row_access(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        print(f"\n✓ m[0]={m[0]}, m[1]={m[1]}, m[2]={m[2]}")
        assert m[0] == [1, 2]
        assert m[1] == [3, 4]
        assert m[2] == [5, 6]

    def test_valid_element_access(self):
        m = Matrix([[1, 2], [3, 4]])
        print(
            f"\n✓ m[0][0]={m[0][0]}, m[0][1]={m[0][1]}, m[1][0]={m[1][0]}, m[1][1]={m[1][1]}"
        )
        assert m[0][0] == 1
        assert m[0][1] == 2
        assert m[1][0] == 3
        assert m[1][1] == 4

    def test_negative_indexing(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ m[-1]={m[-1]} (expected [3,4])")
        assert m[-1] == [3, 4]

    def test_invalid_index_type(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ m[1.5] → raises TypeError")
        with pytest.raises(TypeError):
            _ = m[1.5]


class TestMatrixAddition:
    """Test cases for Matrix addition operations."""

    def test_add_same_dimensions(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        print(
            f"\n✓ [[1,2],[3,4]] + [[5,6],[7,8]] = {result.data} (expected [[6,8],[10,12]])"
        )
        assert result.data == [[6, 8], [10, 12]]

    def test_add_3x3_matrices(self):
        m1 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m2 = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = m1 + m2
        print(
            f"\n✓ Identity + Ones = {result.data} (expected [[2,1,1],[1,2,1],[1,1,2]])"
        )
        assert result.data == [[2, 1, 1], [1, 2, 1], [1, 1, 2]]

    def test_add_different_dimensions(self):
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1], [2]])
        print(f"\n✓ (1x2) + (2x1) → raises ValueError")
        with pytest.raises(ValueError):
            _ = m1 + m2

    def test_add_non_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ Matrix + scalar → raises TypeError")
        with pytest.raises(TypeError):
            result = m + 5


class TestMatrixSubtraction:
    """Test cases for Matrix subtraction operations."""

    def test_subtract_same_dimensions(self):
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        result = m1 - m2
        print(
            f"\n✓ [[5,6],[7,8]] - [[1,2],[3,4]] = {result.data} (expected [[4,4],[4,4]])"
        )
        assert result.data == [[4, 4], [4, 4]]

    def test_subtract_identity(self):
        m = Matrix([[3, 4], [5, 6]])
        identity = Matrix([[1, 0], [0, 1]])
        result = m - identity
        print(f"\n✓ M - I = {result.data} (expected [[2,4],[5,5]])")
        assert result.data == [[2, 4], [5, 5]]

    def test_subtract_different_dimensions(self):
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1], [2]])
        print(f"\n✓ (1x2) - (2x1) → raises ValueError")
        with pytest.raises(ValueError):
            _ = m1 - m2


class TestMatrixMultiplication:
    """Test cases for Matrix-Matrix and Matrix-Vector multiplication."""

    def test_multiply_2x2_matrices(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 @ m2
        print(
            f"\n✓ [[1,2],[3,4]] * [[5,6],[7,8]] = {result.data} (expected [[19,22],[43,50]])"
        )
        assert result.data == [[19, 22], [43, 50]]

    def test_multiply_matrix_by_vector(self):
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([5, 6])
        result = m @ v
        print(f"\n✓ [[1,2],[3,4]] * [5,6] = {result.data} (expected [17,39])")
        assert isinstance(result, Vector)
        assert result.data == [17, 39]

    def test_multiply_3x2_by_2x3(self):
        m1 = Matrix([[1, 2], [3, 4], [5, 6]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        result = m1 @ m2
        print(f"\n✓ (3x2) * (2x3) → shape={result.shape}, data={result.data}")
        assert result.shape == (3, 3)
        assert result.data == [[9, 12, 15], [19, 26, 33], [29, 40, 51]]

    def test_multiply_incompatible_dimensions(self):
        m1 = Matrix([[1, 2, 3]])  # 1x3
        m2 = Matrix([[1, 2], [3, 4]])  # 2x2
        print(f"\n✓ (1x3) * (2x2) → raises ValueError (incompatible)")
        with pytest.raises(ValueError):
            _ = m1 @ m2

    def test_multiply_identity_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        identity = Matrix([[1, 0], [0, 1]])
        result = m @ identity
        print(f"\n✓ M * I = {result.data} (expected [[1,2],[3,4]])")
        assert result.data == [[1, 2], [3, 4]]

    def test_multiply_by_non_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ Matrix * scalar → raises TypeError")
        with pytest.raises(TypeError):
            result = m * 5


class TestMatrixScalarMultiplication:
    """Test cases for scalar multiplication with Matrices."""

    def test_scalar_multiply_integer(self):
        m = Matrix([[1, 2], [3, 4]])
        result = 3 * m
        print(f"\n✓ 3 * [[1,2],[3,4]] = {result.data} (expected [[3,6],[9,12]])")
        assert result.data == [[3, 6], [9, 12]]

    def test_scalar_multiply_float(self):
        m = Matrix([[2, 4], [6, 8]])
        result = 0.5 * m
        print(
            f"\n✓ 0.5 * [[2,4],[6,8]] = {result.data} (expected [[1.0,2.0],[3.0,4.0]])"
        )
        assert result.data == [[1.0, 2.0], [3.0, 4.0]]

    def test_scalar_multiply_zero(self):
        m = Matrix([[1, 2], [3, 4]])
        result = 0 * m
        print(f"\n✓ 0 * M = {result.data} (expected [[0,0],[0,0]])")
        assert result.data == [[0, 0], [0, 0]]

    def test_scalar_multiply_negative(self):
        m = Matrix([[1, 2], [3, 4]])
        result = -1 * m
        print(f"\n✓ -1 * M = {result.data} (expected [[-1,-2],[-3,-4]])")
        assert result.data == [[-1, -2], [-3, -4]]


class TestMatrixEquality:
    """Test cases for Matrix equality comparisons."""

    def test_equal_matrices(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        result = m1 == m2
        print(f"\n✓ M1 == M2 → {result} (expected True)")
        assert result == True

    def test_unequal_matrices_values(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        result = m1 == m2
        print(f"\n✓ [[1,2],[3,4]] == [[1,2],[3,5]] → {result} (expected False)")
        assert result == False

    def test_unequal_matrices_dimensions(self):
        m1 = Matrix([[1, 2]])
        m2 = Matrix([[1], [2]])
        result = m1 == m2
        print(f"\n✓ (1x2) == (2x1) → {result} (expected False)")
        assert result == False

    def test_equal_with_non_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m == [[1, 2], [3, 4]]
        print(f"\n✓ Matrix == list → {result} (expected False)")
        assert result == False


class TestMatrixTranspose:
    """Test cases for Matrix transpose operations."""

    def test_transpose_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        t = m.transpose()
        print(f"\n✓ Transpose([[1,2],[3,4]]) = {t.data} (expected [[1,3],[2,4]])")
        assert t.data == [[1, 3], [2, 4]]

    def test_transpose_3x2(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        t = m.transpose()
        print(f"\n✓ Transpose(3x2) = {t.data}, shape={t.shape} (expected (2,3))")
        assert t.data == [[1, 3, 5], [2, 4, 6]]
        assert t.shape == (2, 3)

    def test_transpose_property(self):
        m = Matrix([[1, 2], [3, 4]])
        t = m.T
        print(f"\n✓ M.T = {t.data} (expected [[1,3],[2,4]])")
        assert t.data == [[1, 3], [2, 4]]

    def test_transpose_twice(self):
        m = Matrix([[1, 2], [3, 4]])
        t = m.T.T
        print(f"\n✓ (M.T).T = {t.data} (expected [[1,2],[3,4]])")
        assert t.data == [[1, 2], [3, 4]]

    def test_transpose_row_vector(self):
        m = Matrix([[1, 2, 3]])
        t = m.T
        print(f"\n✓ Transpose(1x3) = {t.data}, shape={t.shape} (expected (3,1))")
        assert t.data == [[1], [2], [3]]
        assert t.shape == (3, 1)

    def test_transpose_column_vector(self):
        m = Matrix([[1], [2], [3]])
        t = m.T
        print(f"\n✓ Transpose(3x1) = {t.data}, shape={t.shape} (expected (1,3))")
        assert t.data == [[1, 2, 3]]
        assert t.shape == (1, 3)


class TestMatrixStringRepresentation:
    """Test cases for Matrix string representation."""

    def test_str_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        expected = "[[1, 2],\n [3, 4]]"
        result = str(m)
        print(f"\n✓ str(Matrix) =\n{result}")
        assert result == expected

    def test_str_single_row(self):
        m = Matrix([[1, 2, 3]])
        expected = "[[1, 2, 3]]"
        result = str(m)
        print(f"\n✓ str(row vector) = {result}")
        assert result == expected


class TestMatrixPower:
    """Test cases for Matrix exponentiation."""

    def test_power_zero(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m**0
        print(f"\n✓ M^0 = {result.data} (expected [[1,0],[0,1]])")
        assert result.data == [[1, 0], [0, 1]]

    def test_power_one(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m**1
        print(f"\n✓ M^1 = {result.data} (expected [[1,2],[3,4]])")
        assert result.data == [[1, 2], [3, 4]]

    def test_power_two(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m**2
        print(f"\n✓ M^2 = {result.data} (expected [[7,10],[15,22]])")
        assert result.data == [[7, 10], [15, 22]]

    def test_power_three(self):
        m = Matrix([[1, 1], [1, 0]])  # Fibonacci matrix
        result = m**3
        print(f"\n✓ Fibonacci^3 = {result.data} (expected [[3,2],[2,1]])")
        assert result.data == [[3, 2], [2, 1]]

    def test_power_non_square(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ Non-square matrix power → raises ValueError")
        with pytest.raises(ValueError):
            _ = m**2

    def test_power_negative(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ M^(-1) → raises ValueError")
        with pytest.raises(ValueError):
            _ = m**-1

    def test_power_non_integer(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ M^2.5 → raises TypeError")
        with pytest.raises(TypeError):
            _ = m**2.5


class TestMatrixNegation:
    """Test cases for Matrix negation."""

    def test_negate_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        result = -m
        print(f"\n✓ -[[1,2],[3,4]] = {result.data} (expected [[-1,-2],[-3,-4]])")
        assert result.data == [[-1, -2], [-3, -4]]

    def test_double_negation(self):
        m = Matrix([[1, 2], [3, 4]])
        result = -(-m)
        print(f"\n✓ -(-M) = {result.data} (expected [[1,2],[3,4]])")
        assert result.data == [[1, 2], [3, 4]]


class TestMatrixTrace:
    """Test cases for Matrix trace property."""

    def test_trace_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ trace([[1,2],[3,4]]) = {m.trace} (expected 5)")
        assert m.trace == 5

    def test_trace_3x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f"\n✓ trace(3x3) = {m.trace} (expected 15)")
        assert m.trace == 15

    def test_trace_identity(self):
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        print(f"\n✓ trace(I_3) = {m.trace} (expected 3)")
        assert m.trace == 3

    def test_trace_non_square(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ trace(non-square) → raises ValueError")
        with pytest.raises(ValueError):
            _ = m.trace


class TestMatrixDeterminant:
    """Test cases for Matrix determinant property."""

    def test_determinant_1x1(self):
        m = Matrix([[7]])
        print(f"\n✓ det([[7]]) = {m.determinant} (expected 7)")
        assert m.determinant == 7

    def test_determinant_1x1_negative(self):
        m = Matrix([[-3]])
        print(f"\n✓ det([[-3]]) = {m.determinant} (expected -3)")
        assert m.determinant == -3

    def test_determinant_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ det([[1,2],[3,4]]) = {m.determinant} (expected -2)")
        assert m.determinant == -2

    def test_determinant_2x2_positive(self):
        m = Matrix([[3, 1], [2, 4]])
        print(f"\n✓ det([[3,1],[2,4]]) = {m.determinant} (expected 10)")
        assert m.determinant == 10

    def test_determinant_2x2_zero(self):
        m = Matrix([[1, 2], [2, 4]])
        print(f"\n✓ det([[1,2],[2,4]]) = {m.determinant} (expected 0, singular matrix)")
        assert m.determinant == 0

    def test_determinant_2x2_identity(self):
        m = Matrix([[1, 0], [0, 1]])
        print(f"\n✓ det(I_2) = {m.determinant} (expected 1)")
        assert m.determinant == 1

    def test_determinant_3x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f"\n✓ det(3x3 singular) = {m.determinant} (expected 0)")
        assert m.determinant == 0

    def test_determinant_3x3_nonzero(self):
        m = Matrix([[1, 2, 0], [3, 4, 5], [6, 0, 7]])
        print(f"\n✓ det([[1,2,0],[3,4,5],[6,0,7]]) = {m.determinant} (expected 46)")
        assert m.determinant == 46

    def test_determinant_3x3_identity(self):
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        print(f"\n✓ det(I_3) = {m.determinant} (expected 1)")
        assert m.determinant == 1

    def test_determinant_4x4(self):
        m = Matrix(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]
        )
        print(f"\n✓ det(4x4 singular) = {m.determinant} (expected 0)")
        assert m.determinant == 0

    def test_determinant_4x4_nonzero(self):
        m = Matrix(
            [
                [2, 1, 0, 0],
                [1, 3, 1, 0],
                [0, 1, 4, 1],
                [0, 0, 1, 5],
            ]
        )
        print(f"\n✓ det(4x4 tridiagonal) = {m.determinant} (expected 85)")
        assert m.determinant == 85

    def test_determinant_upper_triangular(self):
        m = Matrix([[2, 3, 5], [0, 4, 7], [0, 0, 6]])
        print(
            f"\n✓ det(upper triangular) = {m.determinant} (expected 48, product of diagonal)"
        )
        assert m.determinant == 48

    def test_determinant_lower_triangular(self):
        m = Matrix([[3, 0, 0], [5, 2, 0], [1, 4, 6]])
        print(
            f"\n✓ det(lower triangular) = {m.determinant} (expected 36, product of diagonal)"
        )
        assert m.determinant == 36

    def test_determinant_with_zeros_on_first_row(self):
        m = Matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        print(f"\n✓ det(first row starts with zeros) = {m.determinant} (expected 0)")
        assert m.determinant == 0

    def test_determinant_with_all_zeros_first_row(self):
        m = Matrix([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
        print(f"\n✓ det(all-zero first row) = {m.determinant} (expected 0)")
        assert m.determinant == 0

    def test_determinant_non_square_raises(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ det(non-square) → raises ValueError")
        with pytest.raises(ValueError):
            _ = m.determinant

    def test_determinant_scalar_multiple(self):
        m = Matrix([[2, 4], [6, 8]])
        print(f"\n✓ det([[2,4],[6,8]]) = {m.determinant} (expected -8)")
        assert m.determinant == -8

    def test_determinant_negative_entries(self):
        m = Matrix([[-1, -2], [-3, -4]])
        print(f"\n✓ det([[-1,-2],[-3,-4]]) = {m.determinant} (expected -2)")
        assert m.determinant == -2

    def test_determinant_float_entries(self):
        m = Matrix([[1.5, 2.0], [3.0, 4.5]])
        print(f"\n✓ det([[1.5,2.0],[3.0,4.5]]) = {m.determinant} (expected 0.75)")
        assert abs(m.determinant - 0.75) < 1e-10

    def test_determinant_matches_ad_minus_bc(self):
        m = Matrix([[5, 7], [2, 3]])
        expected = (5 * 3) - (7 * 2)
        print(f"\n✓ det([[5,7],[2,3]]) = {m.determinant} (expected {expected})")
        assert m.determinant == expected

    def test_determinant_transpose_equal(self):
        m = Matrix([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
        print(f"\n✓ det(M) == det(M.T)")
        assert m.determinant == m.T.determinant

    def test_determinant_product_rule(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        print(f"\n✓ det(M1 @ M2) == det(M1) * det(M2)")
        assert abs((m1 @ m2).determinant - (m1.determinant * m2.determinant)) < 1e-10


class TestMatrixGetMethods:
    """Test cases for Matrix get_row, get_col, get_rows, get_cols methods."""

    def test_get_row(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        row = m.get_row(1)
        print(f"\n✓ get_row(1) = {row} (expected [4, 5, 6])")
        assert row == [4, 5, 6]

    def test_get_row_independence(self):
        m = Matrix([[1, 2], [3, 4]])
        row = m.get_row(0)
        row[0] = 99
        print(f"\n✓ get_row() returns copy: m[0][0] = {m[0][0]} (expected 1)")
        assert m[0][0] == 1

    def test_get_row_invalid_index(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ get_row(invalid) → raises IndexError")
        with pytest.raises(IndexError):
            _ = m.get_row(5)

    def test_get_col(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        col = m.get_col(1)
        print(f"\n✓ get_col(1) = {col} (expected [2, 5, 8])")
        assert col == [2, 5, 8]

    def test_get_col_invalid_index(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ get_col(invalid) → raises IndexError")
        with pytest.raises(IndexError):
            _ = m.get_col(5)

    def test_get_rows(self):
        m = Matrix([[1, 2], [3, 4]])
        rows = m.get_rows()
        print(f"\n✓ get_rows() = {rows} (expected [[1,2],[3,4]])")
        assert rows == [[1, 2], [3, 4]]

    def test_get_cols(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        cols = m.get_cols()
        print(f"\n✓ get_cols() = {cols} (expected [[1,4],[2,5],[3,6]])")
        assert cols == [[1, 4], [2, 5], [3, 6]]

    def test_get_cols_as_vectors(self):
        m = Matrix([[1, 2], [3, 4]])
        cols = m.get_cols(to_vector=True)
        print(f"\n✓ get_cols(to_vector=True) returns Vectors")
        assert all(isinstance(col, Vector) for col in cols)
        assert cols[0].to_list() == [1, 3]
        assert cols[1].to_list() == [2, 4]


class TestMatrixTransform:
    """Test cases for Matrix transform method."""

    def test_transform_vector(self):
        m = Matrix([[2, 0], [0, 3]])
        v = Vector([1, 1])
        result = m.transform(v)
        print(f"\n✓ transform([1,1]) = {result.data} (expected [2,3])")
        assert result.data == [2, 3]

    def test_transform_non_vector(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ transform(non-vector) → raises TypeError")
        with pytest.raises(TypeError):
            _ = m.transform([1, 2])


class TestMatrixIdentityProperties:
    """Test cases for left_identity and right_identity properties."""

    def test_left_identity_square(self):
        m = Matrix([[1, 2], [3, 4]])
        left_id = m.left_identity
        print(f"\n✓ left_identity of 2x2 = {left_id.data}, shape={left_id.shape}")
        assert left_id.shape == (2, 2)
        assert left_id.data == [[1, 0], [0, 1]]

    def test_left_identity_rectangular(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        left_id = m.left_identity
        print(f"\n✓ left_identity of 2x3 → shape={left_id.shape} (expected (2,2))")
        assert left_id.shape == (2, 2)

    def test_right_identity_square(self):
        m = Matrix([[1, 2], [3, 4]])
        right_id = m.right_identity
        print(f"\n✓ right_identity of 2x2 = {right_id.data}, shape={right_id.shape}")
        assert right_id.shape == (2, 2)
        assert right_id.data == [[1, 0], [0, 1]]

    def test_right_identity_rectangular(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        right_id = m.right_identity
        print(f"\n✓ right_identity of 2x3 → shape={right_id.shape} (expected (3,3))")
        assert right_id.shape == (3, 3)

    def test_left_identity_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m.left_identity @ m
        print(f"\n✓ I_left * M = M")
        assert result.data == m.data

    def test_right_identity_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m @ m.right_identity
        print(f"\n✓ M * I_right = M")
        assert result.data == m.data


class TestMatrixSubmatrix:
    """Test cases for the internal _get_submatrix method."""

    def test_submatrix_removes_first_row_first_col(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub = m._get_submatrix(0, 0)
        print(f"\n✓ submatrix(0,0) of 3x3 = {sub.data} (expected [[5,6],[8,9]])")
        assert sub.data == [[5, 6], [8, 9]]

    def test_submatrix_removes_middle_row_middle_col(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub = m._get_submatrix(1, 1)
        print(f"\n✓ submatrix(1,1) of 3x3 = {sub.data} (expected [[1,3],[7,9]])")
        assert sub.data == [[1, 3], [7, 9]]

    def test_submatrix_removes_last_row_last_col(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub = m._get_submatrix(2, 2)
        print(f"\n✓ submatrix(2,2) of 3x3 = {sub.data} (expected [[1,2],[4,5]])")
        assert sub.data == [[1, 2], [4, 5]]

    def test_submatrix_shape(self):
        m = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        sub = m._get_submatrix(0, 0)
        print(f"\n✓ submatrix of 4x4 → shape={sub.shape} (expected (3,3))")
        assert sub.shape == (3, 3)

    def test_submatrix_of_2x2(self):
        m = Matrix([[3, 7], [2, 5]])
        sub = m._get_submatrix(0, 1)
        print(f"\n✓ submatrix(0,1) of 2x2 = {sub.data} (expected [[2]])")
        assert sub.data == [[2]]


class TestMatrixConversions:
    """Test cases for Matrix conversion methods."""

    def test_to_list(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m.to_list()
        print(f"\n✓ to_list() = {result} (expected [[1,2],[3,4]])")
        assert result == [[1, 2], [3, 4]]

    def test_to_list_independence(self):
        m = Matrix([[1, 2], [3, 4]])
        lst = m.to_list()
        lst[0][0] = 99
        print(f"\n✓ to_list() returns copy: m[0][0] = {m[0][0]} (expected 1)")
        assert m[0][0] == 1


class TestMatrixCopy:
    """Test cases for Matrix copy method."""

    def test_copy_creates_new_object(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        print(f"\n✓ copy() creates new object: m1 is m2 = {m1 is m2} (expected False)")
        assert m1 is not m2
        assert m1.data == m2.data

    def test_copy_independence(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        m2[0][0] = 99
        print(
            f"\n✓ Modifying copy doesn't affect original: m1[0][0] = {m1[0][0]} (expected 1)"
        )
        assert m1[0][0] == 1
        assert m2[0][0] == 99


class TestMatrixIterator:
    """Test cases for Matrix iteration."""

    def test_iteration(self):
        m = Matrix([[1, 2], [3, 4]])
        result = [row for row in m]
        print(f"\n✓ Iterating over matrix: {result} (expected [[1,2],[3,4]])")
        assert result == [[1, 2], [3, 4]]

    def test_len(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        print(f"\n✓ len(Matrix) = {len(m)} (expected 3)")
        assert len(m) == 3


class TestMatrixProperties:
    """Test cases for Matrix property accessors."""

    def test_rows_property(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ m.rows = {m.rows} (expected 2)")
        assert m.rows == 2

    def test_cols_property(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ m.cols = {m.cols} (expected 3)")
        assert m.cols == 3

    def test_is_square_true(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ 2x2 is_square = {m.is_square} (expected True)")
        assert m.is_square == True

    def test_is_square_false(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        print(f"\n✓ 2x3 is_square = {m.is_square} (expected False)")
        assert m.is_square == False
