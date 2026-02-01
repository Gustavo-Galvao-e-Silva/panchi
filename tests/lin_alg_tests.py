import pytest
from vector import Vector
from matrix import Matrix  # Import your corrected Vector and Matrix classes


# ==================== VECTOR TESTS ====================

class TestVectorInitialization:
    def test_valid_integer_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector([1,2,3]) → data={v.data}, dim={v.dim}, shape={v.shape}")
        assert v.data == [1, 2, 3]
        assert v.dim == 3
        assert v.shape == (3, 1)

    def test_valid_float_vector(self):
        v = Vector([1.5, 2.7, 3.14])
        print(f"\n✓ Vector([1.5,2.7,3.14]) → data={v.data}, dim={v.dim}")
        assert v.data == [1.5, 2.7, 3.14]
        assert v.dim == 3

    def test_mixed_int_float_vector(self):
        v = Vector([1, 2.5, 3])
        print(f"\n✓ Vector([1,2.5,3]) → data={v.data}, dim={v.dim}")
        assert v.data == [1, 2.5, 3]
        assert v.dim == 3

    def test_empty_vector(self):
        v = Vector([])
        print(f"\n✓ Vector([]) → data={v.data}, dim={v.dim}")
        assert v.data == []
        assert v.dim == 0

    def test_single_element_vector(self):
        v = Vector([42])
        print(f"\n✓ Vector([42]) → data={v.data}, dim={v.dim}")
        assert v.data == [42]
        assert v.dim == 1

    def test_invalid_type_string(self):
        print(f"\n✓ Vector('not a list') → raises TypeError")
        with pytest.raises(TypeError):
            Vector("not a list")

    def test_invalid_type_in_list(self):
        print(f"\n✓ Vector([1,2,'three']) → raises TypeError (corrected validation)")
        with pytest.raises(TypeError):
            Vector([1, 2, "three"])


class TestVectorIndexing:
    def test_valid_indexing(self):
        v = Vector([10, 20, 30, 40])
        print(f"\n✓ v[0]={v[0]}, v[1]={v[1]}, v[3]={v[3]}")
        assert v[0] == 10
        assert v[1] == 20
        assert v[3] == 40

    def test_negative_indexing(self):
        v = Vector([10, 20, 30])
        print(f"\n✓ v[-1]={v[-1]}, v[-2]={v[-2]}")
        assert v[-1] == 30
        assert v[-2] == 20

    def test_invalid_index_type_float(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v[1.5] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v[1.5]

    def test_invalid_index_type_string(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v['0'] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v["0"]


class TestVectorAddition:
    def test_add_same_dimension(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        result = v1 + v2
        print(f"\n✓ [1,2,3] + [4,5,6] = {result.data} (expected [5,7,9])")
        assert result.data == [5, 7, 9]

    def test_add_floats(self):
        v1 = Vector([1.5, 2.5])
        v2 = Vector([0.5, 1.5])
        result = v1 + v2
        print(f"\n✓ [1.5,2.5] + [0.5,1.5] = {result.data} (expected [2.0,4.0])")
        assert result.data == [2.0, 4.0]

    def test_add_different_dimensions(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        print(f"\n✓ [1,2] + [1,2,3] → raises TypeError (dimension mismatch)")
        with pytest.raises(TypeError):
            _ = v1 + v2

    def test_add_non_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector + list → raises TypeError")
        with pytest.raises(TypeError):
            _ = v + [1, 2, 3]

    def test_add_zero_vector(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([0, 0, 0])
        result = v1 + v2
        print(f"\n✓ [1,2,3] + [0,0,0] = {result.data} (expected [1,2,3])")
        assert result.data == [1, 2, 3]


class TestVectorSubtraction:
    def test_subtract_same_dimension(self):
        v1 = Vector([5, 7, 9])
        v2 = Vector([1, 2, 3])
        result = v1 - v2
        print(f"\n✓ [5,7,9] - [1,2,3] = {result.data} (expected [4,5,6])")
        assert result.data == [4, 5, 6]

    def test_subtract_negative_result(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([2, 3, 4])
        result = v1 - v2
        print(f"\n✓ [1,2,3] - [2,3,4] = {result.data} (expected [-1,-1,-1])")
        assert result.data == [-1, -1, -1]

    def test_subtract_different_dimensions(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        print(f"\n✓ [1,2] - [1,2,3] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v1 - v2

    def test_subtract_non_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector - scalar → raises TypeError")
        with pytest.raises(TypeError):
            _ = v - 5


class TestVectorScalarMultiplication:
    def test_multiply_by_integer(self):
        v = Vector([1, 2, 3])
        result = 3 * v
        print(f"\n✓ 3 * [1,2,3] = {result.data} (expected [3,6,9])")
        assert result.data == [3, 6, 9]

    def test_multiply_by_float(self):
        v = Vector([2, 4, 6])
        result = 0.5 * v
        print(f"\n✓ 0.5 * [2,4,6] = {result.data} (expected [1.0,2.0,3.0])")
        assert result.data == [1.0, 2.0, 3.0]

    def test_multiply_by_zero(self):
        v = Vector([1, 2, 3])
        result = 0 * v
        print(f"\n✓ 0 * [1,2,3] = {result.data} (expected [0,0,0])")
        assert result.data == [0, 0, 0]

    def test_multiply_by_negative(self):
        v = Vector([1, 2, 3])
        result = -2 * v
        print(f"\n✓ -2 * [1,2,3] = {result.data} (expected [-2,-4,-6])")
        assert result.data == [-2, -4, -6]

    def test_multiply_by_non_scalar(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ 'string' * Vector → raises TypeError")
        with pytest.raises(TypeError):
            result = "string" * v


class TestVectorStringRepresentation:
    def test_str_representation(self):
        v = Vector([1, 2, 3])
        result = str(v)
        print(f"\n✓ str(Vector([1,2,3])) = '{result}' (expected '[1, 2, 3]')")
        assert result == "[1, 2, 3]"

    def test_str_empty_vector(self):
        v = Vector([])
        result = str(v)
        print(f"\n✓ str(Vector([])) = '{result}' (expected '[]')")
        assert result == "[]"


# ==================== MATRIX TESTS ====================

class TestMatrixInitialization:
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
    def test_valid_row_access(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        print(f"\n✓ m[0]={m[0]}, m[1]={m[1]}, m[2]={m[2]}")
        assert m[0] == [1, 2]
        assert m[1] == [3, 4]
        assert m[2] == [5, 6]

    def test_valid_element_access(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ m[0][0]={m[0][0]}, m[0][1]={m[0][1]}, m[1][0]={m[1][0]}, m[1][1]={m[1][1]}")
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
    def test_add_same_dimensions(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        print(f"\n✓ [[1,2],[3,4]] + [[5,6],[7,8]] = {result.data} (expected [[6,8],[10,12]])")
        assert result.data == [[6, 8], [10, 12]]

    def test_add_3x3_matrices(self):
        m1 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m2 = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = m1 + m2
        print(f"\n✓ Identity + Ones = {result.data} (expected [[2,1,1],[1,2,1],[1,1,2]])")
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
    def test_subtract_same_dimensions(self):
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        result = m1 - m2
        print(f"\n✓ [[5,6],[7,8]] - [[1,2],[3,4]] = {result.data} (expected [[4,4],[4,4]])")
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
    def test_multiply_2x2_matrices(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 * m2
        print(f"\n✓ [[1,2],[3,4]] * [[5,6],[7,8]] = {result.data} (expected [[19,22],[43,50]])")
        assert result.data == [[19, 22], [43, 50]]

    def test_multiply_matrix_by_vector(self):
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([5, 6])
        result = m * v
        print(f"\n✓ [[1,2],[3,4]] * [5,6] = {result.data} (expected [17,39])")
        assert isinstance(result, Vector)
        assert result.data == [17, 39]

    def test_multiply_3x2_by_2x3(self):
        m1 = Matrix([[1, 2], [3, 4], [5, 6]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        result = m1 * m2
        print(f"\n✓ (3x2) * (2x3) → shape={result.shape}, data={result.data}")
        assert result.shape == (3, 3)
        assert result.data == [[9, 12, 15], [19, 26, 33], [29, 40, 51]]

    def test_multiply_incompatible_dimensions(self):
        m1 = Matrix([[1, 2, 3]])  # 1x3
        m2 = Matrix([[1, 2], [3, 4]])  # 2x2
        print(f"\n✓ (1x3) * (2x2) → raises ValueError (incompatible)")
        with pytest.raises(ValueError):
            _ = m1 * m2

    def test_multiply_identity_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        identity = Matrix([[1, 0], [0, 1]])
        result = m * identity
        print(f"\n✓ M * I = {result.data} (expected [[1,2],[3,4]])")
        assert result.data == [[1, 2], [3, 4]]

    def test_multiply_by_non_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        print(f"\n✓ Matrix * scalar → raises TypeError")
        with pytest.raises(TypeError):
            result = m * 5


class TestMatrixScalarMultiplication:
    def test_scalar_multiply_integer(self):
        m = Matrix([[1, 2], [3, 4]])
        result = 3 * m
        print(f"\n✓ 3 * [[1,2],[3,4]] = {result.data} (expected [[3,6],[9,12]])")
        assert result.data == [[3, 6], [9, 12]]

    def test_scalar_multiply_float(self):
        m = Matrix([[2, 4], [6, 8]])
        result = 0.5 * m
        print(f"\n✓ 0.5 * [[2,4],[6,8]] = {result.data} (expected [[1.0,2.0],[3.0,4.0]])")
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


class TestMatrixApply:
    def test_apply_to_vector(self):
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([5, 6])
        result = m.apply(v)
        print(f"\n✓ M.apply([5,6]) = {result.data} (expected [17,39])")
        assert isinstance(result, Vector)
        assert result.data == [17, 39]

    def test_apply_identity(self):
        identity = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v = Vector([3, 4, 5])
        result = identity.apply(v)
        print(f"\n✓ I.apply([3,4,5]) = {result.data} (expected [3,4,5])")
        assert result.data == [3, 4, 5]


class TestMatrixStringRepresentation:
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


# ==================== INTEGRATION TESTS ====================

class TestLinearAlgebraOperations:
    def test_vector_matrix_transformation(self):
        # Rotation matrix (90 degrees counterclockwise)
        rotation = Matrix([[0, -1], [1, 0]])
        v = Vector([1, 0])
        result = rotation * v
        print(f"\n✓ Rotation(90°) * [1,0] = {result.data} (expected [0,1])")
        assert result.data == [0, 1]

    def test_matrix_chain_multiplication(self):
        m1 = Matrix([[1, 2]])  # 1x2
        m2 = Matrix([[3], [4]])  # 2x1
        result = m1 * m2
        print(f"\n✓ (1x2) * (2x1) = {result.data} (expected [[11]])")
        assert result.data == [[11]]

    def test_combined_operations(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        v_sum = v1 + v2
        v_scaled = 2 * v_sum
        print(f"\n✓ 2 * ([1,2,3] + [4,5,6]) = {v_scaled.data} (expected [10,14,18])")
        assert v_scaled.data == [10, 14, 18]

    def test_matrix_addition_and_multiplication(self):
        m1 = Matrix([[1, 0], [0, 1]])
        m2 = Matrix([[1, 1], [1, 1]])
        m_sum = m1 + m2
        m_product = m_sum * m_sum
        print(f"\n✓ (I + Ones)² = {m_product.data} (expected [[5,4],[4,5]])")
        assert m_product.data == [[5, 4], [4, 5]]

    def test_transpose_in_computation(self):
        m = Matrix([[1, 2, 3]])  # 1x3
        mt = m.T  # 3x1
        result = mt * m  # 3x1 * 1x3 = 3x3
        print(f"\n✓ (1x3).T * (1x3) → shape={result.shape}, data={result.data}")
        assert result.shape == (3, 3)
        assert result.data == [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
