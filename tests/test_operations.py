import pytest
from math import pi, sqrt, isclose

import mathrix as mx

# ==================== OPERATIONS TESTS ====================


class TestIdentityMatrix:
    """Test cases for identity matrix creation."""

    def test_identity_2x2(self):
        I = mx.identity(2)
        print(f"\n✓ identity(2) = {I.data}")
        assert I.data == [[1, 0], [0, 1]]
        assert I.shape == (2, 2)

    def test_identity_3x3(self):
        I = mx.identity(3)
        print(f"\n✓ identity(3) = {I.data}")
        assert I.data == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_identity_1x1(self):
        I = mx.identity(1)
        print(f"\n✓ identity(1) = {I.data}")
        assert I.data == [[1]]

    def test_identity_invalid_type(self):
        print(f"\n✓ identity(2.5) → raises TypeError")
        with pytest.raises(TypeError):
            mx.identity(2.5)

    def test_identity_negative(self):
        print(f"\n✓ identity(-1) → raises ValueError")
        with pytest.raises(ValueError):
            mx.identity(-1)

    def test_identity_zero(self):
        print(f"\n✓ identity(0) → raises ValueError")
        with pytest.raises(ValueError):
            mx.identity(0)


class TestZeroMatrix:
    """Test cases for zero matrix creation."""

    def test_zero_matrix_2x3(self):
        Z = mx.zero_matrix(2, 3)
        print(f"\n✓ zero_matrix(2, 3) = {Z.data}")
        assert Z.data == [[0, 0, 0], [0, 0, 0]]
        assert Z.shape == (2, 3)

    def test_zero_matrix_square(self):
        Z = mx.zero_matrix(2, 2)
        print(f"\n✓ zero_matrix(2, 2) = {Z.data}")
        assert Z.data == [[0, 0], [0, 0]]

    def test_zero_matrix_invalid_type(self):
        print(f"\n✓ zero_matrix(2.5, 3) → raises TypeError")
        with pytest.raises(TypeError):
            mx.zero_matrix(2.5, 3)

    def test_zero_matrix_negative(self):
        print(f"\n✓ zero_matrix(-1, 2) → raises ValueError")
        with pytest.raises(ValueError):
            mx.zero_matrix(-1, 2)


class TestOneMatrix:
    """Test cases for one matrix creation."""

    def test_one_matrix_2x3(self):
        O = mx.one_matrix(2, 3)
        print(f"\n✓ one_matrix(2, 3) = {O.data}")
        assert O.data == [[1, 1, 1], [1, 1, 1]]
        assert O.shape == (2, 3)

    def test_one_matrix_square(self):
        O = mx.one_matrix(2, 2)
        print(f"\n✓ one_matrix(2, 2) = {O.data}")
        assert O.data == [[1, 1], [1, 1]]

    def test_one_matrix_invalid_type(self):
        print(f"\n✓ one_matrix(2, 3.5) → raises TypeError")
        with pytest.raises(TypeError):
            mx.one_matrix(2, 3.5)

    def test_one_matrix_zero(self):
        print(f"\n✓ one_matrix(0, 2) → raises ValueError")
        with pytest.raises(ValueError):
            mx.one_matrix(0, 2)


class TestZeroVector:
    """Test cases for zero vector creation."""

    def test_zero_vector_3d(self):
        z = mx.zero_vector(3)
        print(f"\n✓ zero_vector(3) = {z.data}")
        assert z.data == [0, 0, 0]
        assert z.dims == 3

    def test_zero_vector_1d(self):
        z = mx.zero_vector(1)
        print(f"\n✓ zero_vector(1) = {z.data}")
        assert z.data == [0]

    def test_zero_vector_invalid_type(self):
        print(f"\n✓ zero_vector(3.5) → raises TypeError")
        with pytest.raises(TypeError):
            mx.zero_vector(3.5)

    def test_zero_vector_negative(self):
        print(f"\n✓ zero_vector(-1) → raises ValueError")
        with pytest.raises(ValueError):
            mx.zero_vector(-1)


class TestOneVector:
    """Test cases for one vector creation."""

    def test_one_vector_3d(self):
        o = mx.one_vector(3)
        print(f"\n✓ one_vector(3) = {o.data}")
        assert o.data == [1, 1, 1]
        assert o.dims == 3

    def test_one_vector_1d(self):
        o = mx.one_vector(1)
        print(f"\n✓ one_vector(1) = {o.data}")
        assert o.data == [1]

    def test_one_vector_invalid_type(self):
        print(f"\n✓ one_vector('3') → raises TypeError")
        with pytest.raises(TypeError):
            mx.one_vector("3")


class TestUnitVector:
    """Test cases for unit vector creation."""

    def test_unit_vector_first(self):
        e0 = mx.unit_vector(3, 0)
        print(f"\n✓ unit_vector(3, 0) = {e0.data}")
        assert e0.data == [1, 0, 0]

    def test_unit_vector_middle(self):
        e1 = mx.unit_vector(3, 1)
        print(f"\n✓ unit_vector(3, 1) = {e1.data}")
        assert e1.data == [0, 1, 0]

    def test_unit_vector_last(self):
        e2 = mx.unit_vector(3, 2)
        print(f"\n✓ unit_vector(3, 2) = {e2.data}")
        assert e2.data == [0, 0, 1]

    def test_unit_vector_invalid_index(self):
        print(f"\n✓ unit_vector(3, 5) → raises ValueError")
        with pytest.raises(ValueError):
            mx.unit_vector(3, 5)

    def test_unit_vector_negative_index(self):
        print(f"\n✓ unit_vector(3, -1) → raises ValueError")
        with pytest.raises(ValueError):
            mx.unit_vector(3, -1)


class TestDiagonal:
    """Test cases for diagonal matrix creation."""

    def test_diagonal_from_list(self):
        D = mx.diagonal([1, 2, 3])
        print(f"\n✓ diagonal([1,2,3]) = {D.data}")
        assert D.data == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

    def test_diagonal_from_vector(self):
        v = mx.Vector([2, 3])
        D = mx.diagonal(v)
        print(f"\n✓ diagonal(Vector([2,3])) = {D.data}")
        assert D.data == [[2, 0], [0, 3]]


class TestRandomVector:
    """Test cases for random vector creation."""

    def test_random_vector_dimensions(self):
        v = mx.random_vector(5)
        print(f"\n✓ random_vector(5) has {v.dims} dimensions")
        assert v.dims == 5

    def test_random_vector_range(self):
        v = mx.random_vector(100, 0.0, 1.0)
        print(f"\n✓ random_vector range check")
        assert all(0.0 <= x <= 1.0 for x in v.data)

    def test_random_vector_custom_range(self):
        v = mx.random_vector(50, -5.0, 5.0)
        print(f"\n✓ random_vector custom range [-5, 5]")
        assert all(-5.0 <= x <= 5.0 for x in v.data)

    def test_random_vector_invalid_range(self):
        print(f"\n✓ random_vector with low >= high → raises ValueError")
        with pytest.raises(ValueError):
            mx.random_vector(3, 5.0, 1.0)


class TestRandomMatrix:
    """Test cases for random matrix creation."""

    def test_random_matrix_shape(self):
        M = mx.random_matrix(3, 4)
        print(f"\n✓ random_matrix(3, 4) shape = {M.shape}")
        assert M.shape == (3, 4)

    def test_random_matrix_range(self):
        M = mx.random_matrix(5, 5, 0.0, 1.0)
        print(f"\n✓ random_matrix range check")
        for row in M.data:
            assert all(0.0 <= x <= 1.0 for x in row)

    def test_random_matrix_invalid_range(self):
        print(f"\n✓ random_matrix with low >= high → raises ValueError")
        with pytest.raises(ValueError):
            mx.random_matrix(2, 2, 10.0, 1.0)


class TestRotationMatrix2D:
    """Test cases for 2D rotation matrix creation."""

    def test_rotation_90_degrees(self):
        R = mx.rotation_matrix_2d(pi / 2, radians=True)
        print(f"\n✓ rotation_matrix_2d(90°) ≈ [[0,-1],[1,0]]")
        assert isclose(R[0][0], 0, abs_tol=1e-10)
        assert isclose(R[0][1], -1, abs_tol=1e-10)
        assert isclose(R[1][0], 1, abs_tol=1e-10)
        assert isclose(R[1][1], 0, abs_tol=1e-10)

    def test_rotation_180_degrees(self):
        R = mx.rotation_matrix_2d(pi, radians=True)
        print(f"\n✓ rotation_matrix_2d(180°) ≈ [[-1,0],[0,-1]]")
        assert isclose(R[0][0], -1, abs_tol=1e-10)
        assert isclose(R[1][1], -1, abs_tol=1e-10)

    def test_rotation_degrees_mode(self):
        R = mx.rotation_matrix_2d(90, radians=False)
        print(f"\n✓ rotation_matrix_2d(90, radians=False)")
        assert isclose(R[0][0], 0, abs_tol=1e-10)
        assert isclose(R[0][1], -1, abs_tol=1e-10)

    def test_rotation_zero(self):
        R = mx.rotation_matrix_2d(0)
        print(f"\n✓ rotation_matrix_2d(0) = identity")
        assert isclose(R[0][0], 1, abs_tol=1e-10)
        assert isclose(R[1][1], 1, abs_tol=1e-10)
        assert isclose(R[0][1], 0, abs_tol=1e-10)
        assert isclose(R[1][0], 0, abs_tol=1e-10)


class TestRotationMatrix3D:
    """Test cases for 3D rotation matrix creation."""

    def test_rotation_z_axis(self):
        axis = mx.Vector([0, 0, 1])
        R = mx.rotation_matrix_3d(pi / 2, axis, radians=True)
        print(f"\n✓ rotation_matrix_3d(90° around z-axis)")
        assert R.shape == (3, 3)
        v = mx.Vector([1, 0, 0])
        result = R * v
        assert isclose(result[0], 0, abs_tol=1e-10)
        assert isclose(result[1], 1, abs_tol=1e-10)
        assert isclose(result[2], 0, abs_tol=1e-10)

    def test_rotation_zero_axis(self):
        axis = mx.Vector([0, 0, 0])
        print(f"\n✓ rotation with zero axis → raises ValueError")
        with pytest.raises(ValueError):
            mx.rotation_matrix_3d(pi / 2, axis)

    def test_rotation_degrees_mode_3d(self):
        axis = mx.Vector([0, 0, 1])
        R = mx.rotation_matrix_3d(90, axis, radians=False)
        print(f"\n✓ rotation_matrix_3d(90, radians=False)")
        assert R.shape == (3, 3)


class TestDotProduct:
    """Test cases for vector dot product."""

    def test_dot_orthogonal(self):
        v1 = mx.Vector([1, 0, 0])
        v2 = mx.Vector([0, 1, 0])
        result = mx.dot(v1, v2)
        print(f"\n✓ dot([1,0,0], [0,1,0]) = {result} (expected 0)")
        assert result == 0

    def test_dot_parallel(self):
        v1 = mx.Vector([1, 2, 3])
        v2 = mx.Vector([2, 4, 6])
        result = mx.dot(v1, v2)
        print(f"\n✓ dot([1,2,3], [2,4,6]) = {result} (expected 28)")
        assert result == 28

    def test_dot_self(self):
        v = mx.Vector([3, 4])
        result = mx.dot(v, v)
        print(f"\n✓ dot([3,4], [3,4]) = {result} (expected 25)")
        assert result == 25

    def test_dot_different_dimensions(self):
        v1 = mx.Vector([1, 2])
        v2 = mx.Vector([1, 2, 3])
        print(f"\n✓ dot with different dimensions → raises ValueError")
        with pytest.raises(ValueError):
            mx.dot(v1, v2)


class TestCrossProduct:
    """Test cases for vector cross product."""

    def test_cross_standard_basis(self):
        v1 = mx.Vector([1, 0, 0])
        v2 = mx.Vector([0, 1, 0])
        result = mx.cross(v1, v2)
        print(f"\n✓ cross([1,0,0], [0,1,0]) = {result.data} (expected [0,0,1])")
        assert result.data == [0, 0, 1]

    def test_cross_anticommutative(self):
        v1 = mx.Vector([1, 2, 3])
        v2 = mx.Vector([4, 5, 6])
        result1 = mx.cross(v1, v2)
        result2 = mx.cross(v2, v1)
        print(f"\n✓ cross(v1, v2) = -{result2.data}")
        assert result1.data == [-x for x in result2.data]

    def test_cross_parallel_vectors(self):
        v1 = mx.Vector([1, 2, 3])
        v2 = mx.Vector([2, 4, 6])
        result = mx.cross(v1, v2)
        print(f"\n✓ cross of parallel vectors = {result.data} (expected [0,0,0])")
        assert result.data == [0, 0, 0]

    def test_cross_non_3d_vectors(self):
        v1 = mx.Vector([1, 2])
        v2 = mx.Vector([3, 4])
        print(f"\n✓ cross with non-3D vectors → raises ValueError")
        with pytest.raises(ValueError):
            mx.cross(v1, v2)


# ==================== INTEGRATION TESTS ====================


class TestLinearAlgebraOperations:
    """
    Integration tests for combined operations involving both Vector and Matrix classes.
    These tests verify that operations work correctly when chained together and
    demonstrate real-world linear algebra use cases.
    """

    def test_vector_matrix_transformation(self):
        """Test applying a transformation matrix to a vector (rotation example)."""
        rotation = mx.Matrix([[0, -1], [1, 0]])
        v = mx.Vector([1, 0])
        result = rotation * v
        print(f"\n✓ Rotation(90°) * [1,0] = {result.data} (expected [0,1])")
        assert result.data == [0, 1]

    def test_matrix_chain_multiplication(self):
        """Test matrix multiplication resulting in a scalar (1x1 matrix)."""
        m1 = mx.Matrix([[1, 2]])
        m2 = mx.Matrix([[3], [4]])
        result = m1 @ m2
        print(f"\n✓ (1x2) @ (2x1) = {result.data} (expected [[11]])")
        assert result.data == [[11]]

    def test_combined_operations(self):
        """Test combining vector addition and scalar multiplication."""
        v1 = mx.Vector([1, 2, 3])
        v2 = mx.Vector([4, 5, 6])
        v_sum = v1 + v2
        v_scaled = 2 * v_sum
        print(f"\n✓ 2 * ([1,2,3] + [4,5,6]) = {v_scaled.data} (expected [10,14,18])")
        assert v_scaled.data == [10, 14, 18]

    def test_matrix_addition_and_multiplication(self):
        """Test combining matrix addition and multiplication operations."""
        m1 = mx.Matrix([[1, 0], [0, 1]])
        m2 = mx.Matrix([[1, 1], [1, 1]])
        m_sum = m1 + m2
        m_product = m_sum @ m_sum
        print(f"\n✓ (I + Ones)² = {m_product.data} (expected [[5,4],[4,5]])")
        assert m_product.data == [[5, 4], [4, 5]]

    def test_transpose_in_computation(self):
        """Test using transpose in matrix multiplication (outer product)."""
        m = mx.Matrix([[1, 2, 3]])
        mt = m.T
        result = mt @ m
        print(f"\n✓ (1x3).T @ (1x3) → shape={result.shape}, data={result.data}")
        assert result.shape == (3, 3)
        assert result.data == [[1, 2, 3], [2, 4, 6], [3, 6, 9]]


class TestVectorOperations:
    """
    Additional integration tests focused on vector operations.
    These complement the vector-specific tests with more complex scenarios.
    """

    def test_vector_linear_combination(self):
        """Test creating a linear combination of vectors."""
        v1 = mx.Vector([1, 0, 0])
        v2 = mx.Vector([0, 1, 0])
        v3 = mx.Vector([0, 0, 1])

        result = 2 * v1 + 3 * v2 + 4 * v3
        print(f"\n✓ 2*v1 + 3*v2 + 4*v3 = {result.data} (expected [2,3,4])")
        assert result.data == [2, 3, 4]

    def test_vector_negation_via_scalar(self):
        """Test vector negation using scalar multiplication."""
        v = mx.Vector([1, -2, 3])
        result = -1 * v
        print(f"\n✓ -1 * [1,-2,3] = {result.data} (expected [-1,2,-3])")
        assert result.data == [-1, 2, -3]

    def test_vector_operations_preserve_type(self):
        """Ensure vector operations return Vector instances."""
        v1 = mx.Vector([1, 2])
        v2 = mx.Vector([3, 4])

        assert isinstance(v1 + v2, mx.Vector)
        assert isinstance(v1 - v2, mx.Vector)
        assert isinstance(2 * v1, mx.Vector)
        assert isinstance(v1 / 2, mx.Vector)
        assert isinstance(-v1, mx.Vector)


class TestMatrixOperations:
    """
    Additional integration tests focused on matrix operations.
    These complement the matrix-specific tests with more complex scenarios.
    """

    def test_matrix_transformation_composition(self):
        """Test composing multiple transformation matrices."""
        scale = mx.Matrix([[2, 0], [0, 2]])
        rotate = mx.Matrix([[0, -1], [1, 0]])

        composed = rotate @ scale

        v = mx.Vector([1, 0])
        result = composed * v
        print(f"\n✓ (Rotate ∘ Scale) * [1,0] = {result.data} (expected [0,2])")
        assert result.data == [0, 2]

    def test_matrix_subtraction_to_zero(self):
        """Test that matrix minus itself gives zero matrix."""
        m = mx.Matrix([[1, 2, 3], [4, 5, 6]])
        zero = m - m
        print(f"\n✓ M - M = {zero.data} (expected all zeros)")
        assert zero.data == [[0, 0, 0], [0, 0, 0]]

    def test_matrix_operations_preserve_type(self):
        """Ensure matrix operations return appropriate types."""
        m1 = mx.Matrix([[1, 2], [3, 4]])
        m2 = mx.Matrix([[5, 6], [7, 8]])
        v = mx.Vector([1, 2])

        assert isinstance(m1 + m2, mx.Matrix)
        assert isinstance(m1 - m2, mx.Matrix)
        assert isinstance(m1 @ m2, mx.Matrix)
        assert isinstance(2 * m1, mx.Matrix)
        assert isinstance(-m1, mx.Matrix)
        assert isinstance(m1.T, mx.Matrix)
        assert isinstance(m1 * v, mx.Vector)


class TestMixedOperations:
    """
    Tests that verify interactions between Vector and Matrix classes.
    """

    def test_matrix_vector_multiple_applications(self):
        """Test applying the same matrix to multiple vectors."""
        m = mx.Matrix([[1, 2], [3, 4]])
        v1 = mx.Vector([1, 0])
        v2 = mx.Vector([0, 1])

        r1 = m * v1
        r2 = m * v2

        print(f"\n✓ M*v1 = {r1.data}, M*v2 = {r2.data}")
        assert r1.data == [1, 3]
        assert r2.data == [2, 4]

    def test_matrix_transform_vs_multiply(self):
        """Verify that matrix.transform(v) and matrix * v give same result."""
        m = mx.Matrix([[2, 3], [4, 5]])
        v = mx.Vector([1, 2])

        result_multiply = m * v
        result_transform = m.transform(v)

        print(
            f"\n✓ M*v = {result_multiply.data}, M.transform(v) = {result_transform.data}"
        )
        assert result_multiply.data == result_transform.data

    def test_identity_transformations(self):
        """Test that identity matrix acts as expected on vectors."""
        identity_2d = mx.identity(2)
        identity_3d = mx.identity(3)

        v2 = mx.Vector([3, 4])
        v3 = mx.Vector([1, 2, 3])

        assert (identity_2d * v2).data == v2.data
        assert (identity_3d * v3).data == v3.data
        print(f"\n✓ Identity matrices preserve vectors")


class TestOperationsIntegration:
    """
    Tests that combine operations.py functions with Vector and Matrix operations.
    """

    def test_unit_vectors_orthogonality(self):
        """Test that unit vectors are orthogonal."""
        e1 = mx.unit_vector(3, 0)
        e2 = mx.unit_vector(3, 1)
        e3 = mx.unit_vector(3, 2)

        assert mx.dot(e1, e2) == 0
        assert mx.dot(e2, e3) == 0
        assert mx.dot(e1, e3) == 0
        print(f"\n✓ Unit vectors are orthogonal")

    def test_rotation_preserves_magnitude(self):
        """Test that rotation preserves vector magnitude."""
        v = mx.Vector([3, 4])
        original_mag = v.magnitude

        R = mx.rotation_matrix_2d(pi / 4)
        rotated = R * v

        print(f"\n✓ Rotation preserves magnitude: {rotated.magnitude} ≈ {original_mag}")
        assert isclose(rotated.magnitude, original_mag, abs_tol=1e-10)

    def test_dot_product_with_operations(self):
        """Test dot product using operations functions."""
        v1 = mx.one_vector(3)
        v2 = mx.unit_vector(3, 0)

        result = mx.dot(v1, v2)
        print(f"\n✓ dot(ones, e_0) = {result} (expected 1)")
        assert result == 1

    def test_cross_product_orthogonality(self):
        """Test that cross product result is orthogonal to inputs."""
        v1 = mx.Vector([1, 2, 3])
        v2 = mx.Vector([4, 5, 6])
        result = mx.cross(v1, v2)

        assert mx.dot(result, v1) == 0
        assert mx.dot(result, v2) == 0
        print(f"\n✓ Cross product is orthogonal to inputs")


class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_empty_operations(self):
        """Test operations on empty vectors and matrices."""
        v_empty = mx.Vector([])
        m_empty = mx.Matrix([])

        result_v = v_empty + v_empty
        assert result_v.data == []
        print(f"\n✓ Empty vector + empty vector = []")

        result_m = m_empty + m_empty
        assert result_m.data == []
        print(f"\n✓ Empty matrix + empty matrix = []")

    def test_single_element_operations(self):
        """Test operations on single-element vectors and matrices."""
        v = mx.Vector([5])
        m = mx.Matrix([[3]])

        v_result = v + mx.Vector([2])
        assert v_result.data == [7]

        mv_result = m * v
        assert mv_result.data == [15]

        print(f"\n✓ Single element operations work correctly")

    def test_large_dimension_compatibility(self):
        """Test that dimension checking works for larger matrices."""
        m1 = mx.Matrix([[i + j for j in range(10)] for i in range(5)])
        m2 = mx.Matrix([[i + j for j in range(8)] for i in range(10)])

        result = m1 @ m2
        assert result.shape == (5, 8)
        print(f"\n✓ (5x10) @ (10x8) = (5x8) ✓")

        with pytest.raises(ValueError):
            _ = m1 + m2
        print(f"\n✓ (5x10) + (10x8) raises ValueError ✓")

    def test_zero_operations(self):
        """Test operations with zero vectors and matrices."""
        z_vec = mx.zero_vector(3)
        z_mat = mx.zero_matrix(2, 2)

        v = mx.Vector([1, 2, 3])
        m = mx.Matrix([[1, 2], [3, 4]])

        assert (v + z_vec).data == v.data
        assert (m + z_mat).data == m.data

        result = z_mat * mx.Vector([v[0], v[1]])
        assert result.data == [0, 0]

        print(f"\n✓ Zero vector/matrix operations work correctly")
