from math import isclose, pi

import pytest

import panchi as pan


class TestLinearAlgebraOperations:
    """
    Integration tests for combined operations involving both Vector and Matrix classes.
    These tests verify that operations work correctly when chained together and
    demonstrate real-world linear algebra use cases.
    """

    def test_vector_matrix_transformation(self):
        """Test applying a transformation matrix to a vector (rotation example)."""
        rotation = pan.Matrix([[0, -1], [1, 0]])
        v = pan.Vector([1, 0])
        result = rotation @ v
        print(f"\n✓ Rotation(90°) * [1,0] = {result.data} (expected [0,1])")
        assert result.data == [0, 1]

    def test_matrix_chain_multiplication(self):
        """Test matrix multiplication resulting in a scalar (1x1 matrix)."""
        m1 = pan.Matrix([[1, 2]])
        m2 = pan.Matrix([[3], [4]])
        result = m1 @ m2
        print(f"\n✓ (1x2) @ (2x1) = {result.data} (expected [[11]])")
        assert result.data == [[11]]

    def test_combined_operations(self):
        """Test combining vector addition and scalar multiplication."""
        v1 = pan.Vector([1, 2, 3])
        v2 = pan.Vector([4, 5, 6])
        v_sum = v1 + v2
        v_scaled = 2 * v_sum
        print(f"\n✓ 2 * ([1,2,3] + [4,5,6]) = {v_scaled.data} (expected [10,14,18])")
        assert v_scaled.data == [10, 14, 18]

    def test_matrix_addition_and_multiplication(self):
        """Test combining matrix addition and multiplication operations."""
        m1 = pan.Matrix([[1, 0], [0, 1]])
        m2 = pan.Matrix([[1, 1], [1, 1]])
        m_sum = m1 + m2
        m_product = m_sum @ m_sum
        print(f"\n✓ (I + Ones)² = {m_product.data} (expected [[5,4],[4,5]])")
        assert m_product.data == [[5, 4], [4, 5]]

    def test_transpose_in_computation(self):
        """Test using transpose in matrix multiplication (outer product)."""
        m = pan.Matrix([[1, 2, 3]])
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
        v1 = pan.Vector([1, 0, 0])
        v2 = pan.Vector([0, 1, 0])
        v3 = pan.Vector([0, 0, 1])

        result = 2 * v1 + 3 * v2 + 4 * v3
        print(f"\n✓ 2*v1 + 3*v2 + 4*v3 = {result.data} (expected [2,3,4])")
        assert result.data == [2, 3, 4]

    def test_vector_negation_via_scalar(self):
        """Test vector negation using scalar multiplication."""
        v = pan.Vector([1, -2, 3])
        result = -1 * v
        print(f"\n✓ -1 * [1,-2,3] = {result.data} (expected [-1,2,-3])")
        assert result.data == [-1, 2, -3]

    def test_vector_operations_preserve_type(self):
        """Ensure vector operations return Vector instances."""
        v1 = pan.Vector([1, 2])
        v2 = pan.Vector([3, 4])

        assert isinstance(v1 + v2, pan.Vector)
        assert isinstance(v1 - v2, pan.Vector)
        assert isinstance(2 * v1, pan.Vector)
        assert isinstance(v1 / 2, pan.Vector)
        assert isinstance(-v1, pan.Vector)


class TestMatrixOperations:
    """
    Additional integration tests focused on matrix operations.
    These complement the matrix-specific tests with more complex scenarios.
    """

    def test_matrix_transformation_composition(self):
        """Test composing multiple transformation matrices."""
        scale = pan.Matrix([[2, 0], [0, 2]])
        rotate = pan.Matrix([[0, -1], [1, 0]])

        composed = rotate @ scale

        v = pan.Vector([1, 0])
        result = composed @ v
        print(f"\n✓ (Rotate ∘ Scale) * [1,0] = {result.data} (expected [0,2])")
        assert result.data == [0, 2]

    def test_matrix_subtraction_to_zero(self):
        """Test that matrix minus itself gives zero matrix."""
        m = pan.Matrix([[1, 2, 3], [4, 5, 6]])
        zero = m - m
        print(f"\n✓ M - M = {zero.data} (expected all zeros)")
        assert zero.data == [[0, 0, 0], [0, 0, 0]]

    def test_matrix_operations_preserve_type(self):
        """Ensure matrix operations return appropriate types."""
        m1 = pan.Matrix([[1, 2], [3, 4]])
        m2 = pan.Matrix([[5, 6], [7, 8]])
        v = pan.Vector([1, 2])

        assert isinstance(m1 + m2, pan.Matrix)
        assert isinstance(m1 - m2, pan.Matrix)
        assert isinstance(m1 @ m2, pan.Matrix)
        assert isinstance(2 * m1, pan.Matrix)
        assert isinstance(-m1, pan.Matrix)
        assert isinstance(m1.T, pan.Matrix)
        assert isinstance(m1 @ v, pan.Vector)


class TestMixedOperations:
    """
    Tests that verify interactions between Vector and Matrix classes.
    """

    def test_matrix_vector_multiple_applications(self):
        """Test applying the same matrix to multiple vectors."""
        m = pan.Matrix([[1, 2], [3, 4]])
        v1 = pan.Vector([1, 0])
        v2 = pan.Vector([0, 1])

        r1 = m @ v1
        r2 = m @ v2

        print(f"\n✓ M*v1 = {r1.data}, M*v2 = {r2.data}")
        assert r1.data == [1, 3]
        assert r2.data == [2, 4]

    def test_matrix_transform_vs_multiply(self):
        """Verify that matrix.transform(v) and matrix * v give same result."""
        m = pan.Matrix([[2, 3], [4, 5]])
        v = pan.Vector([1, 2])

        result_multiply = m @ v
        result_transform = m.transform(v)

        print(
            f"\n✓ M*v = {result_multiply.data}, M.transform(v) = {result_transform.data}"
        )
        assert result_multiply.data == result_transform.data

    def test_identity_transformations(self):
        """Test that identity matrix acts as expected on vectors."""
        identity_2d = pan.identity(2)
        identity_3d = pan.identity(3)

        v2 = pan.Vector([3, 4])
        v3 = pan.Vector([1, 2, 3])

        assert (identity_2d @ v2).data == v2.data
        assert (identity_3d @ v3).data == v3.data
        print("\n✓ Identity matrices preserve vectors")


class TestOperationsIntegration:
    """
    Tests that combine operations.py functions with Vector and Matrix operations.
    """

    def test_unit_vectors_orthogonality(self):
        """Test that unit vectors are orthogonal."""
        e1 = pan.unit_vector(3, 0)
        e2 = pan.unit_vector(3, 1)
        e3 = pan.unit_vector(3, 2)

        assert pan.dot(e1, e2) == 0
        assert pan.dot(e2, e3) == 0
        assert pan.dot(e1, e3) == 0
        print("\n✓ Unit vectors are orthogonal")

    def test_rotation_preserves_magnitude(self):
        """Test that rotation preserves vector magnitude."""
        v = pan.Vector([3, 4])
        original_mag = v.magnitude

        R = pan.rotation_matrix_2d(pi / 4)
        rotated = R @ v

        print(f"\n✓ Rotation preserves magnitude: {rotated.magnitude} ≈ {original_mag}")
        assert isclose(rotated.magnitude, original_mag, abs_tol=1e-10)

    def test_dot_product_with_operations(self):
        """Test dot product using operations functions."""
        v1 = pan.one_vector(3)
        v2 = pan.unit_vector(3, 0)

        result = pan.dot(v1, v2)
        print(f"\n✓ dot(ones, e_0) = {result} (expected 1)")
        assert result == 1

    def test_cross_product_orthogonality(self):
        """Test that cross product result is orthogonal to inputs."""
        v1 = pan.Vector([1, 2, 3])
        v2 = pan.Vector([4, 5, 6])
        result = pan.cross(v1, v2)

        assert pan.dot(result, v1) == 0
        assert pan.dot(result, v2) == 0
        print("\n✓ Cross product is orthogonal to inputs")


class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_empty_operations(self):
        """Test operations on empty vectors and matrices."""
        v_empty = pan.Vector([])
        m_empty = pan.Matrix([])

        result_v = v_empty + v_empty
        assert result_v.data == []
        print("\n✓ Empty vector + empty vector = []")

        result_m = m_empty + m_empty
        assert result_m.data == []
        print("\n✓ Empty matrix + empty matrix = []")

    def test_single_element_operations(self):
        """Test operations on single-element vectors and matrices."""
        v = pan.Vector([5])
        m = pan.Matrix([[3]])

        v_result = v + pan.Vector([2])
        assert v_result.data == [7]

        mv_result = m @ v
        assert mv_result.data == [15]

        print("\n✓ Single element operations work correctly")

    def test_large_dimension_compatibility(self):
        """Test that dimension checking works for larger matrices."""
        m1 = pan.Matrix([[i + j for j in range(10)] for i in range(5)])
        m2 = pan.Matrix([[i + j for j in range(8)] for i in range(10)])

        result = m1 @ m2
        assert result.shape == (5, 8)
        print("\n✓ (5x10) @ (10x8) = (5x8) ✓")

        with pytest.raises(ValueError):
            _ = m1 + m2
        print("\n✓ (5x10) + (10x8) raises ValueError ✓")

    def test_zero_operations(self):
        """Test operations with zero vectors and matrices."""
        z_vec = pan.zero_vector(3)
        z_mat = pan.zero_matrix(2, 2)

        v = pan.Vector([1, 2, 3])
        m = pan.Matrix([[1, 2], [3, 4]])

        assert (v + z_vec).data == v.data
        assert (m + z_mat).data == m.data

        result = z_mat @ pan.Vector([v[0], v[1]])
        assert result.data == [0, 0]

        print("\n✓ Zero vector/matrix operations work correctly")
