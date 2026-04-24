import pytest
from math import pi, isclose

import panchi as pan


class TestIdentityMatrix:
    """Test cases for identity matrix creation."""

    def test_identity_2x2(self):
        I = pan.identity(2)
        print(f"\n✓ identity(2) = {I.data}")
        assert I.data == [[1, 0], [0, 1]]
        assert I.shape == (2, 2)

    def test_identity_3x3(self):
        I = pan.identity(3)
        print(f"\n✓ identity(3) = {I.data}")
        assert I.data == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_identity_1x1(self):
        I = pan.identity(1)
        print(f"\n✓ identity(1) = {I.data}")
        assert I.data == [[1]]

    def test_identity_invalid_type(self):
        print(f"\n✓ identity(2.5) → raises TypeError")
        with pytest.raises(TypeError):
            pan.identity(2.5)

    def test_identity_negative(self):
        print(f"\n✓ identity(-1) → raises ValueError")
        with pytest.raises(ValueError):
            pan.identity(-1)

    def test_identity_zero(self):
        print(f"\n✓ identity(0) → raises ValueError")
        with pytest.raises(ValueError):
            pan.identity(0)


class TestZeroMatrix:
    """Test cases for zero matrix creation."""

    def test_zero_matrix_2x3(self):
        Z = pan.zero_matrix(2, 3)
        print(f"\n✓ zero_matrix(2, 3) = {Z.data}")
        assert Z.data == [[0, 0, 0], [0, 0, 0]]
        assert Z.shape == (2, 3)

    def test_zero_matrix_square(self):
        Z = pan.zero_matrix(2, 2)
        print(f"\n✓ zero_matrix(2, 2) = {Z.data}")
        assert Z.data == [[0, 0], [0, 0]]

    def test_zero_matrix_invalid_type(self):
        print(f"\n✓ zero_matrix(2.5, 3) → raises TypeError")
        with pytest.raises(TypeError):
            pan.zero_matrix(2.5, 3)

    def test_zero_matrix_negative(self):
        print(f"\n✓ zero_matrix(-1, 2) → raises ValueError")
        with pytest.raises(ValueError):
            pan.zero_matrix(-1, 2)


class TestOneMatrix:
    """Test cases for one matrix creation."""

    def test_one_matrix_2x3(self):
        O = pan.one_matrix(2, 3)
        print(f"\n✓ one_matrix(2, 3) = {O.data}")
        assert O.data == [[1, 1, 1], [1, 1, 1]]
        assert O.shape == (2, 3)

    def test_one_matrix_square(self):
        O = pan.one_matrix(2, 2)
        print(f"\n✓ one_matrix(2, 2) = {O.data}")
        assert O.data == [[1, 1], [1, 1]]

    def test_one_matrix_invalid_type(self):
        print(f"\n✓ one_matrix(2, 3.5) → raises TypeError")
        with pytest.raises(TypeError):
            pan.one_matrix(2, 3.5)

    def test_one_matrix_zero(self):
        print(f"\n✓ one_matrix(0, 2) → raises ValueError")
        with pytest.raises(ValueError):
            pan.one_matrix(0, 2)


class TestZeroVector:
    """Test cases for zero vector creation."""

    def test_zero_vector_3d(self):
        z = pan.zero_vector(3)
        print(f"\n✓ zero_vector(3) = {z.data}")
        assert z.data == [0, 0, 0]
        assert z.dims == 3

    def test_zero_vector_1d(self):
        z = pan.zero_vector(1)
        print(f"\n✓ zero_vector(1) = {z.data}")
        assert z.data == [0]

    def test_zero_vector_invalid_type(self):
        print(f"\n✓ zero_vector(3.5) → raises TypeError")
        with pytest.raises(TypeError):
            pan.zero_vector(3.5)

    def test_zero_vector_negative(self):
        print(f"\n✓ zero_vector(-1) → raises ValueError")
        with pytest.raises(ValueError):
            pan.zero_vector(-1)


class TestOneVector:
    """Test cases for one vector creation."""

    def test_one_vector_3d(self):
        o = pan.one_vector(3)
        print(f"\n✓ one_vector(3) = {o.data}")
        assert o.data == [1, 1, 1]
        assert o.dims == 3

    def test_one_vector_1d(self):
        o = pan.one_vector(1)
        print(f"\n✓ one_vector(1) = {o.data}")
        assert o.data == [1]

    def test_one_vector_invalid_type(self):
        print(f"\n✓ one_vector('3') → raises TypeError")
        with pytest.raises(TypeError):
            pan.one_vector("3")


class TestUnitVector:
    """Test cases for unit vector creation."""

    def test_unit_vector_first(self):
        e0 = pan.unit_vector(3, 0)
        print(f"\n✓ unit_vector(3, 0) = {e0.data}")
        assert e0.data == [1, 0, 0]

    def test_unit_vector_middle(self):
        e1 = pan.unit_vector(3, 1)
        print(f"\n✓ unit_vector(3, 1) = {e1.data}")
        assert e1.data == [0, 1, 0]

    def test_unit_vector_last(self):
        e2 = pan.unit_vector(3, 2)
        print(f"\n✓ unit_vector(3, 2) = {e2.data}")
        assert e2.data == [0, 0, 1]

    def test_unit_vector_invalid_index(self):
        print(f"\n✓ unit_vector(3, 5) → raises ValueError")
        with pytest.raises(ValueError):
            pan.unit_vector(3, 5)

    def test_unit_vector_negative_index(self):
        print(f"\n✓ unit_vector(3, -1) → raises ValueError")
        with pytest.raises(ValueError):
            pan.unit_vector(3, -1)


class TestDiagonal:
    """Test cases for diagonal matrix creation."""

    def test_diagonal_from_list(self):
        D = pan.diagonal([1, 2, 3])
        print(f"\n✓ diagonal([1,2,3]) = {D.data}")
        assert D.data == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

    def test_diagonal_from_vector(self):
        v = pan.Vector([2, 3])
        D = pan.diagonal(v)
        print(f"\n✓ diagonal(Vector([2,3])) = {D.data}")
        assert D.data == [[2, 0], [0, 3]]


class TestRandomVector:
    """Test cases for random vector creation."""

    def test_random_vector_dimensions(self):
        v = pan.random_vector(5)
        print(f"\n✓ random_vector(5) has {v.dims} dimensions")
        assert v.dims == 5

    def test_random_vector_range(self):
        v = pan.random_vector(100, 0.0, 1.0)
        print(f"\n✓ random_vector range check")
        assert all(0.0 <= x <= 1.0 for x in v.data)

    def test_random_vector_custom_range(self):
        v = pan.random_vector(50, -5.0, 5.0)
        print(f"\n✓ random_vector custom range [-5, 5]")
        assert all(-5.0 <= x <= 5.0 for x in v.data)

    def test_random_vector_invalid_range(self):
        print(f"\n✓ random_vector with low >= high → raises ValueError")
        with pytest.raises(ValueError):
            pan.random_vector(3, 5.0, 1.0)


class TestRandomMatrix:
    """Test cases for random matrix creation."""

    def test_random_matrix_shape(self):
        M = pan.random_matrix(3, 4)
        print(f"\n✓ random_matrix(3, 4) shape = {M.shape}")
        assert M.shape == (3, 4)

    def test_random_matrix_range(self):
        M = pan.random_matrix(5, 5, 0.0, 1.0)
        print(f"\n✓ random_matrix range check")
        for row in M.data:
            assert all(0.0 <= x <= 1.0 for x in row)

    def test_random_matrix_invalid_range(self):
        print(f"\n✓ random_matrix with low >= high → raises ValueError")
        with pytest.raises(ValueError):
            pan.random_matrix(2, 2, 10.0, 1.0)


class TestRotationMatrix2D:
    """Test cases for 2D rotation matrix creation."""

    def test_rotation_90_degrees(self):
        R = pan.rotation_matrix_2d(pi / 2, radians=True)
        print(f"\n✓ rotation_matrix_2d(90°) ≈ [[0,-1],[1,0]]")
        assert isclose(R[0][0], 0, abs_tol=1e-10)
        assert isclose(R[0][1], -1, abs_tol=1e-10)
        assert isclose(R[1][0], 1, abs_tol=1e-10)
        assert isclose(R[1][1], 0, abs_tol=1e-10)

    def test_rotation_180_degrees(self):
        R = pan.rotation_matrix_2d(pi, radians=True)
        print(f"\n✓ rotation_matrix_2d(180°) ≈ [[-1,0],[0,-1]]")
        assert isclose(R[0][0], -1, abs_tol=1e-10)
        assert isclose(R[1][1], -1, abs_tol=1e-10)

    def test_rotation_degrees_mode(self):
        R = pan.rotation_matrix_2d(90, radians=False)
        print(f"\n✓ rotation_matrix_2d(90, radians=False)")
        assert isclose(R[0][0], 0, abs_tol=1e-10)
        assert isclose(R[0][1], -1, abs_tol=1e-10)

    def test_rotation_zero(self):
        R = pan.rotation_matrix_2d(0)
        print(f"\n✓ rotation_matrix_2d(0) = identity")
        assert isclose(R[0][0], 1, abs_tol=1e-10)
        assert isclose(R[1][1], 1, abs_tol=1e-10)
        assert isclose(R[0][1], 0, abs_tol=1e-10)
        assert isclose(R[1][0], 0, abs_tol=1e-10)


class TestRotationMatrix3D:
    """Test cases for 3D rotation matrix creation."""

    def test_rotation_z_axis(self):
        axis = pan.Vector([0, 0, 1])
        R = pan.rotation_matrix_3d(pi / 2, axis, radians=True)
        print(f"\n✓ rotation_matrix_3d(90° around z-axis)")
        assert R.shape == (3, 3)
        v = pan.Vector([1, 0, 0])
        result = R @ v
        assert isclose(result[0], 0, abs_tol=1e-10)
        assert isclose(result[1], 1, abs_tol=1e-10)
        assert isclose(result[2], 0, abs_tol=1e-10)

    def test_rotation_zero_axis(self):
        axis = pan.Vector([0, 0, 0])
        print(f"\n✓ rotation with zero axis → raises ValueError")
        with pytest.raises(ValueError):
            pan.rotation_matrix_3d(pi / 2, axis)

    def test_rotation_degrees_mode_3d(self):
        axis = pan.Vector([0, 0, 1])
        R = pan.rotation_matrix_3d(90, axis, radians=False)
        print(f"\n✓ rotation_matrix_3d(90, radians=False)")
        assert R.shape == (3, 3)


class TestDotProduct:
    """Test cases for vector dot product."""

    def test_dot_orthogonal(self):
        v1 = pan.Vector([1, 0, 0])
        v2 = pan.Vector([0, 1, 0])
        result = pan.dot(v1, v2)
        print(f"\n✓ dot([1,0,0], [0,1,0]) = {result} (expected 0)")
        assert result == 0

    def test_dot_parallel(self):
        v1 = pan.Vector([1, 2, 3])
        v2 = pan.Vector([2, 4, 6])
        result = pan.dot(v1, v2)
        print(f"\n✓ dot([1,2,3], [2,4,6]) = {result} (expected 28)")
        assert result == 28

    def test_dot_self(self):
        v = pan.Vector([3, 4])
        result = pan.dot(v, v)
        print(f"\n✓ dot([3,4], [3,4]) = {result} (expected 25)")
        assert result == 25

    def test_dot_different_dimensions(self):
        v1 = pan.Vector([1, 2])
        v2 = pan.Vector([1, 2, 3])
        print(f"\n✓ dot with different dimensions → raises ValueError")
        with pytest.raises(ValueError):
            pan.dot(v1, v2)


class TestCrossProduct:
    """Test cases for vector cross product."""

    def test_cross_standard_basis(self):
        v1 = pan.Vector([1, 0, 0])
        v2 = pan.Vector([0, 1, 0])
        result = pan.cross(v1, v2)
        print(f"\n✓ cross([1,0,0], [0,1,0]) = {result.data} (expected [0,0,1])")
        assert result.data == [0, 0, 1]

    def test_cross_anticommutative(self):
        v1 = pan.Vector([1, 2, 3])
        v2 = pan.Vector([4, 5, 6])
        result1 = pan.cross(v1, v2)
        result2 = pan.cross(v2, v1)
        print(f"\n✓ cross(v1, v2) = -{result2.data}")
        assert result1.data == [-x for x in result2.data]

    def test_cross_parallel_vectors(self):
        v1 = pan.Vector([1, 2, 3])
        v2 = pan.Vector([2, 4, 6])
        result = pan.cross(v1, v2)
        print(f"\n✓ cross of parallel vectors = {result.data} (expected [0,0,0])")
        assert result.data == [0, 0, 0]

    def test_cross_non_3d_vectors(self):
        v1 = pan.Vector([1, 2])
        v2 = pan.Vector([3, 4])
        print(f"\n✓ cross with non-3D vectors → raises ValueError")
        with pytest.raises(ValueError):
            pan.cross(v1, v2)
