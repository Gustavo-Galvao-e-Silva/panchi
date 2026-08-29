import pytest

from panchi.algorithms import InverseResult, Solution, determinant_lu, inverse, solve
from panchi.primitives import Matrix, Vector, identity

# ==================== INVERSE TESTS ====================


class TestInverseValidation:
    """Test type and value validation for inverse."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            inverse([[1, 2], [3, 4]])

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            inverse(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_singular_2x2_raises_value_error(self):
        with pytest.raises(ValueError):
            inverse(Matrix([[1, 2], [2, 4]]))

    def test_singular_3x3_raises_value_error(self):
        with pytest.raises(ValueError):
            inverse(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


class TestInverseResultObject:
    """Test InverseResult object structure."""

    def test_returns_inverse_result_instance(self):
        assert isinstance(inverse(Matrix([[1, 2], [3, 4]])), InverseResult)

    def test_original_is_preserved(self):
        m = Matrix([[1, 2], [3, 4]])
        assert inverse(m).original == m

    def test_does_not_modify_original(self):
        m = Matrix([[1, 2], [3, 4]])
        inverse(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_steps_are_recorded(self):
        result = inverse(Matrix([[1, 2], [3, 4]]))
        assert len(result.steps) > 0


class TestInverseCorrectness:
    """Test that A @ A^-1 == I for various inputs."""

    def test_2x2_satisfies_identity(self):
        m = Matrix([[1, 2], [3, 4]])
        result = inverse(m)
        product = m @ result.inverse
        n = m.rows
        for i in range(n):
            for j in range(n):
                assert product[i][j] == pytest.approx(1 if i == j else 0, abs=1e-9)

    def test_3x3_satisfies_identity(self):
        m = Matrix([[1, 2, 0], [3, 4, 5], [6, 0, 7]])
        result = inverse(m)
        product = m @ result.inverse
        n = m.rows
        for i in range(n):
            for j in range(n):
                assert product[i][j] == pytest.approx(1 if i == j else 0, abs=1e-9)

    def test_identity_inverse_is_identity(self):
        result = inverse(identity(3))
        n = 3
        for i in range(n):
            for j in range(n):
                assert result.inverse[i][j] == pytest.approx(
                    1 if i == j else 0, abs=1e-9
                )

    def test_diagonal_matrix_inverse(self):
        m = Matrix([[2, 0], [0, 4]])
        result = inverse(m)
        assert result.inverse[0][0] == pytest.approx(0.5, abs=1e-9)
        assert result.inverse[1][1] == pytest.approx(0.25, abs=1e-9)

    def test_inverse_of_inverse_is_original(self):
        m = Matrix([[1, 2], [3, 4]])
        inv = inverse(m).inverse
        inv_inv = inverse(inv).inverse
        n = m.rows
        for i in range(n):
            for j in range(n):
                assert inv_inv[i][j] == pytest.approx(m[i][j], abs=1e-9)


# ==================== DETERMINANT_LU TESTS ====================


class TestDeterminantLuValidation:
    """Test type and value validation for determinant_lu."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            determinant_lu([[1, 2], [3, 4]])

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            determinant_lu(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestDeterminantLuCorrectness:
    """Test determinant_lu produces correct values."""

    def test_2x2_known_value(self):
        assert determinant_lu(Matrix([[1, 2], [3, 4]])) == pytest.approx(-2, abs=1e-9)

    def test_2x2_identity(self):
        assert determinant_lu(identity(2)) == pytest.approx(1, abs=1e-9)

    def test_3x3_identity(self):
        assert determinant_lu(identity(3)) == pytest.approx(1, abs=1e-9)

    def test_singular_2x2_is_zero(self):
        assert determinant_lu(Matrix([[1, 2], [2, 4]])) == pytest.approx(0, abs=1e-9)

    def test_singular_3x3_is_zero(self):
        assert determinant_lu(
            Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ) == pytest.approx(0, abs=1e-9)

    def test_upper_triangular_is_diagonal_product(self):
        m = Matrix([[2, 3, 5], [0, 4, 7], [0, 0, 6]])
        assert determinant_lu(m) == pytest.approx(48, abs=1e-9)

    def test_single_swap_negates_sign(self):
        assert determinant_lu(Matrix([[0, 1], [1, 0]])) == pytest.approx(-1, abs=1e-9)

    def test_matches_cofactor_expansion(self):
        m = Matrix([[1, 2, 0], [3, 4, 5], [6, 0, 7]])
        assert determinant_lu(m) == pytest.approx(m.determinant, abs=1e-9)

    def test_product_rule(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        assert determinant_lu(m1 @ m2) == pytest.approx(
            determinant_lu(m1) * determinant_lu(m2), abs=1e-9
        )

    def test_transpose_has_same_determinant(self):
        m = Matrix([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
        assert determinant_lu(m) == pytest.approx(determinant_lu(m.T), abs=1e-9)


# ==================== SOLVE VALIDATION TESTS ====================


class TestSolveValidation:
    """Test type and value validation for solve."""

    def test_non_matrix_a_raises_type_error(self):
        with pytest.raises(TypeError):
            solve([[1, 2], [3, 4]], Vector([1, 2]))

    def test_non_vector_b_raises_type_error(self):
        with pytest.raises(TypeError):
            solve(Matrix([[1, 2], [3, 4]]), [1, 2])

    def test_mismatched_rows_raises_value_error(self):
        with pytest.raises(ValueError):
            solve(Matrix([[1, 2], [3, 4]]), Vector([1, 2, 3]))


# ==================== SOLVE RESULT OBJECT ====================


class TestSolveResultObject:
    """Test Solution object structure."""

    def test_returns_solution_instance(self):
        assert isinstance(solve(identity(2), Vector([3, 4])), Solution)

    def test_original_is_preserved(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        assert solve(A, b).original == A

    def test_target_is_preserved(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        assert solve(A, b).target == b

    def test_steps_are_recorded(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        assert len(solve(A, b).steps) > 0

    def test_does_not_modify_a(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        solve(A, b)
        assert A == Matrix([[1, 2], [3, 4]])

    def test_does_not_modify_b(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        solve(A, b)
        assert b == Vector([5, 6])


# ==================== SOLVE UNIQUE ====================


class TestSolveUnique:
    """Test solve correctly identifies and solves unique systems."""

    def test_status_is_unique(self):
        A = Matrix([[2, 1], [5, 3]])
        b = Vector([1, 2])
        assert solve(A, b).status == "unique"

    def test_solution_satisfies_ax_equals_b(self):
        A = Matrix([[2, 1], [5, 3]])
        b = Vector([1, 2])
        result = solve(A, b)
        product = A @ result.solution
        for i in range(b.dims):
            assert product[i] == pytest.approx(b[i], abs=1e-9)

    def test_identity_system(self):
        A = identity(3)
        b = Vector([1, 2, 3])
        result = solve(A, b)
        assert result.status == "unique"
        for i in range(b.dims):
            assert result.solution[i] == pytest.approx(b[i], abs=1e-9)

    def test_3x3_unique_solution_satisfies_ax_equals_b(self):
        A = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        b = Vector([1, 0, 0])
        result = solve(A, b)
        assert result.status == "unique"
        product = A @ result.solution
        for i in range(b.dims):
            assert product[i] == pytest.approx(b[i], abs=1e-9)

    def test_solution_is_vector(self):
        A = Matrix([[1, 2], [3, 4]])
        b = Vector([5, 6])
        assert isinstance(solve(A, b).solution, Vector)

    def test_solution_has_correct_length(self):
        A = Matrix([[2, 1], [5, 3]])
        b = Vector([1, 2])
        assert solve(A, b).solution.dims == A.cols

    def test_tall_full_rank_unique_solution_satisfies_ax_equals_b(self):
        A = Matrix([[1, 0], [0, 1], [1, 1]])
        b = Vector([1, 2, 3])
        result = solve(A, b)
        assert result.status == "unique"
        product = A @ result.solution
        for i in range(b.dims):
            assert product[i] == pytest.approx(b[i], abs=1e-9)

    def test_tall_full_rank_solution_length_equals_cols(self):
        A = Matrix([[1, 0], [0, 1], [1, 1]])
        b = Vector([1, 2, 3])
        assert solve(A, b).solution.dims == A.cols


# ==================== SOLVE INCONSISTENT ====================


class TestSolveInconsistent:
    """Test solve correctly identifies inconsistent systems."""

    def test_status_is_inconsistent(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Vector([1, 3])
        assert solve(A, b).status == "inconsistent"

    def test_solution_is_none(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Vector([1, 3])
        assert solve(A, b).solution is None

    def test_3x3_inconsistent(self):
        A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        b = Vector([1, 3, 6])
        assert solve(A, b).status == "inconsistent"

    def test_consistent_zero_rhs_is_not_inconsistent(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Vector([0, 0])
        assert solve(A, b).status != "inconsistent"


# ==================== SOLVE INFINITE ====================


class TestSolveInfinite:
    """Test solve correctly identifies underdetermined systems."""

    def test_status_is_infinite(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        assert solve(A, b).status == "infinite"

    def test_solution_is_none(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        assert solve(A, b).solution is None

    def test_rank_deficient_square_system(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Vector([1, 2])
        assert solve(A, b).status == "infinite"

    def test_zero_matrix_zero_rhs(self):
        A = Matrix([[0, 0], [0, 0]])
        b = Vector([0, 0])
        assert solve(A, b).status == "infinite"

    def test_particular_satisfies_system(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        result = solve(A, b)
        assert result.particular is not None
        assert A @ result.particular == b

    def test_null_space_vectors_in_kernel(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        result = solve(A, b)
        zero = Vector([0, 0])
        for v in result.null_space:
            assert A @ v == zero

    def test_null_space_dimension_equals_nullity(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        result = solve(A, b)
        assert result.null_space.dims == 1

    def test_rank_deficient_square(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Vector([1, 2])
        result = solve(A, b)
        assert result.particular is not None
        assert A @ result.particular == b
        assert result.null_space.dims == 1

    def test_homogeneous_particular_is_zero(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([0, 0])
        result = solve(A, b)
        zero = Vector([0, 0, 0])
        assert result.particular == zero

    def test_str_nonhomogeneous(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([7, 8])
        result = solve(A, b)
        s = str(result)
        assert "infinite" in s
        assert "x = " in s
        assert "t·" in s

    def test_str_homogeneous_omits_zero(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Vector([0, 0])
        result = solve(A, b)
        s = str(result)
        assert "x = t·" in s

    def test_two_free_variables(self):
        A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = Vector([9, 10])
        result = solve(A, b)
        assert result.null_space.dims == 2
        assert result.particular is not None
        assert A @ result.particular == b
        zero = Vector([0, 0])
        for v in result.null_space:
            assert A @ v == zero
        s = str(result)
        assert "s·" in s
        assert "t·" in s

    def test_three_free_variables(self):
        A = Matrix([[1, 0, 1, 0, 1]])
        b = Vector([1])
        result = solve(A, b)
        assert result.null_space.dims == 4
        s = str(result)
        assert "t1·" in s
        assert "t4·" in s


class TestSolveTolerance:
    """Test the optional tolerance flag on solve()."""

    def test_default_is_exact(self):
        # An approximately-singular matrix is full-rank under exact comparison.
        A = Matrix([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
        result = solve(A, Vector([0.0, 0.0]))
        assert result.status == "unique"
        assert result.null_space is None

    def test_tolerance_detects_near_singular(self):
        # With a tolerance, the tiny pivot is treated as zero → infinite solutions.
        A = Matrix([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
        result = solve(A, Vector([0.0, 0.0]), tolerance=1e-6)
        assert result.status == "infinite"
        assert result.null_space is not None
        assert result.null_space.dims == 1

    def test_tolerance_does_not_change_exact_systems(self):
        # A genuinely non-singular system solves the same with a small tolerance.
        A = Matrix([[2, 1], [5, 3]])
        b = Vector([1, 2])
        exact = solve(A, b)
        toleranced = solve(A, b, tolerance=1e-9)
        assert toleranced.status == exact.status == "unique"
        for i in range(b.dims):
            assert toleranced.solution[i] == pytest.approx(exact.solution[i], abs=1e-12)
