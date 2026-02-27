from panchi.primitives.matrix import Matrix
from panchi.primitives.factories import identity
from panchi.algorithms.decompositions import lu
from panchi.algorithms.results import LUDecomposition

# ==================== LU RESULT OBJECT TESTS ====================


class TestLUDecompositionProperties:
    """Test the LUDecomposition result object's structure and attributes."""

    def test_original_is_preserved(self):
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).original == m

    def test_result_is_lu_decomposition_instance(self):
        m = Matrix([[2, 1], [4, 3]])
        assert isinstance(lu(m), LUDecomposition)

    def test_lower_has_correct_shape(self):
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).lower.shape == (2, 2)

    def test_upper_has_correct_shape(self):
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).upper.shape == (2, 2)

    def test_permutation_has_correct_shape(self):
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).permutation.shape == (2, 2)

    def test_lower_has_ones_on_diagonal(self):
        m = Matrix([[2, 1], [4, 3]])
        lower = lu(m).lower
        for i in range(lower.rows):
            assert lower[i][i] == 1

    def test_lower_is_lower_triangular(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        lower = lu(m).lower
        for i in range(lower.rows):
            for j in range(i + 1, lower.cols):
                assert lower[i][j] == 0

    def test_upper_is_upper_triangular(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        upper = lu(m).upper
        for i in range(1, upper.rows):
            for j in range(i):
                assert upper[i][j] == 0

    def test_permutation_is_a_valid_permutation_matrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        p = lu(m).permutation
        assert p @ p.T == identity(3)

    def test_does_not_modify_original_matrix(self):
        m = Matrix([[2, 1], [4, 3]])
        lu(m)
        assert m == Matrix([[2, 1], [4, 3]])


# ==================== LU FACTORISATION INVARIANT ====================


class TestLUFactorisationInvariant:
    """Test that P @ A == L @ U holds for all inputs."""

    def test_2x2_no_pivoting_needed(self):
        m = Matrix([[2, 1], [4, 3]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_2x2_requires_swap(self):
        m = Matrix([[0, 1], [1, 0]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_3x3_full_rank(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_3x3_requires_partial_pivoting(self):
        m = Matrix([[0, 1, 2], [3, 4, 5], [6, 7, 9]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_identity_matrix(self):
        m = identity(3)
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_diagonal_matrix(self):
        from panchi.primitives.factories import diagonal

        m = diagonal([2, 3, 4])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_matrix_with_negative_entries(self):
        m = Matrix([[1, -2], [-3, 4]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_matrix_with_float_entries(self):
        m = Matrix([[1.5, 2.5], [3.5, 4.5]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper


# ==================== LU EXACT RESULTS ====================


class TestLUExactResults:
    """Test exact L, U, and P values for hand-verified inputs."""

    def test_2x2_no_swap_lower(self):
        # A = [[2,1],[4,3]], no swap needed
        # multiplier = 4/2 = 2, so L[1][0] = 2
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).lower == Matrix([[1, 0], [2, 1]])

    def test_2x2_no_swap_upper(self):
        # U = [[2,1],[0,1]] after R1 -> R1 + (-2)*R0
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).upper == Matrix([[2, 1], [0, 1]])

    def test_2x2_no_swap_permutation_is_identity(self):
        m = Matrix([[2, 1], [4, 3]])
        assert lu(m).permutation == identity(2)

    def test_2x2_swap_required_permutation(self):
        # A = [[0,1],[1,2]], pivot in col 0 is 0, must swap with R1
        m = Matrix([[0, 1], [1, 2]])
        assert lu(m).permutation == Matrix([[0, 1], [1, 0]])

    def test_2x2_swap_required_upper(self):
        # After swap: [[1,2],[0,1]]
        m = Matrix([[0, 1], [1, 2]])
        assert lu(m).upper == Matrix([[1, 2], [0, 1]])

    def test_2x2_swap_required_lower_is_identity(self):
        # No elimination needed after swap since lower entry is already 0
        m = Matrix([[0, 1], [1, 2]])
        assert lu(m).lower == identity(2)

    def test_3x3_full_rank_upper(self):
        # A = [[1,2,3],[4,5,6],[7,8,10]]
        # R1 -> R1 + (-4)*R0: [[1,2,3],[0,-3,-6],[7,8,10]]
        # R2 -> R2 + (-7)*R0: [[1,2,3],[0,-3,-6],[0,-6,-11]]
        # R2 -> R2 + (-2)*R1: [[1,2,3],[0,-3,-6],[0,0,1]]
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        assert lu(m).upper == Matrix([[1, 2, 3], [0, -3, -6], [0, 0, 1]])

    def test_3x3_full_rank_lower(self):
        # L encodes the multipliers: L[1][0]=4, L[2][0]=7, L[2][1]=2
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        assert lu(m).lower == Matrix([[1, 0, 0], [4, 1, 0], [7, 2, 1]])


# ==================== LU RANK DEFICIENT ====================


class TestLURankDeficient:
    """Test LU decomposition on singular and rank-deficient matrices."""

    def test_singular_matrix_factorisation_invariant_still_holds(self):
        m = Matrix([[1, 2], [2, 4]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper

    def test_singular_matrix_upper_has_zero_row(self):
        m = Matrix([[1, 2], [2, 4]])
        upper = lu(m).upper
        assert upper[1][0] == 0
        assert upper[1][1] == 0

    def test_zero_matrix_factorisation_invariant_still_holds(self):
        m = Matrix([[0, 0], [0, 0]])
        decomp = lu(m)
        assert decomp.permutation @ m == decomp.lower @ decomp.upper


# ==================== LU STRING REPRESENTATIONS ====================


class TestLUStringRepresentations:
    """Test __str__ and __repr__ on LUDecomposition."""

    def test_str_contains_form_label(self):
        assert "LU decomposition" in str(lu(Matrix([[2, 1], [4, 3]])))

    def test_str_contains_equation_label(self):
        assert "P @ A = L @ U" in str(lu(Matrix([[2, 1], [4, 3]])))

    def test_str_contains_shape(self):
        assert "2×2" in str(lu(Matrix([[2, 1], [4, 3]])))

    def test_repr_contains_shape(self):
        assert "2×2" in repr(lu(Matrix([[2, 1], [4, 3]])))

    def test_repr_format(self):
        assert repr(lu(Matrix([[2, 1], [4, 3]]))).startswith("LUDecomposition(")
