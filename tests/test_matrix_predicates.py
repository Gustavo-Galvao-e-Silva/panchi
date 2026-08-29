import pytest

from panchi import (
    Matrix,
    Vector,
    VectorSpace,
    is_invertible,
    is_symmetric,
    nullity,
    rank,
)


class TestRankMatrix:
    """rank() on matrices."""

    def test_full_rank_square(self):
        assert rank(Matrix([[1, 2], [3, 4]])) == 2

    def test_rank_deficient_square(self):
        assert rank(Matrix([[1, 2], [2, 4]])) == 1

    def test_zero_matrix(self):
        assert rank(Matrix([[0, 0], [0, 0]])) == 0

    def test_wide_matrix(self):
        assert rank(Matrix([[1, 2, 3], [4, 5, 6]])) == 2

    def test_tall_matrix(self):
        assert rank(Matrix([[1, 2], [3, 4], [5, 6]])) == 2

    def test_single_row_matrix(self):
        # A 1-row matrix is reducible via ref (no RowScale), so rank works.
        assert rank(Matrix([[1, 2, 3]])) == 1

    def test_single_element(self):
        assert rank(Matrix([[7]])) == 1
        assert rank(Matrix([[0]])) == 0


class TestRankVectorSpace:
    """rank() on vector spaces — unified with the matrix rank."""

    def test_independent_generators(self):
        assert rank(VectorSpace([Vector([1, 0]), Vector([0, 1])])) == 2

    def test_dependent_generators(self):
        assert rank(VectorSpace([Vector([1, 2]), Vector([2, 4])])) == 1

    def test_single_vector(self):
        assert rank(VectorSpace([Vector([3, 1, 4])])) == 1

    def test_matches_column_space_rank(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert rank(VectorSpace(m.col_vectors)) == rank(m)


class TestNullity:
    """nullity() and the rank-nullity theorem."""

    def test_rank_nullity_theorem(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert rank(m) + nullity(m) == m.cols

    def test_full_rank_square_has_zero_nullity(self):
        assert nullity(Matrix([[1, 2], [3, 4]])) == 0

    def test_rank_deficient(self):
        assert nullity(Matrix([[1, 2], [2, 4]])) == 1


class TestIsInvertible:
    """is_invertible()."""

    def test_invertible(self):
        assert is_invertible(Matrix([[1, 2], [3, 4]])) is True

    def test_singular(self):
        assert is_invertible(Matrix([[1, 2], [2, 4]])) is False

    def test_non_square(self):
        assert is_invertible(Matrix([[1, 2, 3], [4, 5, 6]])) is False

    def test_identity(self):
        assert is_invertible(Matrix([[1, 0], [0, 1]])) is True


class TestIsSymmetric:
    """is_symmetric()."""

    def test_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [2, 1]])) is True

    def test_not_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [3, 4]])) is False

    def test_non_square(self):
        assert is_symmetric(Matrix([[1, 2, 3], [4, 5, 6]])) is False

    def test_diagonal_is_symmetric(self):
        assert is_symmetric(Matrix([[5, 0], [0, 3]])) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
