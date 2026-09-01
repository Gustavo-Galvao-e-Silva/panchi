import pytest

from panchi import (
    Matrix,
    Vector,
    VectorSpace,
    column_space,
    null_space,
    nullity,
    rank,
    row_space,
    span,
    standard_basis,
)


class TestSpan:
    def test_span_equals_vectorspace(self):
        vecs = [Vector([1, 0]), Vector([0, 1])]
        assert span(vecs) == VectorSpace(vecs)

    def test_span_rank(self):
        assert rank(span([Vector([1, 2]), Vector([2, 4])])) == 1


class TestStandardBasis:
    def test_dimensions(self):
        sb = standard_basis(3)
        assert sb.ambient_dims == 3
        assert rank(sb) == 3
        assert len(sb) == 3

    def test_is_the_identity_columns(self):
        assert standard_basis(2) == VectorSpace([Vector([1, 0]), Vector([0, 1])])


class TestColumnSpace:
    def test_generators_are_columns(self):
        m = Matrix([[1, 2], [3, 4]])
        assert column_space(m) == VectorSpace(m.col_vectors)

    def test_rank_matches_matrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert rank(column_space(m)) == rank(m)

    def test_rank_deficient(self):
        assert rank(column_space(Matrix([[1, 2], [3, 6]]))) == 1


class TestRowSpace:
    def test_rank_matches_matrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert rank(row_space(m)) == rank(m)

    def test_row_space_is_column_space_of_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert row_space(m) == column_space(m.transpose())


class TestNullSpace:
    def test_vectors_are_in_kernel(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        zero = Vector([0, 0])
        for v in null_space(A):
            assert A @ v == zero

    def test_rank_equals_nullity(self):
        A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert rank(null_space(A)) == nullity(A)

    def test_full_rank_gives_zero_subspace(self):
        ns = null_space(Matrix([[1, 0], [0, 1]]))
        assert rank(ns) == 0
        assert ns == VectorSpace([Vector([0, 0])])

    def test_single_free_variable(self):
        ns = null_space(Matrix([[1, 1], [1, 1]]))
        assert rank(ns) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
