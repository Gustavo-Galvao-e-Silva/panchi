import pytest

from panchi import Vector, dot, gram_schmidt, vector_projection


class TestVectorProjection:
    """Test cases for orthogonal projection of one vector onto another."""

    def test_projection_onto_axis(self):
        v = Vector([2, 3])
        a = Vector([1, 0])
        assert vector_projection(v, a).to_list() == pytest.approx([2.0, 0.0])

    def test_projection_onto_non_unit_axis(self):
        # Projecting onto a non-unit vector must divide by (a . a), not |a|.
        v = Vector([1, 0])
        a = Vector([2, 0])
        assert vector_projection(v, a).to_list() == pytest.approx([1.0, 0.0])

    def test_orthogonal_projection_is_zero(self):
        v = Vector([0, 5])
        a = Vector([1, 0])
        assert vector_projection(v, a).to_list() == pytest.approx([0.0, 0.0])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            vector_projection(Vector([1, 2]), Vector([1, 2, 3]))


class TestGramSchmidt:
    """Test cases for Gram-Schmidt orthonormalization."""

    def test_result_is_orthonormal(self):
        q = gram_schmidt([Vector([1, 1, 0]), Vector([1, 0, 1]), Vector([0, 1, 1])])
        for i in range(len(q)):
            assert q[i].magnitude == pytest.approx(1.0, abs=1e-9)
            for j in range(i + 1, len(q)):
                assert dot(q[i], q[j]) == pytest.approx(0.0, abs=1e-9)

    def test_preserves_count(self):
        q = gram_schmidt([Vector([1, 1, 0]), Vector([1, 0, 1])])
        assert len(q) == 2

    def test_already_orthonormal_is_preserved(self):
        q = gram_schmidt([Vector([1, 0]), Vector([0, 1])])
        assert q[0].to_list() == pytest.approx([1.0, 0.0])
        assert q[1].to_list() == pytest.approx([0.0, 1.0])

    def test_single_vector_is_normalized(self):
        q = gram_schmidt([Vector([3, 4])])
        assert q[0].to_list() == pytest.approx([0.6, 0.8])

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            gram_schmidt([])

    def test_linearly_dependent_raises(self):
        with pytest.raises(ZeroDivisionError):
            gram_schmidt([Vector([1, 0]), Vector([2, 0])])

    def test_stays_exact_when_all_lengths_rational(self):
        from fractions import Fraction

        from panchi import exact_vector

        q = gram_schmidt([exact_vector([3, 4]), exact_vector([4, 3])])
        assert all(isinstance(x, Fraction) for vec in q for x in vec.data)
        assert q[0].to_list() == [Fraction(3, 5), Fraction(4, 5)]
