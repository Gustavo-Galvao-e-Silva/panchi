import pytest

from panchi import (
    Vector,
    VectorSpace,
    basis,
    dot,
    orthogonal_complement,
    rank,
    same_subspace,
)


class TestOrthogonalComplement:
    """Test cases for orthogonal complement computation."""

    def test_xy_plane_complement_in_r3(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 1
        assert basis(comp)[0].to_list() == [0, 0, 1]

    def test_single_vector_in_r3(self):
        vs = VectorSpace([Vector([1, 0, 0])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 2

    def test_full_space_trivial_complement(self):
        vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 0

    def test_line_in_r2(self):
        vs = VectorSpace([Vector([1, 1])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 1
        basis_vec = basis(comp)[0]
        assert dot(Vector([1, 1]), basis_vec) == 0

    def test_rank_nullity_theorem(self):
        vs = VectorSpace([Vector([1, 2, 3]), Vector([4, 5, 6])])
        comp = orthogonal_complement(vs)
        assert rank(comp) + rank(vs) == vs.ambient_dims

    def test_orthogonality_of_complement(self):
        vs = VectorSpace([Vector([1, 2, 3]), Vector([4, 5, 6])])
        comp = orthogonal_complement(vs)
        for b_orig in basis(vs):
            for b_comp in basis(comp):
                assert abs(dot(b_orig, b_comp)) < 1e-10

    def test_non_standard_basis(self):
        vs = VectorSpace([Vector([1, 1, 0]), Vector([0, 1, 1])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 1
        for b in basis(vs):
            assert abs(dot(b, basis(comp)[0])) < 1e-10

    def test_single_vector_in_r4(self):
        vs = VectorSpace([Vector([1, 0, 0, 0])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 3
        assert rank(comp) + rank(vs) == 4

    def test_complement_of_complement(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        comp = orthogonal_complement(vs)
        comp_comp = orthogonal_complement(comp)
        assert same_subspace(vs, comp_comp)

    def test_type_error_non_vector_space(self):
        with pytest.raises(TypeError):
            orthogonal_complement("not a space")

    def test_type_error_matrix(self):
        from panchi import Matrix

        with pytest.raises(TypeError):
            orthogonal_complement(Matrix([[1, 0], [0, 1]]))

    def test_redundant_generators(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
        comp = orthogonal_complement(vs)
        assert rank(comp) == 1
        assert basis(comp)[0].to_list() == [0, 0, 1]
