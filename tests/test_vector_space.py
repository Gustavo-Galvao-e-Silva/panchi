import pytest

from panchi.algorithms.matrix_operations import rank
from panchi.algorithms.vector_space_operations import (
    basis,
    contains,
    is_full_rank,
    same_subspace,
)
from panchi.primitives import Vector
from panchi.primitives.vector_space import VectorSpace

# ==================== VECTOR SPACE TESTS ====================


class TestVectorSpaceInitialization:
    """Test cases for VectorSpace object initialization and validation."""

    def test_valid_single_vector(self):
        v = Vector([1, 2, 3])
        vs = VectorSpace([v])
        print(f"\n✓ VectorSpace([Vector([1,2,3])]) → data={vs.data}")
        assert vs.data == [v]

    def test_valid_two_independent_vectors(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ VectorSpace([e1, e2]) → data has {len(vs.data)} vectors")
        assert len(vs.data) == 2

    def test_valid_with_dependent_vector(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v3 = Vector([1, 1])  # linearly dependent on v1 and v2
        vs = VectorSpace([v1, v2, v3])
        print(
            f"\n✓ VectorSpace([e1, e2, e1+e2]) → data has {len(vs.data)} vectors (all 3 stored)"
        )
        assert len(vs.data) == 3

    def test_valid_float_vectors(self):
        v1 = Vector([1.0, 0.0])
        v2 = Vector([0.0, 1.0])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ VectorSpace with float vectors → {len(vs.data)} vectors stored")
        assert len(vs.data) == 2

    def test_invalid_not_a_list(self):
        print("\n✓ VectorSpace(Vector([1,2])) → raises TypeError")
        with pytest.raises(TypeError):
            VectorSpace(Vector([1, 2]))

    def test_invalid_empty_list(self):
        print("\n✓ VectorSpace([]) → raises ValueError")
        with pytest.raises(ValueError):
            VectorSpace([])

    def test_invalid_non_vector_element(self):
        print("\n✓ VectorSpace([[1,2,3]]) → raises TypeError (list not Vector)")
        with pytest.raises(TypeError):
            VectorSpace([[1, 2, 3]])

    def test_invalid_mixed_elements(self):
        v = Vector([1, 2])
        print("\n✓ VectorSpace([Vector, 5]) → raises TypeError")
        with pytest.raises(TypeError):
            VectorSpace([v, 5])

    def test_invalid_inconsistent_dimensions(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        print("\n✓ VectorSpace([2d, 3d]) → raises ValueError (dimension mismatch)")
        with pytest.raises(ValueError):
            VectorSpace([v1, v2])

    def test_invalid_inconsistent_dimensions_first_matches(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = Vector([1, 1])  # wrong dims
        print("\n✓ VectorSpace([3d, 3d, 2d]) → raises ValueError")
        with pytest.raises(ValueError):
            VectorSpace([v1, v2, v3])


class TestVectorSpaceBasis:
    """Test cases for VectorSpace basis computation."""

    def test_basis_all_independent(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        computed = basis(vs)
        print(f"\n✓ basis([e1, e2]) → {len(computed)} vectors (expected 2)")
        assert len(computed) == 2

    def test_basis_with_one_dependent(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v3 = Vector([1, 1])  # v1 + v2
        vs = VectorSpace([v1, v2, v3])
        computed = basis(vs)
        print(f"\n✓ basis([e1, e2, e1+e2]) → {len(computed)} vectors (expected 2)")
        assert len(computed) == 2

    def test_basis_all_dependent_same_vector(self):
        v1 = Vector([1, 0])
        v2 = Vector([2, 0])  # 2 * v1
        v3 = Vector([3, 0])  # 3 * v1
        vs = VectorSpace([v1, v2, v3])
        computed = basis(vs)
        print(f"\n✓ basis([v, 2v, 3v]) → {len(computed)} vector (expected 1)")
        assert len(computed) == 1

    def test_basis_3d_three_independent(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = Vector([0, 0, 1])
        vs = VectorSpace([v1, v2, v3])
        computed = basis(vs)
        print(f"\n✓ basis(standard R³ basis) → {len(computed)} vectors (expected 3)")
        assert len(computed) == 3

    def test_basis_3d_with_dependent(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        v3 = Vector([7, 8, 9])  # v3 = 2*v2 - v1
        vs = VectorSpace([v1, v2, v3])
        computed = basis(vs)
        print(f"\n✓ basis([v1,v2,v3]) with v3 dependent → {len(computed)} vectors (expected 2)")
        assert len(computed) == 2

    def test_basis_returns_vectors_from_original_set(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        computed = basis(vs)
        print("\n✓ basis vectors are drawn from original data")
        assert all(b in vs.data for b in computed)

    def test_basis_single_vector(self):
        v = Vector([3, 1, 4])
        vs = VectorSpace([v])
        computed = basis(vs)
        print(f"\n✓ basis([v]) → {len(computed)} vector (expected 1)")
        assert len(computed) == 1
        assert computed[0] is v


class TestVectorSpaceGetItem:
    """Test cases for VectorSpace __getitem__."""

    def test_getitem_first(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ vs[0] = {vs[0]} (expected Vector([1, 0]))")
        assert vs[0] == v1

    def test_getitem_last(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ vs[1] = {vs[1]} (expected Vector([0, 1]))")
        assert vs[1] == v2

    def test_getitem_negative_index(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ vs[-1] = {vs[-1]} (expected Vector([0, 1]))")
        assert vs[-1] == v2

    def test_getitem_invalid_index_type(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ vs[1.0] → raises TypeError")
        with pytest.raises(TypeError):
            _ = vs[1.0]

    def test_getitem_invalid_index_string(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ vs['0'] → raises TypeError")
        with pytest.raises(TypeError):
            _ = vs["0"]


class TestVectorSpaceSetItem:
    """Test cases for VectorSpace __setitem__."""

    def test_setitem_valid(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        new_v = Vector([3, 0])
        vs[0] = new_v
        print(f"\n✓ vs[0] = Vector([3, 0]) → vs[0] = {vs[0]}")
        assert vs[0] == new_v

    def test_setitem_negative_index(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        new_v = Vector([2, 2])
        vs[-1] = new_v
        print(f"\n✓ vs[-1] = Vector([2, 2]) → vs[-1] = {vs[-1]}")
        assert vs[-1] == new_v

    def test_setitem_invalid_index_type(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ vs[1.0] = Vector([1, 0]) → raises TypeError")
        with pytest.raises(TypeError):
            vs[1.0] = Vector([1, 0])

    def test_setitem_invalid_value_not_vector(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ vs[0] = [1, 0] → raises TypeError (list not Vector)")
        with pytest.raises(TypeError):
            vs[0] = [1, 0]

    def test_setitem_invalid_dims_mismatch(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ vs[0] = Vector([1, 0, 0]) → raises ValueError (wrong dims)")
        with pytest.raises(ValueError):
            vs[0] = Vector([1, 0, 0])


class TestVectorSpaceLen:
    """Test cases for VectorSpace __len__."""

    def test_len_single(self):
        vs = VectorSpace([Vector([1, 2])])
        print(f"\n✓ len(VectorSpace([v])) = {len(vs)} (expected 1)")
        assert len(vs) == 1

    def test_len_multiple(self):
        v1, v2, v3 = Vector([1, 0]), Vector([0, 1]), Vector([1, 1])
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ len(VectorSpace([v1,v2,v3])) = {len(vs)} (expected 3)")
        assert len(vs) == 3

    def test_len_not_equal_to_dims(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v3 = Vector([1, 1])  # dependent
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ len={len(vs)} != dims={rank(vs)} when dependent vectors present")
        assert len(vs) == 3
        assert rank(vs) == 2


class TestVectorSpaceIter:
    """Test cases for VectorSpace __iter__."""

    def test_iter_yields_all_vectors(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        result = list(vs)
        print(f"\n✓ list(vs) = {result}")
        assert result == [v1, v2]

    def test_iter_order_preserved(self):
        v1 = Vector([3, 0])
        v2 = Vector([0, 5])
        v3 = Vector([1, 1])
        vs = VectorSpace([v1, v2, v3])
        result = list(vs)
        print("\n✓ iteration order matches insertion order")
        assert result[0] == v1
        assert result[1] == v2
        assert result[2] == v3


class TestVectorSpaceEquality:
    """Test cases for VectorSpace __eq__."""

    def test_equal_same_data(self):
        v1, v2 = Vector([1, 0]), Vector([0, 1])
        vs1 = VectorSpace([v1, v2])
        vs2 = VectorSpace([v1, v2])
        print("\n✓ VectorSpace([v1,v2]) == VectorSpace([v1,v2]) → True")
        assert vs1 == vs2

    def test_equal_different_order(self):
        v1, v2 = Vector([1, 0]), Vector([0, 1])
        vs1 = VectorSpace([v1, v2])
        vs2 = VectorSpace([v2, v1])
        print(
            "\n✓ VectorSpace([v1,v2]) == VectorSpace([v2,v1]) → True (order does not matter)"
        )
        assert vs1 == vs2

    def test_not_equal_different_vectors(self):
        vs1 = VectorSpace([Vector([1, 0])])
        vs2 = VectorSpace([Vector([0, 1])])
        print("\n✓ VectorSpace([v1]) != VectorSpace([v2]) → False")
        assert vs1 != vs2

    def test_not_equal_different_length(self):
        v1, v2 = Vector([1, 0]), Vector([0, 1])
        vs1 = VectorSpace([v1])
        vs2 = VectorSpace([v1, v2])
        print("\n✓ VectorSpace([v1]) != VectorSpace([v1, v2]) → False")
        assert vs1 != vs2

    def test_not_equal_to_non_vectorspace(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ VectorSpace != list → NotImplemented")
        assert vs.__eq__([Vector([1, 0])]) is NotImplemented


class TestVectorSpaceStrRepr:
    """Test cases for VectorSpace __str__ and __repr__."""

    def test_str_contains_ambient_dim(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        result = str(vs)
        print(f"\n✓ str(vs) = '{result}'")
        assert "R^3" in result

    def test_str_contains_generator_count(self):
        vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        result = str(vs)
        print("\n✓ str(vs) contains 'spanned by 2 vectors'")
        assert "spanned by 2 vectors" in result

    def test_str_contains_generator_vectors(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        result = str(vs)
        print("\n✓ str(vs) contains generator vector representations")
        assert "[1, 0]" in result
        assert "[0, 1]" in result

    def test_repr_format(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
        result = repr(vs)
        print(f"\n✓ repr(vs) = '{result}'")
        assert result == "VectorSpace(ambient=3, generators=3)"

    def test_repr_single_vector(self):
        vs = VectorSpace([Vector([1, 2])])
        result = repr(vs)
        print(f"\n✓ repr(single vector space) = '{result}'")
        assert result == "VectorSpace(ambient=2, generators=1)"


class TestVectorSpaceDims:
    """Test cases for VectorSpace dims property."""

    def test_dims_two_independent(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ dims([e1, e2]) = {rank(vs)} (expected 2)")
        assert rank(vs) == 2

    def test_dims_with_dependent_vector(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        v3 = Vector([1, 1])
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ dims([e1, e2, e1+e2]) = {rank(vs)} (expected 2)")
        assert rank(vs) == 2

    def test_dims_single_vector(self):
        v = Vector([1, 2, 3])
        vs = VectorSpace([v])
        print(f"\n✓ dims([v]) = {rank(vs)} (expected 1)")
        assert rank(vs) == 1

    def test_dims_all_dependent(self):
        v1 = Vector([2, 4])
        v2 = Vector([1, 2])  # 0.5 * v1
        v3 = Vector([3, 6])  # 1.5 * v1
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ dims of collinear set = {rank(vs)} (expected 1)")
        assert rank(vs) == 1

    def test_dims_equals_len_basis(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = Vector([1, 1, 0])  # dependent
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ dims == len(basis): {rank(vs)} == {len(basis(vs))}")
        assert rank(vs) == len(basis(vs))

    def test_dims_full_rank_3d(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = Vector([0, 0, 1])
        vs = VectorSpace([v1, v2, v3])
        print(f"\n✓ dims(standard R³) = {rank(vs)} (expected 3)")
        assert rank(vs) == 3


class TestVectorSpaceAmbientDims:
    """Test cases for VectorSpace ambient_dims property."""

    def test_ambient_dims_2d(self):
        vs = VectorSpace([Vector([1, 0])])
        print(f"\n✓ ambient_dims of R²-space = {vs.ambient_dims} (expected 2)")
        assert vs.ambient_dims == 2

    def test_ambient_dims_3d(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        print(f"\n✓ ambient_dims of R³-subspace = {vs.ambient_dims} (expected 3)")
        assert vs.ambient_dims == 3

    def test_ambient_dims_independent_of_num_vectors(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])
        print("\n✓ ambient_dims unchanged by number of spanning vectors")
        assert vs.ambient_dims == 3


class TestVectorSpaceIsFullRank:
    """Test cases for VectorSpace is_full_rank property."""

    def test_full_rank_standard_basis_2d(self):
        vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        print(f"\n✓ standard R² basis is full rank")
        assert is_full_rank(vs) is True

    def test_full_rank_standard_basis_3d(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])
        print(f"\n✓ standard R³ basis is full rank")
        assert is_full_rank(vs) is True

    def test_not_full_rank_subspace(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        print(f"\n✓ plane in R³ is not full rank")
        assert is_full_rank(vs) is False

    def test_not_full_rank_single_vector_in_3d(self):
        vs = VectorSpace([Vector([1, 2, 3])])
        print(f"\n✓ line in R³ is not full rank")
        assert is_full_rank(vs) is False

    def test_not_full_rank_with_dependent_vectors(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
        print(f"\n✓ dependent spanning set in R³ does not become full rank")
        assert is_full_rank(vs) is False


class TestVectorSpaceContains:
    """Test cases for VectorSpace contains method."""

    def test_contains_basis_vector(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        vs = VectorSpace([v1, v2])
        print(f"\n✓ basis vector is contained in its own space")
        assert contains(vs, v1) is True

    def test_contains_linear_combination(self):
        vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        print(f"\n✓ [3, 4] is in span of standard R² basis")
        assert contains(vs, Vector([3, 4])) is True

    def test_not_contains_out_of_plane(self):
        vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        print(f"\n✓ [0, 0, 1] is not in the xy-plane subspace")
        assert contains(vs, Vector([0, 0, 1])) is False

    def test_contains_vector_in_line(self):
        vs = VectorSpace([Vector([1, 2])])
        print(f"\n✓ [3, 6] = 3*[1,2] is in the span")
        assert contains(vs, Vector([3, 6])) is True

    def test_not_contains_off_line(self):
        vs = VectorSpace([Vector([1, 2])])
        print(f"\n✓ [1, 3] is not a multiple of [1, 2]")
        assert contains(vs, Vector([1, 3])) is False

    def test_contains_invalid_type(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ contains([1, 0]) → raises TypeError")
        with pytest.raises(TypeError):
            contains(vs, [1, 0])

    def test_contains_wrong_dims(self):
        vs = VectorSpace([Vector([1, 0])])
        print("\n✓ contains(Vector([1, 0, 0])) in R²-space → raises ValueError")
        with pytest.raises(ValueError):
            contains(vs, Vector([1, 0, 0]))


class TestVectorSpaceSameSubspace:
    """Test cases for mathematical subspace equality comparison."""

    def test_same_generators(self):
        v1, v2 = Vector([1, 0]), Vector([0, 1])
        vs1 = VectorSpace([v1, v2])
        vs2 = VectorSpace([v1, v2])
        assert same_subspace(vs1, vs2) is True

    def test_different_generators_same_span(self):
        v1, v2 = Vector([1, 0]), Vector([0, 1])
        vs1 = VectorSpace([v1, v2])
        vs2 = VectorSpace([v1, v1 + v2])
        assert same_subspace(vs1, vs2) is True

    def test_scalar_multiple_single_vector(self):
        vs1 = VectorSpace([Vector([1, 2])])
        vs2 = VectorSpace([Vector([2, 4])])
        assert same_subspace(vs1, vs2) is True

    def test_different_dimensions(self):
        vs1 = VectorSpace([Vector([1, 0, 0])])
        vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        assert same_subspace(vs1, vs2) is False

    def test_same_dimension_different_subspaces(self):
        vs1 = VectorSpace([Vector([1, 0])])
        vs2 = VectorSpace([Vector([0, 1])])
        assert same_subspace(vs1, vs2) is False

    def test_full_space_different_bases(self):
        vs1 = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        vs2 = VectorSpace([Vector([1, 1]), Vector([1, -1])])
        assert same_subspace(vs1, vs2) is True

    def test_planes_in_r3(self):
        vs1 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        vs2 = VectorSpace([Vector([1, 1, 0]), Vector([1, -1, 0])])
        assert same_subspace(vs1, vs2) is True

    def test_different_planes_in_r3(self):
        vs1 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 0, 1])])
        assert same_subspace(vs1, vs2) is False

    def test_type_error(self):
        vs = VectorSpace([Vector([1, 0])])
        with pytest.raises(TypeError):
            same_subspace(vs, "not a space")

    def test_ambient_dimension_mismatch(self):
        vs1 = VectorSpace([Vector([1, 0])])
        vs2 = VectorSpace([Vector([1, 0, 0])])
        with pytest.raises(ValueError):
            same_subspace(vs1, vs2)
