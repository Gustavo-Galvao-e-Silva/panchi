import pytest

from panchi.primitives import Matrix, Vector, identity
from panchi.algorithms import RowAdd, RowScale, RowSwap, ref, rref



# ==================== ROW SWAP TESTS ====================


class TestRowSwapInit:
    """Test RowSwap construction and validation."""

    def test_valid_construction(self):
        op = RowSwap(0, 1)
        assert op.a == 0
        assert op.b == 1

    def test_non_integer_row_a_raises_type_error(self):
        with pytest.raises(TypeError):
            RowSwap(0.0, 1)

    def test_non_integer_row_b_raises_type_error(self):
        with pytest.raises(TypeError):
            RowSwap(0, "1")


class TestRowSwapElementaryMatrix:
    """Test RowSwap elementary matrix construction."""

    def test_elementary_matrix_2x2(self):
        op = RowSwap(0, 1)
        assert op.elementary_matrix(2) == Matrix([[0, 1], [1, 0]])

    def test_elementary_matrix_3x3(self):
        op = RowSwap(0, 2)
        assert op.elementary_matrix(3) == Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    def test_elementary_matrix_non_integer_n_raises_type_error(self):
        with pytest.raises(TypeError):
            RowSwap(0, 1).elementary_matrix(2.0)

    def test_elementary_matrix_n_less_than_2_raises_value_error(self):
        with pytest.raises(ValueError):
            RowSwap(0, 1).elementary_matrix(1)

    def test_elementary_matrix_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowSwap(0, 3).elementary_matrix(3)


class TestRowSwapApplyMatrix:
    """Test RowSwap applied to matrices."""

    def test_apply_swaps_first_and_last_rows(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert RowSwap(0, 2).apply(m) == Matrix([[5, 6], [3, 4], [1, 2]])

    def test_apply_swaps_adjacent_rows(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert RowSwap(0, 1).apply(m) == Matrix([[3, 4], [1, 2], [5, 6]])

    def test_apply_does_not_modify_original(self):
        m = Matrix([[1, 2], [3, 4]])
        RowSwap(0, 1).apply(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_apply_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            RowSwap(0, 1).apply([[1, 2], [3, 4]])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowSwap(0, 5).apply(Matrix([[1, 2], [3, 4]]))

    def test_apply_consistent_with_elementary_matrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        op = RowSwap(0, 2)
        assert op.apply(m) == op.elementary_matrix(3) @ m


class TestRowSwapApplyVector:
    """Test RowSwap applied to vectors."""

    def test_apply_swaps_first_and_last_entries(self):
        v = Vector([1, 2, 3])
        print(RowSwap(0, 2).apply(v))
        assert RowSwap(0, 2).apply(v) == Vector([3, 2, 1])

    def test_apply_swaps_adjacent_entries(self):
        v = Vector([1, 2, 3])
        print(RowSwap(0, 1).apply(v))
        assert RowSwap(0, 1).apply(v) == Vector([2, 1, 3])

    def test_apply_does_not_modify_original(self):
        v = Vector([1, 2, 3])
        RowSwap(0, 2).apply(v)
        assert v == Vector([1, 2, 3])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowSwap(0, 5).apply(Vector([1, 2, 3]))

    def test_apply_returns_vector(self):
        assert isinstance(RowSwap(0, 1).apply(Vector([1, 2])), Vector)

    def test_apply_twice_restores_original(self):
        v = Vector([1, 2, 3])
        op = RowSwap(0, 2)
        assert op.apply(op.apply(v)) == v


class TestRowSwapInverse:
    """Test RowSwap inverse."""

    def test_inverse_returns_same_indices(self):
        inv = RowSwap(0, 2).inverse()
        assert inv.a == 0
        assert inv.b == 2

    def test_inverse_undoes_swap(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        op = RowSwap(0, 2)
        assert op.inverse().apply(op.apply(m)) == m


class TestRowSwapStringRepresentation:
    """Test RowSwap string representations."""

    def test_str(self):
        assert str(RowSwap(0, 2)) == "R0 <-> R2"

    def test_repr(self):
        assert repr(RowSwap(0, 2)) == "RowSwap(row_a=0, row_b=2)"


# ==================== ROW SCALE TESTS ====================


class TestRowScaleInit:
    """Test RowScale construction and validation."""

    def test_valid_construction(self):
        op = RowScale(1, 3)
        assert op.row == 1
        assert op.scalar == 3

    def test_float_scalar_allowed(self):
        assert RowScale(0, 0.5).scalar == 0.5

    def test_non_integer_row_raises_type_error(self):
        with pytest.raises(TypeError):
            RowScale(1.0, 3)

    def test_non_numeric_scalar_raises_type_error(self):
        with pytest.raises(TypeError):
            RowScale(0, "3")

    def test_zero_scalar_raises_value_error(self):
        with pytest.raises(ValueError):
            RowScale(0, 0)


class TestRowScaleElementaryMatrix:
    """Test RowScale elementary matrix construction."""

    def test_elementary_matrix_scales_diagonal(self):
        assert RowScale(1, 3).elementary_matrix(2) == Matrix([[1, 0], [0, 3]])

    def test_elementary_matrix_first_row(self):
        assert RowScale(0, 2).elementary_matrix(3) == Matrix(
            [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def test_elementary_matrix_n_less_than_2_raises_value_error(self):
        with pytest.raises(ValueError):
            RowScale(0, 2).elementary_matrix(1)

    def test_elementary_matrix_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowScale(5, 2).elementary_matrix(3)


class TestRowScaleApplyMatrix:
    """Test RowScale applied to matrices."""

    def test_apply_scales_target_row(self):
        m = Matrix([[1, 2], [3, 4]])
        assert RowScale(1, 3).apply(m) == Matrix([[1, 2], [9, 12]])

    def test_apply_scales_with_negative_scalar(self):
        m = Matrix([[1, 2], [3, 4]])
        assert RowScale(0, -1).apply(m) == Matrix([[-1, -2], [3, 4]])

    def test_apply_does_not_modify_original(self):
        m = Matrix([[1, 2], [3, 4]])
        RowScale(0, 5).apply(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_apply_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            RowScale(0, 2).apply([[1, 2], [3, 4]])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowScale(5, 2).apply(Matrix([[1, 2], [3, 4]]))

    def test_apply_consistent_with_elementary_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        op = RowScale(1, 3)
        assert op.apply(m) == op.elementary_matrix(2) @ m


class TestRowScaleApplyVector:
    """Test RowScale applied to vectors."""

    def test_apply_scales_target_entry(self):
        v = Vector([1, 2, 3])
        assert RowScale(1, 3).apply(v) == Vector([1, 6, 3])

    def test_apply_scales_first_entry(self):
        v = Vector([2, 3])
        assert RowScale(0, 5).apply(v) == Vector([10, 3])

    def test_apply_scales_with_negative_scalar(self):
        v = Vector([1, 2, 3])
        assert RowScale(2, -1).apply(v) == Vector([1, 2, -3])

    def test_apply_does_not_modify_original(self):
        v = Vector([1, 2, 3])
        RowScale(0, 10).apply(v)
        assert v == Vector([1, 2, 3])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowScale(5, 2).apply(Vector([1, 2, 3]))

    def test_apply_returns_vector(self):
        assert isinstance(RowScale(0, 2).apply(Vector([1, 2])), Vector)

    def test_apply_inverse_restores_original(self):
        v = Vector([1, 2, 3])
        op = RowScale(1, 4)
        assert op.inverse().apply(op.apply(v)) == v


class TestRowScaleInverse:
    """Test RowScale inverse."""

    def test_inverse_has_reciprocal_scalar(self):
        inv = RowScale(1, 3).inverse()
        assert inv.row == 1
        assert abs(inv.scalar - (1 / 3)) < 1e-10

    def test_inverse_undoes_scale(self):
        m = Matrix([[1, 2], [3, 4]])
        op = RowScale(0, 4)
        assert op.inverse().apply(op.apply(m)) == m


class TestRowScaleStringRepresentation:
    """Test RowScale string representations."""

    def test_str(self):
        assert str(RowScale(1, 3)) == "R1 -> 3 * R1"

    def test_repr(self):
        assert repr(RowScale(1, 3)) == "RowScale(row=1, scalar=3)"


# ==================== ROW ADD TESTS ====================


class TestRowAddInit:
    """Test RowAdd construction and validation."""

    def test_valid_construction(self):
        op = RowAdd(1, 0, -3)
        assert op.target == 1
        assert op.source == 0
        assert op.scalar == -3

    def test_float_scalar_allowed(self):
        assert RowAdd(1, 0, 0.5).scalar == 0.5

    def test_non_integer_target_raises_type_error(self):
        with pytest.raises(TypeError):
            RowAdd(1.0, 0, -3)

    def test_non_integer_source_raises_type_error(self):
        with pytest.raises(TypeError):
            RowAdd(1, "0", -3)

    def test_non_numeric_scalar_raises_type_error(self):
        with pytest.raises(TypeError):
            RowAdd(1, 0, None)


class TestRowAddElementaryMatrix:
    """Test RowAdd elementary matrix construction."""

    def test_elementary_matrix_places_scalar_correctly(self):
        assert RowAdd(target=1, source=0, scalar=-3).elementary_matrix(2) == Matrix(
            [[1, 0], [-3, 1]]
        )

    def test_elementary_matrix_3x3(self):
        assert RowAdd(target=2, source=0, scalar=4).elementary_matrix(3) == Matrix(
            [[1, 0, 0], [0, 1, 0], [4, 0, 1]]
        )

    def test_elementary_matrix_n_less_than_2_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(1, 0, 2).elementary_matrix(1)

    def test_elementary_matrix_same_target_and_source_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(1, 1, 2).elementary_matrix(2)


class TestRowAddApplyMatrix:
    """Test RowAdd applied to matrices."""

    def test_apply_eliminates_entry(self):
        m = Matrix([[1, 2], [3, 4]])
        assert RowAdd(target=1, source=0, scalar=-3).apply(m) == Matrix(
            [[1, 2], [0, -2]]
        )

    def test_apply_adds_positive_multiple(self):
        m = Matrix([[2, 1], [6, 4]])
        assert RowAdd(target=1, source=0, scalar=-3).apply(m) == Matrix(
            [[2, 1], [0, 1]]
        )

    def test_apply_does_not_modify_original(self):
        m = Matrix([[1, 2], [3, 4]])
        RowAdd(target=1, source=0, scalar=2).apply(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_apply_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            RowAdd(1, 0, 2).apply([[1, 2], [3, 4]])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(5, 0, 2).apply(Matrix([[1, 2], [3, 4]]))

    def test_apply_same_target_and_source_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(1, 1, 2).apply(Matrix([[1, 2], [3, 4]]))

    def test_apply_consistent_with_elementary_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        op = RowAdd(target=1, source=0, scalar=-3)
        assert op.apply(m) == op.elementary_matrix(2) @ m


class TestRowAddApplyVector:
    """Test RowAdd applied to vectors."""

    def test_apply_adds_multiple_to_target_entry(self):
        v = Vector([1, 3])
        assert RowAdd(target=1, source=0, scalar=-3).apply(v) == Vector([1, 0])

    def test_apply_adds_upward(self):
        v = Vector([2, 4])
        assert RowAdd(target=0, source=1, scalar=1).apply(v) == Vector([6, 4])

    def test_apply_with_float_scalar(self):
        v = Vector([2, 0])
        assert RowAdd(target=1, source=0, scalar=0.5).apply(v) == Vector([2, 1])

    def test_apply_does_not_modify_original(self):
        v = Vector([1, 2, 3])
        RowAdd(target=2, source=0, scalar=5).apply(v)
        assert v == Vector([1, 2, 3])

    def test_apply_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(5, 0, 2).apply(Vector([1, 2, 3]))

    def test_apply_same_target_and_source_raises_value_error(self):
        with pytest.raises(ValueError):
            RowAdd(1, 1, 2).apply(Vector([1, 2, 3]))

    def test_apply_returns_vector(self):
        assert isinstance(RowAdd(1, 0, -3).apply(Vector([1, 2])), Vector)

    def test_apply_inverse_restores_original(self):
        v = Vector([1, 4])
        op = RowAdd(target=1, source=0, scalar=-3)
        assert op.inverse().apply(op.apply(v)) == v


class TestRowAddInverse:
    """Test RowAdd inverse."""

    def test_inverse_has_negated_scalar(self):
        inv = RowAdd(target=1, source=0, scalar=-3).inverse()
        assert inv.target == 1
        assert inv.source == 0
        assert inv.scalar == 3

    def test_inverse_undoes_row_add(self):
        m = Matrix([[1, 2], [3, 4]])
        op = RowAdd(target=1, source=0, scalar=-3)
        assert op.inverse().apply(op.apply(m)) == m


class TestRowAddStringRepresentation:
    """Test RowAdd string representations."""

    def test_str(self):
        assert str(RowAdd(1, 0, -3)) == "R1 -> R1 + (-3) * R0"

    def test_repr(self):
        assert repr(RowAdd(1, 0, -3)) == "RowAdd(target=1, source=0, scalar=-3)"


# ==================== REDUCTION TESTS ====================


class TestReductionProperties:
    """Test the Reduction result object's computed properties."""

    def test_rank_equals_number_of_pivots(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = ref(m)
        assert reduction.rank == len(reduction.pivots)

    def test_pivots_are_row_col_tuples(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = ref(m)
        for pivot in reduction.pivots:
            assert isinstance(pivot, tuple)
            assert len(pivot) == 2

    def test_nullity_satisfies_rank_nullity_theorem(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        reduction = ref(m)
        assert reduction.rank + reduction.nullity == m.cols

    def test_original_is_preserved(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = ref(m)
        assert reduction.original == m

    def test_form_is_set_correctly(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = ref(m)
        assert reduction.form == "REF"

    def test_str_contains_form_label(self):
        m = Matrix([[1, 2], [3, 4]])
        assert "REF" in str(ref(m))

    def test_repr_contains_form_label(self):
        m = Matrix([[1, 2], [3, 4]])
        assert "REF" in repr(ref(m))


# ==================== REF TESTS ====================


class TestRef:
    """Test row echelon form reduction."""

    def test_ref_of_identity_is_identity(self):
        m = identity(3)
        assert ref(m).result == m

    def test_ref_2x2_result(self):
        m = Matrix([[1, 2], [3, 4]])
        assert ref(m).result == Matrix([[1, 2], [0, -2]])

    def test_ref_2x2_pivots(self):
        m = Matrix([[1, 2], [3, 4]])
        assert ref(m).pivots == [(0, 0), (1, 1)]

    def test_ref_2x2_rank_and_nullity(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = ref(m)
        assert reduction.rank == 2
        assert reduction.nullity == 0

    def test_ref_3x3_result(self):
        m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        assert ref(m).result == Matrix([[1, 2, 3], [0, 1, 1], [0, 0, 1]])

    def test_ref_3x3_rank(self):
        m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        assert ref(m).rank == 3

    def test_ref_rank_deficient_2x3_result(self):
        m = Matrix([[1, 2, 3], [2, 4, 6]])
        assert ref(m).result == Matrix([[1, 2, 3], [0, 0, 0]])

    def test_ref_rank_deficient_2x3_rank_and_nullity(self):
        m = Matrix([[1, 2, 3], [2, 4, 6]])
        reduction = ref(m)
        assert reduction.rank == 1
        assert reduction.nullity == 2

    def test_ref_zero_matrix_result(self):
        m = Matrix([[0, 0], [0, 0]])
        assert ref(m).result == Matrix([[0, 0], [0, 0]])

    def test_ref_zero_matrix_rank_and_pivots(self):
        m = Matrix([[0, 0], [0, 0]])
        reduction = ref(m)
        assert reduction.rank == 0
        assert reduction.pivots == []

    def test_ref_wide_matrix_result(self):
        m = Matrix([[1, 2, 3, 4], [0, 1, 2, 3]])
        assert ref(m).result == Matrix([[1, 2, 3, 4], [0, 1, 2, 3]])

    def test_ref_wide_matrix_rank_and_nullity(self):
        m = Matrix([[1, 2, 3, 4], [0, 1, 2, 3]])
        reduction = ref(m)
        assert reduction.rank == 2
        assert reduction.nullity == 2

    def test_ref_tall_full_rank_result(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert ref(m).result == Matrix([[1, 2], [0, -2], [0, 0]])

    def test_ref_tall_full_rank_rank(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert ref(m).rank == 2

    def test_ref_tall_all_dependent_rows_result(self):
        m = Matrix([[1, 2], [2, 4], [3, 6]])
        assert ref(m).result == Matrix([[1, 2], [0, 0], [0, 0]])

    def test_ref_tall_all_dependent_rows_rank(self):
        m = Matrix([[1, 2], [2, 4], [3, 6]])
        assert ref(m).rank == 1

    def test_ref_tall_zero_row_in_input_result(self):
        m = Matrix([[1, 2, 3], [0, 0, 0], [0, 1, 2]])
        assert ref(m).result == Matrix([[1, 2, 3], [0, 1, 2], [0, 0, 0]])

    def test_ref_tall_zero_row_in_input_rank(self):
        m = Matrix([[1, 2, 3], [0, 0, 0], [0, 1, 2]])
        assert ref(m).rank == 2

    def test_ref_tall_zero_row_sinks_to_bottom(self):
        m = Matrix([[1, 2], [0, 0], [3, 4]])
        assert ref(m).result == Matrix([[1, 2], [0, -2], [0, 0]])

    def test_ref_tall_pivot_count_never_exceeds_cols(self):
        m = Matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert ref(m).rank <= m.cols

    def test_ref_does_not_modify_original_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        ref(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_ref_form_label(self):
        assert ref(Matrix([[1, 2], [3, 4]])).form == "REF"

    def test_ref_original_is_preserved(self):
        m = Matrix([[1, 2], [3, 4]])
        assert ref(m).original == m


# ==================== RREF TESTS ====================


class TestRref:
    """Test reduced row echelon form reduction."""

    def test_rref_of_identity_is_identity(self):
        m = identity(3)
        assert rref(m).result == m

    def test_rref_2x2_result(self):
        m = Matrix([[1, 2], [3, 4]])
        assert rref(m).result == identity(2)

    def test_rref_2x2_pivots(self):
        m = Matrix([[1, 2], [3, 4]])
        assert rref(m).pivots == [(0, 0), (1, 1)]

    def test_rref_2x2_rank_and_nullity(self):
        m = Matrix([[1, 2], [3, 4]])
        reduction = rref(m)
        assert reduction.rank == 2
        assert reduction.nullity == 0

    def test_rref_3x3_result(self):
        m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        assert rref(m).result == identity(3)

    def test_rref_3x3_rank(self):
        m = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        assert rref(m).rank == 3

    def test_rref_pivot_entries_are_one(self):
        m = Matrix([[2, 4], [1, 3]])
        assert rref(m).result == Matrix([[1, 0], [0, 1]])

    def test_rref_has_zeros_above_and_below_each_pivot(self):
        m = Matrix([[1, 2], [3, 4]])
        assert rref(m).result == identity(2)

    def test_rref_rank_deficient_2x3_result(self):
        m = Matrix([[1, 2, 3], [2, 4, 6]])
        assert rref(m).result == Matrix([[1, 2, 3], [0, 0, 0]])

    def test_rref_rank_deficient_2x3_rank_and_nullity(self):
        m = Matrix([[1, 2, 3], [2, 4, 6]])
        reduction = rref(m)
        assert reduction.rank == 1
        assert reduction.nullity == 2

    def test_rref_wide_matrix_result(self):
        m = Matrix([[1, 2, 3, 4], [0, 1, 2, 3]])
        assert rref(m).result == Matrix([[1, 0, -1, -2], [0, 1, 2, 3]])

    def test_rref_wide_matrix_rank_and_nullity(self):
        m = Matrix([[1, 2, 3, 4], [0, 1, 2, 3]])
        reduction = rref(m)
        assert reduction.rank == 2
        assert reduction.nullity == 2

    def test_rref_zero_matrix_result(self):
        m = Matrix([[0, 0], [0, 0]])
        assert rref(m).result == Matrix([[0, 0], [0, 0]])

    def test_rref_zero_matrix_rank_and_pivots(self):
        m = Matrix([[0, 0], [0, 0]])
        reduction = rref(m)
        assert reduction.rank == 0
        assert reduction.pivots == []

    def test_rref_tall_full_rank_result(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert rref(m).result == Matrix([[1, 0], [0, 1], [0, 0]])

    def test_rref_tall_full_rank_rank(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert rref(m).rank == 2

    def test_rref_tall_all_dependent_rows_result(self):
        m = Matrix([[1, 2], [2, 4], [3, 6]])
        assert rref(m).result == Matrix([[1, 2], [0, 0], [0, 0]])

    def test_rref_tall_all_dependent_rows_rank(self):
        m = Matrix([[1, 2], [2, 4], [3, 6]])
        assert rref(m).rank == 1

    def test_rref_tall_zero_row_in_input_result(self):
        m = Matrix([[1, 2, 3], [0, 0, 0], [0, 1, 2]])
        assert rref(m).result == Matrix([[1, 0, -1], [0, 1, 2], [0, 0, 0]])

    def test_rref_tall_zero_row_in_input_rank(self):
        m = Matrix([[1, 2, 3], [0, 0, 0], [0, 1, 2]])
        assert rref(m).rank == 2

    def test_rref_tall_zero_row_sinks_to_bottom(self):
        m = Matrix([[1, 2], [0, 0], [3, 4]])
        assert rref(m).result == Matrix([[1, 0], [0, 1], [0, 0]])

    def test_rref_tall_pivot_count_never_exceeds_cols(self):
        m = Matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert rref(m).rank <= m.cols

    def test_rref_satisfies_rank_nullity_theorem(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        reduction = rref(m)
        assert reduction.rank + reduction.nullity == m.cols

    def test_rref_does_not_modify_original_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        rref(m)
        assert m == Matrix([[1, 2], [3, 4]])

    def test_rref_form_label(self):
        assert rref(Matrix([[1, 2], [3, 4]])).form == "RREF"

    def test_rref_original_is_preserved(self):
        m = Matrix([[1, 2], [3, 4]])
        assert rref(m).original == m
