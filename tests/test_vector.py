import pytest

from panchi.primitives.vector import Vector

# ==================== VECTOR TESTS ====================


class TestVectorInitialization:
    """Test cases for Vector object initialization and validation."""

    def test_valid_integer_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector([1,2,3]) → data={v.data}, dim={v.dims}, shape={v.shape}")
        assert v.data == [1, 2, 3]
        assert v.dims == 3
        assert v.shape == (3, 1)

    def test_valid_float_vector(self):
        v = Vector([1.5, 2.7, 3.14])
        print(f"\n✓ Vector([1.5,2.7,3.14]) → data={v.data}, dim={v.dims}")
        assert v.data == [1.5, 2.7, 3.14]
        assert v.dims == 3

    def test_mixed_int_float_vector(self):
        v = Vector([1, 2.5, 3])
        print(f"\n✓ Vector([1,2.5,3]) → data={v.data}, dim={v.dims}")
        assert v.data == [1, 2.5, 3]
        assert v.dims == 3

    def test_empty_vector(self):
        v = Vector([])
        print(f"\n✓ Vector([]) → data={v.data}, dim={v.dims}")
        assert v.data == []
        assert v.dims == 0

    def test_single_element_vector(self):
        v = Vector([42])
        print(f"\n✓ Vector([42]) → data={v.data}, dim={v.dims}")
        assert v.data == [42]
        assert v.dims == 1

    def test_invalid_type_string(self):
        print(f"\n✓ Vector('not a list') → raises TypeError")
        with pytest.raises(TypeError):
            Vector("not a list")

    def test_invalid_type_in_list(self):
        print(f"\n✓ Vector([1,2,'three']) → raises TypeError (corrected validation)")
        with pytest.raises(TypeError):
            Vector([1, 2, "three"])


class TestVectorIndexing:
    """Test cases for Vector indexing operations."""

    def test_valid_indexing(self):
        v = Vector([10, 20, 30, 40])
        print(f"\n✓ v[0]={v[0]}, v[1]={v[1]}, v[3]={v[3]}")
        assert v[0] == 10
        assert v[1] == 20
        assert v[3] == 40

    def test_negative_indexing(self):
        v = Vector([10, 20, 30])
        print(f"\n✓ v[-1]={v[-1]}, v[-2]={v[-2]}")
        assert v[-1] == 30
        assert v[-2] == 20

    def test_invalid_index_type_float(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v[1.5] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v[1.5]

    def test_invalid_index_type_string(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v['0'] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v["0"]


class TestVectorAddition:
    """Test cases for Vector addition operations."""

    def test_add_same_dimension(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        result = v1 + v2
        print(f"\n✓ [1,2,3] + [4,5,6] = {result.data} (expected [5,7,9])")
        assert result.data == [5, 7, 9]

    def test_add_floats(self):
        v1 = Vector([1.5, 2.5])
        v2 = Vector([0.5, 1.5])
        result = v1 + v2
        print(f"\n✓ [1.5,2.5] + [0.5,1.5] = {result.data} (expected [2.0,4.0])")
        assert result.data == [2.0, 4.0]

    def test_add_different_dimensions(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        print(f"\n✓ [1,2] + [1,2,3] → raises TypeError (dimension mismatch)")
        with pytest.raises(TypeError):
            _ = v1 + v2

    def test_add_non_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector + list → raises TypeError")
        with pytest.raises(TypeError):
            _ = v + [1, 2, 3]

    def test_add_zero_vector(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([0, 0, 0])
        result = v1 + v2
        print(f"\n✓ [1,2,3] + [0,0,0] = {result.data} (expected [1,2,3])")
        assert result.data == [1, 2, 3]


class TestVectorSubtraction:
    """Test cases for Vector subtraction operations."""

    def test_subtract_same_dimension(self):
        v1 = Vector([5, 7, 9])
        v2 = Vector([1, 2, 3])
        result = v1 - v2
        print(f"\n✓ [5,7,9] - [1,2,3] = {result.data} (expected [4,5,6])")
        assert result.data == [4, 5, 6]

    def test_subtract_negative_result(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([2, 3, 4])
        result = v1 - v2
        print(f"\n✓ [1,2,3] - [2,3,4] = {result.data} (expected [-1,-1,-1])")
        assert result.data == [-1, -1, -1]

    def test_subtract_different_dimensions(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        print(f"\n✓ [1,2] - [1,2,3] → raises TypeError")
        with pytest.raises(TypeError):
            _ = v1 - v2

    def test_subtract_non_vector(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector - scalar → raises TypeError")
        with pytest.raises(TypeError):
            _ = v - 5


class TestVectorScalarMultiplication:
    """Test cases for scalar multiplication with Vectors."""

    def test_multiply_by_integer(self):
        v = Vector([1, 2, 3])
        result = 3 * v
        print(f"\n✓ 3 * [1,2,3] = {result.data} (expected [3,6,9])")
        assert result.data == [3, 6, 9]

    def test_multiply_by_float(self):
        v = Vector([2, 4, 6])
        result = 0.5 * v
        print(f"\n✓ 0.5 * [2,4,6] = {result.data} (expected [1.0,2.0,3.0])")
        assert result.data == [1.0, 2.0, 3.0]

    def test_multiply_by_zero(self):
        v = Vector([1, 2, 3])
        result = 0 * v
        print(f"\n✓ 0 * [1,2,3] = {result.data} (expected [0,0,0])")
        assert result.data == [0, 0, 0]

    def test_multiply_by_negative(self):
        v = Vector([1, 2, 3])
        result = -2 * v
        print(f"\n✓ -2 * [1,2,3] = {result.data} (expected [-2,-4,-6])")
        assert result.data == [-2, -4, -6]

    def test_multiply_by_non_scalar(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ 'string' * Vector → raises TypeError")
        with pytest.raises(TypeError):
            result = "string" * v


class TestVectorSetItem:
    """Test cases for Vector element assignment."""

    def test_set_valid_integer(self):
        v = Vector([1, 2, 3])
        v[1] = 10
        print(f"\n✓ v[1] = 10 → v.data = {v.data} (expected [1, 10, 3])")
        assert v.data == [1, 10, 3]

    def test_set_valid_float(self):
        v = Vector([1, 2, 3])
        v[0] = 3.14
        print(f"\n✓ v[0] = 3.14 → v.data = {v.data} (expected [3.14, 2, 3])")
        assert v.data == [3.14, 2, 3]

    def test_set_invalid_index_type(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v['0'] = 5 → raises TypeError")
        with pytest.raises(TypeError):
            v["0"] = 5

    def test_set_invalid_value_type(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ v[0] = 'string' → raises TypeError")
        with pytest.raises(TypeError):
            v[0] = "string"


class TestVectorDivision:
    """Test cases for Vector division by scalars."""

    def test_divide_by_integer(self):
        v = Vector([10, 20, 30])
        result = v / 2
        print(f"\n✓ [10, 20, 30] / 2 = {result.data} (expected [5.0, 10.0, 15.0])")
        assert result.data == [5.0, 10.0, 15.0]

    def test_divide_by_float(self):
        v = Vector([1, 2, 3])
        result = v / 2.0
        print(f"\n✓ [1, 2, 3] / 2.0 = {result.data} (expected [0.5, 1.0, 1.5])")
        assert result.data == [0.5, 1.0, 1.5]

    def test_divide_by_non_scalar(self):
        v = Vector([1, 2, 3])
        print(f"\n✓ Vector / 'string' → raises TypeError")
        with pytest.raises(TypeError):
            _ = v / "string"


class TestVectorNegation:
    """Test cases for Vector negation."""

    def test_negate_vector(self):
        v = Vector([1, -2, 3])
        result = -v
        print(f"\n✓ -[1, -2, 3] = {result.data} (expected [-1, 2, -3])")
        assert result.data == [-1, 2, -3]

    def test_double_negation(self):
        v = Vector([1, 2, 3])
        result = -(-v)
        print(f"\n✓ -(-[1, 2, 3]) = {result.data} (expected [1, 2, 3])")
        assert result.data == [1, 2, 3]


class TestVectorMagnitude:
    """Test cases for Vector magnitude property."""

    def test_magnitude_unit_vector(self):
        v = Vector([1, 0, 0])
        print(f"\n✓ |[1, 0, 0]| = {v.magnitude} (expected 1.0)")
        assert v.magnitude == 1.0

    def test_magnitude_3_4_vector(self):
        v = Vector([3, 4])
        print(f"\n✓ |[3, 4]| = {v.magnitude} (expected 5.0)")
        assert v.magnitude == 5.0

    def test_magnitude_zero_vector(self):
        v = Vector([0, 0, 0])
        print(f"\n✓ |[0, 0, 0]| = {v.magnitude} (expected 0.0)")
        assert v.magnitude == 0.0

    def test_magnitude_general(self):
        v = Vector([1, 2, 2])
        expected = 3.0  # sqrt(1 + 4 + 4) = sqrt(9) = 3
        print(f"\n✓ |[1, 2, 2]| = {v.magnitude} (expected {expected})")
        assert v.magnitude == expected


class TestVectorNormalize:
    """Test cases for Vector normalization."""

    def test_normalize_vector(self):
        v = Vector([3, 4])
        normalized = v.normalize()
        print(f"\n✓ normalize([3, 4]) = {normalized.data} (expected [0.6, 0.8])")
        assert normalized.data == [0.6, 0.8]
        assert abs(normalized.magnitude - 1.0) < 1e-10

    def test_normalize_already_unit(self):
        v = Vector([1, 0])
        normalized = v.normalize()
        print(f"\n✓ normalize([1, 0]) = {normalized.data} (expected [1.0, 0.0])")
        assert normalized.data == [1.0, 0.0]

    def test_normalize_preserves_original(self):
        v = Vector([3, 4])
        original_data = v.data.copy()
        normalized = v.normalize()
        print(
            f"\n✓ Normalization preserves original: {v.data} (expected {original_data})"
        )
        assert v.data == original_data


class TestVectorCopy:
    """Test cases for Vector copying."""

    def test_copy_creates_new_object(self):
        v1 = Vector([1, 2, 3])
        v2 = v1.copy()
        print(f"\n✓ copy() creates new object: v1 is v2 = {v1 is v2} (expected False)")
        assert v1 is not v2
        assert v1.data == v2.data

    def test_copy_independence(self):
        v1 = Vector([1, 2, 3])
        v2 = v1.copy()
        v2[0] = 99
        print(
            f"\n✓ Modifying copy doesn't affect original: v1[0] = {v1[0]} (expected 1)"
        )
        assert v1[0] == 1
        assert v2[0] == 99


class TestVectorConversions:
    """Test cases for Vector conversion methods."""

    def test_to_list(self):
        v = Vector([1, 2, 3])
        result = v.to_list()
        print(f"\n✓ to_list() = {result} (expected [1, 2, 3])")
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_to_list_independence(self):
        v = Vector([1, 2, 3])
        lst = v.to_list()
        lst[0] = 99
        print(f"\n✓ to_list() returns independent copy: v[0] = {v[0]} (expected 1)")
        assert v[0] == 1

    def test_to_tuple(self):
        v = Vector([1, 2, 3])
        result = v.to_tuple()
        print(f"\n✓ to_tuple() = {result} (expected (1, 2, 3))")
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_to_tuple_empty(self):
        v = Vector([])
        result = v.to_tuple()
        print(f"\n✓ to_tuple() on empty vector = {result} (expected ())")
        assert result == ()


class TestVectorIterator:
    """Test cases for Vector iteration."""

    def test_iteration(self):
        v = Vector([1, 2, 3])
        result = [x for x in v]
        print(f"\n✓ Iterating over vector: {result} (expected [1, 2, 3])")
        assert result == [1, 2, 3]

    def test_len(self):
        v = Vector([1, 2, 3, 4, 5])
        print(f"\n✓ len(Vector([1,2,3,4,5])) = {len(v)} (expected 5)")
        assert len(v) == 5


class TestVectorStringRepresentation:
    """Test cases for Vector string representation."""

    def test_str_representation(self):
        v = Vector([1, 2, 3])
        result = str(v)
        print(f"\n✓ str(Vector([1,2,3])) = '{result}' (expected '[1, 2, 3]')")
        assert result == "[1, 2, 3]"

    def test_str_empty_vector(self):
        v = Vector([])
        result = str(v)
        print(f"\n✓ str(Vector([])) = '{result}' (expected '[]')")
        assert result == "[]"
