import pytest

from panchi import Matrix, Vector, vector_column_matrix


class TestVectorColumnMatrix:
    """Test cases for assembling a matrix from column vectors."""

    def test_columns_match_input_vectors(self):
        a = Vector([1, 2, 3])
        b = Vector([4, 5, 6])
        m = vector_column_matrix([a, b])
        assert m.col_vectors[0].to_list() == [1, 2, 3]
        assert m.col_vectors[1].to_list() == [4, 5, 6]

    def test_shape_is_dims_by_count(self):
        a = Vector([1, 2, 3])
        b = Vector([4, 5, 6])
        assert vector_column_matrix([a, b]).shape == (3, 2)

    def test_single_vector_is_single_column(self):
        m = vector_column_matrix([Vector([1, 2])])
        assert m.shape == (2, 1)
        assert m.col_vectors[0].to_list() == [1, 2]

    def test_result_is_matrix(self):
        m = vector_column_matrix([Vector([1, 2]), Vector([3, 4])])
        assert isinstance(m, Matrix)

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            vector_column_matrix([])

    def test_dimension_mismatch_raises_value_error(self):
        with pytest.raises(ValueError):
            vector_column_matrix([Vector([1, 2]), Vector([1, 2, 3])])
