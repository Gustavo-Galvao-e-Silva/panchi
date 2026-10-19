from fractions import Fraction

import pytest

from panchi import Matrix, Vector
from panchi.utils.types import parse_scalar


class TestParseScalarValid:
    def test_int_passthrough(self):
        assert parse_scalar(3) == 3
        assert isinstance(parse_scalar(3), int)

    def test_float_passthrough(self):
        assert parse_scalar(2.5) == 2.5
        assert isinstance(parse_scalar(2.5), float)

    def test_fraction_passthrough(self):
        assert parse_scalar(Fraction(1, 3)) == Fraction(1, 3)

    def test_string_int(self):
        assert parse_scalar("42") == 42
        assert isinstance(parse_scalar("42"), int)

    def test_string_float(self):
        assert parse_scalar("2.5") == 2.5
        assert isinstance(parse_scalar("2.5"), float)

    def test_string_fraction(self):
        assert parse_scalar("1/3") == Fraction(1, 3)


class TestParseScalarBoolRejected:
    def test_true_raises(self):
        try:
            parse_scalar(True)
            raise AssertionError("Should have raised")
        except TypeError:
            pass

    def test_false_raises(self):
        try:
            parse_scalar(False)
            raise AssertionError("Should have raised")
        except TypeError:
            pass

    def test_error_is_informative(self):
        try:
            parse_scalar(True)
            raise AssertionError("Should have raised")
        except TypeError as exc:
            assert "bool" in str(exc)
            assert "1 or 0" in str(exc)

    def test_vector_rejects_bools(self):
        try:
            Vector([True, False])
            raise AssertionError("Should have raised")
        except TypeError:
            pass

    def test_matrix_rejects_bools(self):
        try:
            Matrix([[True, False], [False, True]])
            raise AssertionError("Should have raised")
        except TypeError:
            pass


class TestParseScalarInvalidStrings:
    def test_non_numeric_string_raises(self):
        with pytest.raises(TypeError, match="Cannot convert string 'abc' to a number"):
            parse_scalar("abc")

    def test_empty_string_raises(self):
        with pytest.raises(TypeError, match="Cannot convert string '' to a number"):
            parse_scalar("")

    def test_zero_denominator_fraction_raises(self):
        with pytest.raises(TypeError, match="Cannot convert string '1/0' to a number"):
            parse_scalar("1/0")

    def test_malformed_fraction_raises(self):
        with pytest.raises(
            TypeError, match="Cannot convert string '1/2/3' to a number"
        ):
            parse_scalar("1/2/3")

    def test_malformed_float_raises(self):
        with pytest.raises(
            TypeError, match="Cannot convert string '1.2.3' to a number"
        ):
            parse_scalar("1.2.3")


class TestParseScalarInvalidTypes:
    def test_none_raises(self):
        with pytest.raises(TypeError, match="Cannot convert NoneType to a number"):
            parse_scalar(None)

    def test_list_raises(self):
        with pytest.raises(TypeError, match="Cannot convert list to a number"):
            parse_scalar([1, 2])

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="Cannot convert dict to a number"):
            parse_scalar({"a": 1})
