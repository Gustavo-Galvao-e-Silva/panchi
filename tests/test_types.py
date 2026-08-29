from fractions import Fraction

from panchi import Matrix, Vector
from panchi.types import parse_scalar


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
