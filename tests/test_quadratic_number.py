from fractions import Fraction

import pytest

from panchi import Matrix, Vector
from panchi.symbolic import QuadraticNumber, sqrt

GOLDEN = QuadraticNumber(Fraction(1, 2), Fraction(1, 2), 5)
I = QuadraticNumber(0, 1, -1)


class TestConstruction:
    """Test construction and normalization of QuadraticNumber."""

    def test_stores_parts_as_fractions(self):
        q = QuadraticNumber(1, 2, 3)
        assert q.rational == Fraction(1)
        assert q.coefficient == Fraction(2)
        assert q.radicand == 3

    def test_radicand_reduced_to_squarefree(self):
        q = QuadraticNumber(0, 1, 12)
        assert q.coefficient == 2
        assert q.radicand == 3

    def test_negative_radicand_reduced(self):
        q = QuadraticNumber(0, 1, -8)
        assert q.coefficient == 2
        assert q.radicand == -2

    def test_perfect_square_radicand_raises(self):
        with pytest.raises(ValueError):
            QuadraticNumber(0, 1, 9)

    def test_radicand_one_raises(self):
        with pytest.raises(ValueError):
            QuadraticNumber(0, 1, 1)

    def test_zero_radicand_raises(self):
        with pytest.raises(ValueError):
            QuadraticNumber(0, 1, 0)

    def test_float_part_raises(self):
        with pytest.raises(TypeError):
            QuadraticNumber(0.5, 1, 2)

    def test_non_int_radicand_raises(self):
        with pytest.raises(TypeError):
            QuadraticNumber(0, 1, 2.0)


class TestSqrt:
    """Test the exact sqrt() factory."""

    def test_perfect_square_int(self):
        assert sqrt(9) == 3
        assert isinstance(sqrt(9), int)

    def test_perfect_square_fraction(self):
        assert sqrt(Fraction(4, 9)) == Fraction(2, 3)

    def test_zero(self):
        assert sqrt(0) == 0

    def test_simplest_radical_form(self):
        assert sqrt(12) == QuadraticNumber(0, 2, 3)

    def test_fraction_rationalizes_denominator(self):
        assert sqrt(Fraction(1, 2)) == QuadraticNumber(0, Fraction(1, 2), 2)

    def test_negative_gives_imaginary(self):
        assert sqrt(-1) == I
        assert sqrt(-4) == 2 * I

    def test_square_of_sqrt(self):
        assert sqrt(7) * sqrt(7) == 7

    def test_float_raises(self):
        with pytest.raises(TypeError):
            sqrt(2.0)

    def test_bool_raises(self):
        with pytest.raises(TypeError):
            sqrt(True)


class TestArithmetic:
    """Test the field operations."""

    def test_add_with_int(self):
        assert sqrt(2) + 1 == QuadraticNumber(1, 1, 2)
        assert 1 + sqrt(2) == QuadraticNumber(1, 1, 2)

    def test_add_collapses_to_rational(self):
        result = (1 + sqrt(2)) + (1 - sqrt(2))
        assert result == 2
        assert isinstance(result, int)

    def test_subtract(self):
        assert (3 + sqrt(2)) - (1 + sqrt(2)) == 2
        assert 1 - sqrt(2) == QuadraticNumber(1, -1, 2)

    def test_multiply(self):
        assert (1 + sqrt(2)) * (1 + sqrt(2)) == 3 + 2 * sqrt(2)

    def test_multiply_conjugates_is_rational(self):
        result = (3 + 2 * I) * (3 - 2 * I)
        assert result == 13
        assert isinstance(result, int)

    def test_i_squared(self):
        assert I * I == -1

    def test_golden_ratio_identity(self):
        assert GOLDEN * GOLDEN == GOLDEN + 1

    def test_divide_by_quadratic(self):
        assert 1 / (1 + I) == QuadraticNumber(Fraction(1, 2), Fraction(-1, 2), -1)

    def test_divide_round_trip(self):
        x = 2 + 3 * sqrt(5)
        y = Fraction(1, 3) - sqrt(5)
        assert (x / y) * y == x

    def test_divide_by_rational(self):
        assert (2 + 4 * sqrt(3)) / 2 == 1 + 2 * sqrt(3)

    def test_divide_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            sqrt(2) / 0

    def test_negation(self):
        assert -(1 + sqrt(2)) == QuadraticNumber(-1, -1, 2)

    def test_power(self):
        assert I**4 == 1
        assert sqrt(2) ** 3 == 2 * sqrt(2)

    def test_negative_power(self):
        assert sqrt(2) ** -2 == Fraction(1, 2)

    def test_mixed_radicands_raises(self):
        with pytest.raises(ValueError, match="same radicand"):
            sqrt(2) + sqrt(3)

    def test_float_operand_raises(self):
        with pytest.raises(TypeError, match="float"):
            sqrt(2) + 0.5
        with pytest.raises(TypeError, match="float"):
            0.5 * sqrt(2)


class TestComparison:
    """Test equality, hashing and ordering."""

    def test_equal_to_int_when_rational(self):
        assert QuadraticNumber(3, 0, 2) == 3

    def test_not_equal_to_rational_part(self):
        assert 1 + sqrt(2) != 1

    def test_hash_matches_rational(self):
        assert hash(QuadraticNumber(Fraction(1, 2), 0, 2)) == hash(Fraction(1, 2))

    def test_usable_as_dict_key(self):
        assert {1 + sqrt(2): "a"}[QuadraticNumber(1, 1, 2)] == "a"

    def test_bool(self):
        assert bool(sqrt(2))
        assert not QuadraticNumber(0, 0, 2)

    def test_ordering_real(self):
        assert sqrt(2) > 1
        assert sqrt(2) < Fraction(3, 2)
        assert 1 - sqrt(2) < 0
        assert GOLDEN >= 1

    def test_ordering_close_values(self):
        assert 99 - 70 * sqrt(2) > 0
        assert 70 * sqrt(2) - 99 < 0

    def test_ordering_with_float(self):
        assert sqrt(2) > 0.0
        assert sqrt(2) < 1.5

    def test_ordering_complex_raises(self):
        with pytest.raises(TypeError):
            _ = I < 1


class TestConversions:
    """Test conjugate, parts, abs and numeric conversion."""

    def test_conjugate(self):
        assert (3 + 2 * I).conjugate() == 3 - 2 * I

    def test_norm(self):
        assert (3 + 2 * I).norm() == 13
        assert (1 + sqrt(2)).norm() == -1

    def test_is_real(self):
        assert (1 + sqrt(2)).is_real
        assert not (1 + I).is_real

    def test_real_and_imag(self):
        z = 1 + sqrt(-2)
        assert z.real == 1
        assert z.imag == sqrt(2)

    def test_imag_of_gaussian_number(self):
        z = 3 + 2 * sqrt(-1)
        assert z.imag == 2
        assert isinstance(z.imag, int)
        assert (Fraction(1, 2) - sqrt(-1) / 3).imag == Fraction(-1, 3)

    def test_abs_real(self):
        assert abs(1 - sqrt(2)) == sqrt(2) - 1

    def test_abs_complex(self):
        assert abs(3 + 4 * I) == 5
        assert abs(1 + I) == sqrt(2)

    def test_float(self):
        assert float(sqrt(2)) == pytest.approx(2**0.5)

    def test_float_of_complex_raises(self):
        with pytest.raises(TypeError):
            float(1 + I)

    def test_complex(self):
        assert complex(1 + 2 * sqrt(-3)) == pytest.approx(complex(1, 2 * 3**0.5))


class TestDisplay:
    """Test str, repr and LaTeX output."""

    def test_str_common_denominator(self):
        assert str(GOLDEN) == "(1 + √5)/2"

    def test_str_radical_only(self):
        assert str(sqrt(Fraction(1, 2))) == "√2/2"
        assert str(-sqrt(5)) == "-√5"

    def test_str_complex(self):
        assert str(3 - 2 * I) == "3 - 2i"
        assert str(I) == "i"
        assert str(-I) == "-i"

    def test_str_complex_with_radical(self):
        assert str(1 - 2 * sqrt(-3)) == "1 - 2i√3"

    def test_repr(self):
        assert repr(GOLDEN) == "QuadraticNumber(Fraction(1, 2), Fraction(1, 2), 5)"

    def test_latex(self):
        assert GOLDEN._latex() == "\\frac{1 + \\sqrt{5}}{2}"
        assert (1 + sqrt(-2))._latex() == "1 + i\\sqrt{2}"
        assert GOLDEN._repr_latex_() == "$\\frac{1 + \\sqrt{5}}{2}$"


class TestWithPrimitives:
    """Test that QuadraticNumber works as a Vector/Matrix scalar."""

    def test_vector_of_quadratic_numbers(self):
        v = Vector([1 + sqrt(2), 2])
        assert str(v) == "[1 + √2, 2]"

    def test_scalar_times_vector(self):
        assert sqrt(2) * Vector([1, sqrt(2)]) == Vector([sqrt(2), 2])

    def test_matrix_vector_product(self):
        m = Matrix([[0, -1], [1, 0]])
        v = Vector([I, 1])
        assert m @ v == I * v

    def test_matrix_latex(self):
        assert "\\sqrt{2}" in Matrix([[sqrt(2), 0], [0, 1]])._repr_latex_()


class TestReprRoundTrip:
    """Test that repr reconstructs the value."""

    def test_repr_whole_parts_are_ints(self):
        assert repr(2 - sqrt(-1)) == "QuadraticNumber(2, -1, -1)"

    def test_repr_round_trip(self):
        assert eval(repr(GOLDEN)) == GOLDEN
