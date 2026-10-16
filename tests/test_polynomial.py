from fractions import Fraction

import pytest

from panchi.symbolic import (
    Polynomial,
    QuadraticNumber,
    exact_roots,
    quadratic_roots,
    sqrt,
)


class TestConstruction:
    """Test Polynomial construction."""

    def test_coefficients_low_to_high(self):
        p = Polynomial([6, -5, 1])
        assert p.coefficients == [6, -5, 1]
        assert p.degree == 2
        assert p.leading_coefficient == 1

    def test_trailing_zeros_dropped(self):
        assert Polynomial([1, 2, 0, 0]).degree == 1

    def test_zero_polynomial(self):
        assert Polynomial([0, 0]).coefficients == [0]

    def test_float_coefficient_raises(self):
        with pytest.raises(TypeError):
            Polynomial([1.5, 1])


class TestArithmetic:
    """Test polynomial arithmetic and evaluation."""

    def test_add(self):
        assert Polynomial([1, 2]) + Polynomial([3, 0, 1]) == Polynomial([4, 2, 1])

    def test_add_scalar(self):
        assert Polynomial([1, 1]) + 2 == Polynomial([3, 1])
        assert 2 + Polynomial([1, 1]) == Polynomial([3, 1])

    def test_subtract(self):
        assert Polynomial([1, 2]) - Polynomial([1, 2]) == Polynomial([0])

    def test_multiply(self):
        assert Polynomial([-2, 1]) * Polynomial([-3, 1]) == Polynomial([6, -5, 1])

    def test_evaluate(self):
        p = Polynomial([6, -5, 1])
        assert p(2) == 0
        assert p(0) == 6
        assert p(Fraction(1, 2)) == Fraction(15, 4)

    def test_evaluate_at_quadratic_number(self):
        assert Polynomial([-1, -1, 1])((1 + sqrt(5)) / 2) == 0

    def test_divide_by_root(self):
        quotient, remainder = Polynomial([6, -5, 1]).divide_by_root(2)
        assert quotient == Polynomial([-3, 1])
        assert remainder == 0

    def test_divide_by_non_root_has_remainder(self):
        _, remainder = Polynomial([6, -5, 1]).divide_by_root(1)
        assert remainder == 2


class TestRationalRoots:
    """Test the rational root theorem."""

    def test_integer_roots(self):
        assert Polynomial([-6, 11, -6, 1]).rational_roots() == [1, 2, 3]

    def test_fraction_roots(self):
        assert Polynomial([-1, 2]).rational_roots() == [Fraction(1, 2)]

    def test_fraction_coefficients(self):
        roots = Polynomial([Fraction(1, 6), Fraction(-5, 6), 1]).rational_roots()
        assert roots == [Fraction(1, 3), Fraction(1, 2)]

    def test_zero_root(self):
        assert Polynomial([0, -1, 1]).rational_roots() == [0, 1]

    def test_no_rational_roots(self):
        assert Polynomial([-2, 0, 0, 1]).rational_roots() == []

    def test_irrational_coefficient_raises(self):
        with pytest.raises(TypeError):
            Polynomial([sqrt(2), 1]).rational_roots()


class TestRoots:
    """Test the exact root finders."""

    def test_quadratic_rational(self):
        assert quadratic_roots(Polynomial([6, -5, 1])) == [(2, 1), (3, 1)]

    def test_quadratic_repeated(self):
        assert quadratic_roots(Polynomial([1, -2, 1])) == [(1, 2)]

    def test_quadratic_irrational(self):
        roots = quadratic_roots(Polynomial([-1, -1, 1]))
        assert [r for r, _ in roots] == [(1 - sqrt(5)) / 2, (1 + sqrt(5)) / 2]

    def test_quadratic_complex(self):
        roots = quadratic_roots(Polynomial([2, -2, 1]))
        assert [r for r, _ in roots] == [1 - sqrt(-1), 1 + sqrt(-1)]

    def test_quadratic_wrong_degree_raises(self):
        with pytest.raises(ValueError):
            quadratic_roots(Polynomial([1, 1]))

    def test_exact_roots_linear(self):
        assert exact_roots(Polynomial([3, 2])) == [(Fraction(-3, 2), 1)]

    def test_exact_roots_cubic_with_repeat(self):
        assert exact_roots(Polynomial([-4, 8, -5, 1])) == [(1, 1), (2, 2)]

    def test_exact_roots_triple(self):
        assert exact_roots(Polynomial([-1, 3, -3, 1])) == [(1, 3)]

    def test_exact_roots_cubic_with_complex_pair(self):
        roots = exact_roots(Polynomial([-5, 9, -5, 1]))
        assert roots == [
            (1, 1),
            (2 - sqrt(-1), 1),
            (2 + sqrt(-1), 1),
        ]

    def test_exact_roots_negative_leading(self):
        assert exact_roots(Polynomial([22, -35, 14, -1])) == [(1, 1), (2, 1), (11, 1)]

    def test_irreducible_cubic_raises(self):
        with pytest.raises(ValueError, match="no rational roots"):
            exact_roots(Polynomial([-2, 0, 0, 1]))

    def test_constant_raises(self):
        with pytest.raises(ValueError):
            exact_roots(Polynomial([5]))

    def test_roots_are_quadratic_numbers_or_rationals(self):
        for root, _ in exact_roots(Polynomial([-2, 0, 1])):
            assert isinstance(root, QuadraticNumber)


class TestDisplay:
    """Test polynomial printing."""

    def test_str(self):
        assert str(Polynomial([6, -5, 1])) == "λ² - 5λ + 6"

    def test_str_skips_zero_terms(self):
        assert str(Polynomial([-2, 0, 0, 1])) == "λ³ - 2"

    def test_str_negative_leading(self):
        assert str(Polynomial([22, -35, 14, -1])) == "-λ³ + 14λ² - 35λ + 22"

    def test_str_fraction_coefficient(self):
        assert str(Polynomial([Fraction(1, 6), Fraction(-5, 6), 1])) == (
            "λ² - (5/6)λ + 1/6"
        )

    def test_str_constant(self):
        assert str(Polynomial([0])) == "0"

    def test_custom_variable(self):
        assert str(Polynomial([1, 1], variable="x")) == "x + 1"

    def test_latex(self):
        assert Polynomial([6, -5, 1])._latex() == "\\lambda^{2} - 5\\lambda + 6"

    def test_repr(self):
        assert repr(Polynomial([6, -5, 1])) == "Polynomial([6, -5, 1])"
