from fractions import Fraction

import pytest

from panchi import Matrix, Vector, characteristic_polynomial, eigen, identity, inverse
from panchi.algorithms import EigenResult
from panchi.symbolic import Polynomial, QuadraticNumber, sqrt

I = sqrt(-1)

CASES = [
    [[2, 1], [1, 2]],
    [[1, 1], [1, 0]],
    [[0, -1], [1, 0]],
    [[1, 1], [0, 1]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[2, 0, 0], [0, 3, 4], [0, 4, 9]],
    [[1, 0, 0], [0, 1, -2], [0, 1, 3]],
    [[4, 1, 0], [0, 4, 0], [0, 0, 2]],
    [[1, 2, 0], [0, 1, 2], [2, 0, 1]],
    [[Fraction(1, 2), 1], [0, Fraction(1, 3)]],
    [[5]],
]


class TestValidation:
    """Test input validation for eigen() and characteristic_polynomial()."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            eigen("not a matrix")

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            eigen(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_too_large_points_to_eigen_qr(self):
        with pytest.raises(ValueError, match="eigen_qr"):
            eigen(identity(4))

    def test_float_entries_raise(self):
        with pytest.raises(TypeError, match="float"):
            eigen(Matrix([[1.5, 0], [0, 1]]))

    def test_irrational_entries_raise(self):
        with pytest.raises(TypeError):
            eigen(Matrix([[sqrt(2), 0], [0, 1]]))

    def test_irreducible_cubic_raises_with_hint(self):
        companion = Matrix([[0, 0, 2], [1, 0, 0], [0, 1, 0]])
        with pytest.raises(ValueError, match="eigen_qr"):
            eigen(companion)


class TestCharacteristicPolynomial:
    """Test characteristic_polynomial()."""

    def test_2x2(self):
        assert characteristic_polynomial(Matrix([[2, 1], [1, 2]])) == Polynomial(
            [3, -4, 1]
        )

    def test_3x3_is_det_a_minus_lambda_i(self):
        p = characteristic_polynomial(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]]))
        assert p == Polynomial([22, -35, 14, -1])

    def test_1x1(self):
        assert characteristic_polynomial(Matrix([[5]])) == Polynomial([5, -1])

    def test_constant_term_is_determinant(self):
        m = Matrix([[1, 2, 0], [0, 1, 2], [2, 0, 1]])
        assert characteristic_polynomial(m)(0) == 9

    def test_fraction_entries(self):
        p = characteristic_polynomial(
            Matrix([[Fraction(1, 2), 1], [0, Fraction(1, 3)]])
        )
        assert p == Polynomial([Fraction(1, 6), Fraction(-5, 6), 1])


class TestEigenvalues:
    """Test the exact eigenvalues."""

    def test_returns_eigen_result(self):
        assert isinstance(eigen(Matrix([[2, 1], [1, 2]])), EigenResult)

    def test_rational(self):
        assert eigen(Matrix([[2, 1], [1, 2]])).eigenvalues == [1, 3]

    def test_irrational(self):
        values = eigen(Matrix([[1, 1], [1, 0]])).eigenvalues
        assert values == [(1 - sqrt(5)) / 2, (1 + sqrt(5)) / 2]

    def test_complex(self):
        result = eigen(Matrix([[0, -1], [1, 0]]))
        assert result.eigenvalues == [-I, I]
        assert result.has_complex_eigenvalues

    def test_real_has_no_complex_eigenvalues(self):
        assert not eigen(Matrix([[1, 1], [1, 0]])).has_complex_eigenvalues

    def test_3x3_symmetric(self):
        assert eigen(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]])).eigenvalues == [
            1,
            2,
            11,
        ]

    def test_3x3_with_complex_pair(self):
        values = eigen(Matrix([[1, 0, 0], [0, 1, -2], [0, 1, 3]])).eigenvalues
        assert values == [1, 2 - I, 2 + I]

    def test_fraction_eigenvalues(self):
        values = eigen(Matrix([[Fraction(1, 2), 1], [0, Fraction(1, 3)]])).eigenvalues
        assert values == [Fraction(1, 3), Fraction(1, 2)]

    def test_multiplicities(self):
        result = eigen(Matrix([[4, 1, 0], [0, 4, 0], [0, 0, 2]]))
        assert result.eigenvalues == [2, 4]
        assert result.algebraic_multiplicities == [1, 2]
        assert result.geometric_multiplicities == [1, 1]

    def test_sum_equals_trace(self):
        m = Matrix([[1, 2, 0], [0, 1, 2], [2, 0, 1]])
        result = eigen(m)
        total = sum(
            v * k
            for v, k in zip(
                result.eigenvalues, result.algebraic_multiplicities, strict=True
            )
        )
        assert total == m.trace


class TestEigenvectors:
    """Test that every eigenvector satisfies A v = λ v exactly."""

    @pytest.mark.parametrize("data", CASES)
    def test_av_equals_lambda_v(self, data):
        m = Matrix(data)
        for value, vector in eigen(m).pairs:
            assert m @ vector == value * vector

    @pytest.mark.parametrize("data", CASES)
    def test_eigenvectors_are_nonzero(self, data):
        for _, vector in eigen(Matrix(data)).pairs:
            assert any(component != 0 for component in vector)

    def test_denominators_cleared(self):
        result = eigen(Matrix([[Fraction(1, 2), 1], [0, Fraction(1, 3)]]))
        for _, vector in result.pairs:
            assert all(isinstance(c, int) for c in vector)

    def test_rational_case_has_integer_components(self):
        result = eigen(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]]))
        assert [vector for _, vector in result.pairs] == [
            Vector([0, -2, 1]),
            Vector([1, 0, 0]),
            Vector([0, 1, 2]),
        ]

    def test_complex_eigenvectors(self):
        result = eigen(Matrix([[0, -1], [1, 0]]))
        assert [vector for _, vector in result.pairs] == [
            Vector([-I, 1]),
            Vector([I, 1]),
        ]

    def test_repeated_eigenvalue_full_eigenspace(self):
        result = eigen(identity(3))
        assert result.eigenvalues == [1]
        assert result.algebraic_multiplicities == [3]
        assert result.geometric_multiplicities == [3]


class TestDiagonalization:
    """Test is_diagonalizable and diagonalization()."""

    @pytest.mark.parametrize("data", CASES)
    def test_ap_equals_pd(self, data):
        m = Matrix(data)
        result = eigen(m)
        if result.is_diagonalizable:
            p, d = result.diagonalization()
            assert m @ p == p @ d

    def test_reconstruction_rational(self):
        m = Matrix([[2, 1], [1, 2]])
        p, d = eigen(m).diagonalization()
        assert p @ d @ inverse(p).inverse == m

    def test_diagonal_holds_eigenvalues(self):
        _, d = eigen(Matrix([[2, 1], [1, 2]])).diagonalization()
        assert d == Matrix([[1, 0], [0, 3]])

    def test_defective_not_diagonalizable(self):
        assert eigen(Matrix([[1, 1], [0, 1]])).is_diagonalizable is False

    def test_defective_diagonalization_raises(self):
        with pytest.raises(ValueError, match="not diagonalizable"):
            eigen(Matrix([[1, 1], [0, 1]])).diagonalization()

    def test_rotation_diagonalizable_over_complex(self):
        assert eigen(Matrix([[0, -1], [1, 0]])).is_diagonalizable


class TestDisplay:
    """Test str, repr and LaTeX output."""

    def test_str_shows_derivation(self):
        text = str(eigen(Matrix([[2, 1], [1, 2]])))
        assert "det(A - λI) = λ² - 4λ + 3 = (λ - 1)(λ - 3)" in text
        assert "λ = 1  (algebraic multiplicity 1, geometric multiplicity 1)" in text
        assert "Diagonalizable: yes" in text

    def test_str_factored_3x3_has_sign(self):
        text = str(eigen(Matrix([[4, 1, 0], [0, 4, 0], [0, 0, 2]])))
        assert "= -(λ - 2)(λ - 4)²" in text
        assert "Diagonalizable: no" in text

    def test_str_complex_factor(self):
        assert "(λ + i)(λ - i)" in str(eigen(Matrix([[0, -1], [1, 0]])))

    def test_repr(self):
        assert repr(eigen(Matrix([[2, 1], [1, 2]]))) == (
            "EigenResult(shape=2×2, eigenvalues=[1, 3], diagonalizable=True)"
        )

    def test_latex(self):
        out = eigen(Matrix([[1, 1], [1, 0]]))._repr_latex_()
        assert out.startswith("$$\\begin{aligned}")
        assert "\\frac{1 + \\sqrt{5}}{2}" in out
        assert "\\operatorname{span}" in out

    def test_eigenvalue_types(self):
        result = eigen(Matrix([[1, 1], [1, 0]]))
        assert all(isinstance(v, QuadraticNumber) for v in result.eigenvalues)
