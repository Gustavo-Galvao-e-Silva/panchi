import random
from fractions import Fraction

import pytest

from panchi.algorithms import (
    CofactorExpansion,
    CramerResult,
    DeterminantReduction,
    adjugate,
    cofactor_expansion,
    cramer,
    determinant,
    determinant_by_reduction,
    solve,
)
from panchi.primitives import Matrix, Vector, identity


def _random_matrices(count: int = 25, seed: int = 7) -> list[Matrix]:
    rng = random.Random(seed)
    matrices = []
    for _ in range(count):
        n = rng.randint(2, 4)
        matrices.append(
            Matrix(
                [
                    [Fraction(rng.randint(-5, 5), rng.randint(1, 3)) for _ in range(n)]
                    for _ in range(n)
                ]
            )
        )
    return matrices


SINGULAR = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
TEXTBOOK = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])


# ==================== COFACTOR EXPANSION ====================


class TestCofactorExpansionValidation:
    """Test type and value validation for cofactor_expansion."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            cofactor_expansion([[1, 2], [3, 4]])

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            cofactor_expansion(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_1x1_raises_value_error(self):
        with pytest.raises(ValueError, match="at least a 2×2"):
            cofactor_expansion(Matrix([[5]]))

    def test_bad_along_raises_value_error(self):
        with pytest.raises(ValueError, match="'row' or 'column'"):
            cofactor_expansion(TEXTBOOK, along="diagonal")

    def test_non_int_index_raises_type_error(self):
        with pytest.raises(TypeError):
            cofactor_expansion(TEXTBOOK, index=1.0)

    def test_out_of_range_index_raises_value_error(self):
        with pytest.raises(ValueError, match="out of range"):
            cofactor_expansion(TEXTBOOK, along="column", index=3)


class TestCofactorExpansionCorrectness:
    """Test cofactor_expansion terms and determinant."""

    def test_returns_cofactor_expansion(self):
        assert isinstance(cofactor_expansion(TEXTBOOK), CofactorExpansion)

    def test_textbook_terms(self):
        result = cofactor_expansion(TEXTBOOK)
        assert [term.value for term in result.terms] == [22, -4, 0]
        assert [term.sign for term in result.terms] == [1, -1, 1]
        assert [term.minor_determinant for term in result.terms] == [11, 4, 1]
        assert result.determinant == 18

    def test_keeps_zero_terms(self):
        result = cofactor_expansion(TEXTBOOK)
        assert len(result.terms) == 3
        assert result.terms[2].entry == 0

    def test_minor_deletes_row_and_column(self):
        term = cofactor_expansion(TEXTBOOK, along="column", index=1).terms[2]
        assert (term.row, term.col) == (2, 1)
        assert term.minor == Matrix([[2, 0], [1, 1]])

    def test_every_row_and_column_agree(self):
        for matrix in _random_matrices():
            expected = determinant(matrix)
            for along in ("row", "column"):
                for index in range(matrix.rows):
                    assert (
                        cofactor_expansion(matrix, along, index).determinant == expected
                    )

    def test_singular_is_zero(self):
        assert cofactor_expansion(SINGULAR).determinant == 0

    def test_str_shows_each_term(self):
        text = str(cofactor_expansion(TEXTBOOK))
        assert "(+1) · 2 · 11 = 22" in text
        assert "(-1) · 1 · 4 = -4" in text
        assert text.endswith("det(A) = 18")

    def test_str_parenthesizes_negatives(self):
        text = str(cofactor_expansion(Matrix([[1, -2], [3, -4]])))
        assert "(-1) · (-2) · 3 = 6" in text

    def test_latex_uses_determinant_bars(self):
        latex = cofactor_expansion(TEXTBOOK)._repr_latex_()
        assert latex.startswith("$$\\begin{aligned}")
        assert "vmatrix" in latex
        assert "&= 18" in latex


# ==================== DETERMINANT BY REDUCTION ====================


class TestDeterminantByReduction:
    """Test determinant_by_reduction."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            determinant_by_reduction([[1, 2], [3, 4]])

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            determinant_by_reduction(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_returns_determinant_reduction(self):
        assert isinstance(determinant_by_reduction(TEXTBOOK), DeterminantReduction)

    def test_swap_flips_sign(self):
        result = determinant_by_reduction(Matrix([[0, 2], [3, 4]]))
        assert result.swaps == 1
        assert result.diagonal == [3, 2]
        assert result.determinant == -6

    def test_matches_determinant_on_random_exact_matrices(self):
        for matrix in _random_matrices():
            assert determinant_by_reduction(matrix).determinant == determinant(matrix)

    def test_stays_exact(self):
        matrix = Matrix([[Fraction(1, 3), 1], [1, 2]])
        result = determinant_by_reduction(matrix).determinant
        assert result == Fraction(-1, 3)
        assert isinstance(result, Fraction)

    def test_singular_is_zero(self):
        assert determinant_by_reduction(SINGULAR).determinant == 0

    def test_1x1(self):
        assert determinant_by_reduction(Matrix([[7]])).determinant == 7

    def test_str_shows_formula(self):
        text = str(determinant_by_reduction(Matrix([[0, 2], [3, 4]])))
        assert "1 step, 1 row swap" in text
        assert text.endswith("det(A) = (-1)^1 · 3 · 2 = -6")

    def test_latex(self):
        latex = determinant_by_reduction(Matrix([[0, 2], [3, 4]]))._repr_latex_()
        assert "(-1)^{1}" in latex
        assert "&= -6" in latex


# ==================== ADJUGATE ====================


class TestAdjugate:
    """Test adjugate."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            adjugate([[1, 2], [3, 4]])

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            adjugate(Matrix([[1, 2, 3], [4, 5, 6]]))

    def test_2x2(self):
        assert adjugate(Matrix([[1, 2], [3, 4]])) == Matrix([[4, -2], [-3, 1]])

    def test_1x1(self):
        assert adjugate(Matrix([[5]])) == Matrix([[1]])

    def test_a_times_adjugate_is_det_times_identity(self):
        for matrix in _random_matrices():
            n = matrix.rows
            assert matrix @ adjugate(matrix) == determinant(matrix) * identity(n)

    def test_singular_still_defined(self):
        assert SINGULAR @ adjugate(SINGULAR) == Matrix(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )


# ==================== CRAMER ====================


class TestCramer:
    """Test cramer."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            cramer([[1, 2], [3, 4]], Vector([1, 2]))

    def test_non_vector_b_raises_type_error(self):
        with pytest.raises(TypeError):
            cramer(Matrix([[1, 2], [3, 4]]), [1, 2])

    def test_wrong_dimension_raises_value_error(self):
        with pytest.raises(ValueError, match="one component per row"):
            cramer(Matrix([[1, 2], [3, 4]]), Vector([1, 2, 3]))

    def test_singular_raises_value_error(self):
        with pytest.raises(ValueError, match="solve"):
            cramer(SINGULAR, Vector([1, 2, 3]))

    def test_returns_cramer_result(self):
        result = cramer(Matrix([[2, 1], [1, 3]]), Vector([3, 5]))
        assert isinstance(result, CramerResult)
        assert result.determinant == 5
        assert result.replaced_determinants == [4, 7]
        assert result.replaced[0] == Matrix([[3, 1], [5, 3]])

    def test_exact_inputs_give_exact_solution(self):
        A = Matrix([[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]])
        result = cramer(A, Vector([Fraction(3), Fraction(5)]))
        assert result.solution == Vector([Fraction(4, 5), Fraction(7, 5)])

    def test_matches_solve(self):
        rng = random.Random(3)
        for matrix in _random_matrices():
            if determinant(matrix) == 0:
                continue
            b = Vector([Fraction(rng.randint(-5, 5)) for _ in range(matrix.rows)])
            assert cramer(matrix, b).solution == solve(matrix, b).solution

    def test_str_and_latex(self):
        result = cramer(Matrix([[2, 1], [1, 3]]), Vector([3, 5]))
        assert "x0 = det(A0) / det(A) = 4 / 5 = 0.8" in str(result)
        assert "\\frac{4}{5}" in result._repr_latex_()
