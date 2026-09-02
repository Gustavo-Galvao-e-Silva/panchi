from fractions import Fraction

from panchi import Matrix, Vector, VectorSpace, exact_vector
from panchi.algorithms import eigen, inverse, lu, qr_decomposition, solve
from panchi.algorithms.reductions import rref
from panchi.algorithms.row_operations import RowAdd, RowScale, RowSwap
from panchi.utils.latex import (
    matrix_to_latex,
    row_op_to_latex,
    scalar_to_latex,
    vector_to_latex,
)


class TestScalarToLatex:
    def test_int_passthrough(self):
        assert scalar_to_latex(5) == "5"

    def test_float_passthrough(self):
        assert scalar_to_latex(1.5) == "1.5"

    def test_integer_valued_fraction_is_bare(self):
        assert scalar_to_latex(Fraction(6, 2)) == "3"

    def test_proper_fraction(self):
        assert scalar_to_latex(Fraction(1, 3)) == "\\frac{1}{3}"

    def test_negative_fraction_sign_outside(self):
        assert scalar_to_latex(Fraction(-2, 5)) == "-\\frac{2}{5}"


class TestBuildersLatex:
    def test_vector_to_latex_is_column(self):
        out = vector_to_latex(Vector([1, 2, 3]))
        assert out.startswith("\\begin{bmatrix}")
        assert "1 \\\\ 2 \\\\ 3" in out

    def test_matrix_to_latex_rows_and_cols(self):
        out = matrix_to_latex(Matrix([[1, 2], [3, 4]]))
        assert "1 & 2 \\\\ 3 & 4" in out

    def test_matrix_fraction_renders_frac(self):
        out = matrix_to_latex(Matrix([["1/2", 2], [3, 4]]))
        assert "\\frac{1}{2}" in out

    def test_row_op_labels(self):
        assert "\\leftrightarrow" in row_op_to_latex(RowSwap(0, 1))
        assert "\\to" in row_op_to_latex(RowScale(0, 5))
        assert row_op_to_latex(RowAdd(target=1, source=0, scalar=-3)).startswith(
            "R_{1}"
        )


class TestPrimitiveReprLatex:
    def test_vector(self):
        assert (
            Vector([1, 2])._repr_latex_()
            == "$\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}$"
        )

    def test_matrix_wrapped_in_math(self):
        out = Matrix([[1, 2], [3, 4]])._repr_latex_()
        assert out.startswith("$") and out.endswith("$")
        assert "bmatrix" in out

    def test_exact_vector_shows_fraction(self):
        assert "\\frac{1}{3}" in exact_vector(["1/3", "2/3"])._repr_latex_()

    def test_vectorspace_span(self):
        out = VectorSpace([Vector([1, 0]), Vector([0, 1])])._repr_latex_()
        assert "\\operatorname{span}" in out
        assert out.count("bmatrix") == 4  # two open + two close tokens per vector


class TestResultReprLatex:
    def test_reduction_is_arrow_sequence(self):
        out = rref(Matrix([[1, 2], [3, 4]]))._repr_latex_()
        assert "\\begin{aligned}" in out
        assert "\\xrightarrow" in out

    def test_solve_unique(self):
        out = solve(Matrix([[1, 2], [3, 4]]), Vector([5, 6]))._repr_latex_()
        assert out.startswith("$x = ")

    def test_solve_infinite_has_parameter(self):
        out = solve(Matrix([[1, 1], [2, 2]]), Vector([1, 2]))._repr_latex_()
        assert "t\\," in out

    def test_solve_inconsistent(self):
        out = solve(Matrix([[1, 1], [2, 2]]), Vector([1, 5]))._repr_latex_()
        assert "No solution" in out

    def test_lu_factorization(self):
        out = lu(Matrix([[2, 1], [6, 4]]))._repr_latex_()
        assert " = " in out and "bmatrix" in out

    def test_qr_factorization(self):
        out = qr_decomposition(Matrix([[1, 0], [0, 1]]))._repr_latex_()
        assert " = " in out

    def test_inverse(self):
        out = inverse(Matrix([[1, 2], [3, 4]]))._repr_latex_()
        assert "^{-1}" in out

    def test_eigen_pairs(self):
        out = eigen(Matrix([[2, 1], [1, 2]]))._repr_latex_()
        assert "\\lambda_{1}" in out


class TestStrUnchanged:
    """Rich display is additive: __str__ must be unaffected."""

    def test_matrix_str(self):
        assert str(Matrix([[1, 2], [3, 4]])) == "[[1, 2],\n [3, 4]]"

    def test_vector_str(self):
        assert str(Vector([1, 2, 3])) == "[1, 2, 3]"

    def test_repr_latex_differs_from_str(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m._repr_latex_() != str(m)
