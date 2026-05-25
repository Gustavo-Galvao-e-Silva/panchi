from fractions import Fraction

import panchi as pan
from panchi import (
    Vector,
    Matrix,
    exact_vector,
    exact_matrix,
    dot,
    cross,
    identity,
    solve,
    inverse,
    determinant_lu,
)
from panchi.algorithms.reductions import rref


class TestVectorFractionConstruction:
    def test_string_fractions(self):
        v = Vector(["1/3", "2/3", "1/2"])
        assert v[0] == Fraction(1, 3)
        assert v[1] == Fraction(2, 3)
        assert v[2] == Fraction(1, 2)

    def test_mixed_types(self):
        v = Vector(["1/3", 2, 3.0])
        assert v[0] == Fraction(1, 3)
        assert v[1] == 2
        assert isinstance(v[1], int)
        assert v[2] == 3.0
        assert isinstance(v[2], float)

    def test_integer_strings(self):
        v = Vector(["1", "2", "3"])
        assert v[0] == 1
        assert isinstance(v[0], int)

    def test_fraction_objects(self):
        v = Vector([Fraction(1, 3), Fraction(2, 3)])
        assert v[0] == Fraction(1, 3)

    def test_invalid_string_raises(self):
        try:
            Vector(["hello"])
            assert False, "Should have raised"
        except TypeError:
            pass

    def test_negative_fractions(self):
        v = Vector(["-1/3", "-2/5"])
        assert v[0] == Fraction(-1, 3)
        assert v[1] == Fraction(-2, 5)


class TestMatrixFractionConstruction:
    def test_string_fractions(self):
        m = Matrix([["1/2", "1/3"], ["1/4", "1/5"]])
        assert m[0][0] == Fraction(1, 2)
        assert m[0][1] == Fraction(1, 3)
        assert m[1][0] == Fraction(1, 4)
        assert m[1][1] == Fraction(1, 5)

    def test_mixed_types(self):
        m = Matrix([["1/3", 2], [0, "3/4"]])
        assert m[0][0] == Fraction(1, 3)
        assert m[0][1] == 2
        assert m[1][0] == 0
        assert m[1][1] == Fraction(3, 4)

    def test_invalid_string_raises(self):
        try:
            Matrix([["abc", 1], [2, 3]])
            assert False, "Should have raised"
        except TypeError:
            pass


class TestExactFactories:
    def test_exact_vector(self):
        v = exact_vector([1, 2, 3])
        assert all(isinstance(v[i], Fraction) for i in range(v.dims))
        assert v[0] == Fraction(1)
        assert v[1] == Fraction(2)

    def test_exact_matrix(self):
        m = exact_matrix([[1, 2], [3, 4]])
        assert isinstance(m[0][0], Fraction)
        assert m[0][0] == Fraction(1)

    def test_exact_vector_preserves_fractions(self):
        v = exact_vector([Fraction(1, 3), 2])
        assert v[0] == Fraction(1, 3)
        assert v[1] == Fraction(2)


class TestFractionArithmetic:
    def test_vector_addition(self):
        v1 = Vector(["1/3", "1/3", "1/3"])
        v2 = Vector(["2/3", "2/3", "2/3"])
        result = v1 + v2
        assert result[0] == Fraction(1, 1)
        assert result[1] == Fraction(1, 1)

    def test_vector_subtraction(self):
        v1 = Vector(["1/2", "3/4"])
        v2 = Vector(["1/4", "1/4"])
        result = v1 - v2
        assert result[0] == Fraction(1, 4)
        assert result[1] == Fraction(1, 2)

    def test_scalar_multiplication(self):
        v = Vector(["1/3", "2/3"])
        result = Fraction(3) * v
        assert result[0] == Fraction(1, 1)
        assert result[1] == Fraction(2, 1)

    def test_scalar_division(self):
        v = exact_vector([1, 2, 3])
        result = v / Fraction(3)
        assert result[0] == Fraction(1, 3)
        assert result[1] == Fraction(2, 3)
        assert result[2] == Fraction(1, 1)

    def test_matrix_addition(self):
        m1 = Matrix([["1/2", "1/3"], ["1/4", "1/5"]])
        m2 = Matrix([["1/2", "2/3"], ["3/4", "4/5"]])
        result = m1 + m2
        assert result[0][0] == Fraction(1, 1)
        assert result[0][1] == Fraction(1, 1)
        assert result[1][0] == Fraction(1, 1)
        assert result[1][1] == Fraction(1, 1)

    def test_matrix_scalar_multiplication(self):
        m = Matrix([["1/3", "2/3"], ["1/6", "5/6"]])
        result = Fraction(6) * m
        assert result[0][0] == Fraction(2)
        assert result[0][1] == Fraction(4)
        assert result[1][0] == Fraction(1)
        assert result[1][1] == Fraction(5)

    def test_matrix_vector_multiplication(self):
        m = exact_matrix([[1, 2], [3, 4]])
        v = exact_vector([1, 1])
        result = m @ v
        assert result[0] == Fraction(3)
        assert result[1] == Fraction(7)

    def test_matrix_matrix_multiplication(self):
        m1 = exact_matrix([[1, 2], [3, 4]])
        m2 = exact_matrix([[5, 6], [7, 8]])
        result = m1 @ m2
        assert result[0][0] == Fraction(19)
        assert result[0][1] == Fraction(22)


class TestFractionDotCross:
    def test_dot_product(self):
        v1 = Vector(["1/2", "1/3", "1/4"])
        v2 = Vector(["2", "3", "4"])
        result = dot(v1, v2)
        assert result == Fraction(1) + Fraction(1) + Fraction(1)
        assert result == Fraction(3)

    def test_cross_product(self):
        v1 = exact_vector([1, 0, 0])
        v2 = exact_vector([0, 1, 0])
        result = cross(v1, v2)
        assert result == Vector([0, 0, 1])


class TestFractionAlgorithms:
    def test_rref_exact(self):
        m = exact_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        reduction = rref(m)
        rref_matrix = reduction.result
        assert rref_matrix[0][0] == Fraction(1)
        assert rref_matrix[1][1] == Fraction(1)
        assert rref_matrix[2][2] == Fraction(1)
        assert rref_matrix[0][1] == Fraction(0)
        assert rref_matrix[0][2] == Fraction(0)

    def test_determinant_exact(self):
        m = exact_matrix([[1, 2], [3, 4]])
        det = m.determinant
        assert det == Fraction(-2)

    def test_inverse_exact(self):
        m = exact_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        inv_result = inverse(m)
        inv_matrix = inv_result.inverse

        product = m @ inv_matrix
        n = m.rows
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert product[i][j] == Fraction(1)
                else:
                    assert product[i][j] == Fraction(0)

    def test_solve_exact(self):
        A = exact_matrix([[2, 1], [5, 3]])
        b = exact_vector([1, 2])
        result = solve(A, b)
        assert result.solution[0] == Fraction(1)
        assert result.solution[1] == Fraction(-1)

    def test_determinant_lu_exact(self):
        m = exact_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        det = determinant_lu(m)
        assert det == Fraction(-3)


class TestFractionDisplay:
    def test_vector_str(self):
        v = Vector(["1/3", "2/3", 1])
        assert str(v) == "[1/3, 2/3, 1]"

    def test_vector_repr(self):
        v = Vector(["1/3", "2/3"])
        assert "Fraction" in repr(v)

    def test_matrix_str(self):
        m = Matrix([["1/2", "1/3"], ["1/4", "1/5"]])
        s = str(m)
        assert "1/2" in s
        assert "1/3" in s


class TestBackwardCompatibility:
    def test_int_stays_int(self):
        v = Vector([1, 2, 3])
        result = v + Vector([4, 5, 6])
        assert isinstance(result[0], int)
        assert result[0] == 5

    def test_float_stays_float(self):
        v = Vector([1.0, 2.0, 3.0])
        result = v + Vector([4.0, 5.0, 6.0])
        assert isinstance(result[0], float)

    def test_int_matrix_stays_int(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m + Matrix([[5, 6], [7, 8]])
        assert isinstance(result[0][0], int)
