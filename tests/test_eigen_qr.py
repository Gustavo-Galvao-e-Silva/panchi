import math

import pytest

from panchi import Matrix, determinant, eigen_qr
from panchi.algorithms import QREigenResult


def _sorted_round(values, digits=6):
    return sorted(round(v, digits) for v in values)


class TestEigenValidation:
    """Test input validation for eigen_qr()."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            eigen_qr("not a matrix")

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            eigen_qr(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestQREigenResultObject:
    """Test the QREigenResult object's structure and attributes."""

    def test_returns_eigen_result_instance(self):
        assert isinstance(eigen_qr(Matrix([[2, 1], [1, 2]])), QREigenResult)

    def test_original_is_preserved(self):
        m = Matrix([[2, 1], [1, 2]])
        assert eigen_qr(m).original == m

    def test_converges_for_symmetric_matrix(self):
        assert eigen_qr(Matrix([[2, 1], [1, 2]])).converged is True

    def test_eigenvalue_count_matches_dimension(self):
        assert len(eigen_qr(Matrix([[2, 1], [1, 2]])).eigenvalues) == 2


class TestEigenvalueCorrectness:
    """Test the numerical correctness of the computed eigenvalues."""

    def test_diagonal_matrix(self):
        result = eigen_qr(Matrix([[2, 0], [0, 3]]))
        assert _sorted_round(result.eigenvalues) == [2.0, 3.0]

    def test_symmetric_2x2(self):
        result = eigen_qr(Matrix([[2, 1], [1, 2]]))
        assert _sorted_round(result.eigenvalues) == [1.0, 3.0]

    def test_symmetric_3x3(self):
        # Eigenvalues of [[2,0,0],[0,3,4],[0,4,9]] are 2, 1, 11.
        result = eigen_qr(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]]))
        assert _sorted_round(result.eigenvalues) == [1.0, 2.0, 11.0]

    def test_sum_of_eigenvalues_equals_trace(self):
        m = Matrix([[4, 1], [2, 3]])
        result = eigen_qr(m)
        assert sum(result.eigenvalues) == pytest.approx(m.trace, abs=1e-6)

    def test_product_of_eigenvalues_equals_determinant(self):
        m = Matrix([[4, 1], [2, 3]])
        result = eigen_qr(m)
        assert math.prod(result.eigenvalues) == pytest.approx(determinant(m), abs=1e-6)


class TestEigenvectorCorrectness:
    """Test that computed eigenvectors satisfy A @ v == λ * v."""

    def _assert_eigenpairs(self, matrix):
        result = eigen_qr(matrix)
        assert len(result.eigenvectors) == matrix.rows
        for value, vector in result.pairs:
            av = matrix @ vector
            lv = value * vector
            for i in range(vector.dims):
                assert av[i] == pytest.approx(lv[i], abs=1e-6)

    def test_symmetric_2x2_eigenpairs(self):
        self._assert_eigenpairs(Matrix([[2, 1], [1, 2]]))

    def test_symmetric_3x3_eigenpairs(self):
        self._assert_eigenpairs(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]]))

    def test_non_symmetric_eigenpairs(self):
        self._assert_eigenpairs(Matrix([[2, 1], [0, 3]]))

    def test_eigenvectors_are_unit_length(self):
        result = eigen_qr(Matrix([[2, 1], [1, 2]]))
        for vector in result.eigenvectors:
            assert vector.magnitude == pytest.approx(1.0, abs=1e-9)

    def test_pairs_align_values_and_vectors(self):
        result = eigen_qr(Matrix([[2, 1], [1, 2]]))
        assert result.pairs == list(
            zip(result.eigenvalues, result.eigenvectors, strict=False)
        )

    def test_str_lists_eigenvectors_when_present(self):
        text = str(eigen_qr(Matrix([[2, 1], [1, 2]])))
        assert "λ =" in text
        assert "v =" in text


class TestEigenConvergence:
    """Test convergence reporting for non-convergent matrices."""

    def test_rotation_matrix_does_not_converge(self):
        # A 90° rotation has complex eigenvalues; unshifted QR will not settle.
        result = eigen_qr(Matrix([[0, -1], [1, 0]]), max_iterations=200)
        assert result.converged is False

    def test_non_converged_has_no_eigenvectors(self):
        result = eigen_qr(Matrix([[0, -1], [1, 0]]), max_iterations=200)
        assert result.eigenvectors == []
