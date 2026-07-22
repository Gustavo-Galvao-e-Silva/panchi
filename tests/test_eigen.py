import math

import pytest

from panchi import Matrix, eigen, determinant_lu
from panchi.algorithms import EigenResult


def _sorted_round(values, digits=6):
    return sorted(round(v, digits) for v in values)


class TestEigenValidation:
    """Test input validation for eigen()."""

    def test_non_matrix_raises_type_error(self):
        with pytest.raises(TypeError):
            eigen("not a matrix")

    def test_non_square_raises_value_error(self):
        with pytest.raises(ValueError):
            eigen(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestEigenResultObject:
    """Test the EigenResult object's structure and attributes."""

    def test_returns_eigen_result_instance(self):
        assert isinstance(eigen(Matrix([[2, 1], [1, 2]])), EigenResult)

    def test_original_is_preserved(self):
        m = Matrix([[2, 1], [1, 2]])
        assert eigen(m).original == m

    def test_converges_for_symmetric_matrix(self):
        assert eigen(Matrix([[2, 1], [1, 2]])).converged is True

    def test_eigenvalue_count_matches_dimension(self):
        assert len(eigen(Matrix([[2, 1], [1, 2]])).eigenvalues) == 2


class TestEigenvalueCorrectness:
    """Test the numerical correctness of the computed eigenvalues."""

    def test_diagonal_matrix(self):
        result = eigen(Matrix([[2, 0], [0, 3]]))
        assert _sorted_round(result.eigenvalues) == [2.0, 3.0]

    def test_symmetric_2x2(self):
        result = eigen(Matrix([[2, 1], [1, 2]]))
        assert _sorted_round(result.eigenvalues) == [1.0, 3.0]

    def test_symmetric_3x3(self):
        # Eigenvalues of [[2,0,0],[0,3,4],[0,4,9]] are 2, 1, 11.
        result = eigen(Matrix([[2, 0, 0], [0, 3, 4], [0, 4, 9]]))
        assert _sorted_round(result.eigenvalues) == [1.0, 2.0, 11.0]

    def test_sum_of_eigenvalues_equals_trace(self):
        m = Matrix([[4, 1], [2, 3]])
        result = eigen(m)
        assert sum(result.eigenvalues) == pytest.approx(m.trace, abs=1e-6)

    def test_product_of_eigenvalues_equals_determinant(self):
        m = Matrix([[4, 1], [2, 3]])
        result = eigen(m)
        assert math.prod(result.eigenvalues) == pytest.approx(
            determinant_lu(m), abs=1e-6
        )


class TestEigenConvergence:
    """Test convergence reporting for non-convergent matrices."""

    def test_rotation_matrix_does_not_converge(self):
        # A 90° rotation has complex eigenvalues; unshifted QR will not settle.
        result = eigen(Matrix([[0, -1], [1, 0]]), max_iterations=200)
        assert result.converged is False

    def test_non_converged_has_no_eigenvectors(self):
        result = eigen(Matrix([[0, -1], [1, 0]]), max_iterations=200)
        assert result.eigenvectors == []
