import pytest

from panchi import Matrix, qr_decomposition
from panchi.algorithms import QRDecomposition


class TestQRDecompositionProperties:
    """Test the QRDecomposition result object's structure and attributes."""

    def test_returns_qr_decomposition_instance(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        assert isinstance(qr_decomposition(m), QRDecomposition)

    def test_original_is_preserved(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        assert qr_decomposition(m).original == m

    def test_step_count_matches_columns(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        assert len(qr_decomposition(m).steps) == m.cols

    def test_q_and_r_have_correct_shape(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        decomp = qr_decomposition(m)
        assert decomp.q.shape == (3, 3)
        assert decomp.r.shape == (3, 3)


class TestQRDecompositionCorrectness:
    """Test the mathematical correctness of the QR factorisation."""

    def test_reconstruction(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        decomp = qr_decomposition(m)
        product = decomp.q @ decomp.r
        for i in range(m.rows):
            for j in range(m.cols):
                assert product[i][j] == pytest.approx(m[i][j], abs=1e-9)

    def test_q_has_orthonormal_columns(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        q = qr_decomposition(m).q
        gram = q.T @ q
        for i in range(q.cols):
            for j in range(q.cols):
                assert gram[i][j] == pytest.approx(1 if i == j else 0, abs=1e-9)

    def test_r_is_upper_triangular(self):
        m = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        r = qr_decomposition(m).r
        for i in range(1, r.rows):
            for j in range(i):
                assert r[i][j] == pytest.approx(0, abs=1e-9)

    def test_identity_decomposes_to_identity(self):
        decomp = qr_decomposition(Matrix([[1, 0], [0, 1]]))
        assert decomp.q @ decomp.r == Matrix([[1, 0], [0, 1]])


class TestQRDecompositionDisplay:
    """Test the string representations of QRDecomposition."""

    def test_repr_is_compact(self):
        decomp = qr_decomposition(Matrix([[1, 0], [0, 1]]))
        assert repr(decomp) == "QRDecomposition(shape=2×2, steps=2)"

    def test_str_mentions_relationship_and_steps(self):
        text = str(qr_decomposition(Matrix([[1, 1], [0, 1]])))
        assert "A = Q @ R" in text
        assert "Step 1" in text
        assert "normalize" in text
