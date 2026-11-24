import math
import random
from fractions import Fraction

import pytest

from panchi import Matrix, diagonal, identity, qr_decomposition, svd, zero_matrix
from panchi.algorithms import JacobiRotation, JacobiSweep, SVDResult


def random_matrix_seeded(rows, cols, seed):
    rng = random.Random(seed)
    return Matrix([[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)])


def frobenius(matrix):
    return math.sqrt(sum(x * x for row in matrix.to_list() for x in row))


def assert_matrix_close(actual, expected, tol):
    a = actual.to_list()
    e = expected.to_list()
    assert len(a) == len(e)
    for row_a, row_e in zip(a, e, strict=True):
        assert len(row_a) == len(row_e)
        for x, y in zip(row_a, row_e, strict=True):
            assert x == pytest.approx(y, abs=tol)


def assert_orthonormal_columns(matrix, tol=1e-12):
    assert_matrix_close(matrix.T @ matrix, identity(matrix.cols), tol)


def assert_reconstructs(result, tol_factor=1e-12):
    a = result.original
    tol = tol_factor * max(frobenius(a), 1.0)
    assert_matrix_close(result.u @ result.s @ result.vt, a, tol)


def rectangles_bitmap(size=16):
    image = [[0] * size for _ in range(size)]
    for r0, r1, c0, c1 in [(2, 6, 3, 10), (8, 13, 1, 5), (10, 14, 9, 14)]:
        for i in range(r0, r1 + 1):
            for j in range(c0, c1 + 1):
                image[i][j] = 1
    return Matrix(image)


SHAPES = [(4, 4), (6, 3), (3, 6), (1, 5), (5, 1), (8, 8), (10, 7)]


class TestSVDStructure:
    """Test the SVDResult object's type, attributes and shapes."""

    def test_returns_svd_result(self):
        assert isinstance(svd(Matrix([[1, 2], [3, 4]])), SVDResult)

    def test_original_is_preserved(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        assert svd(m).original == m

    def test_tall_shapes(self):
        r = svd(random_matrix_seeded(5, 3, 1))
        assert r.u.shape == (5, 3)
        assert r.s.shape == (3, 3)
        assert r.v.shape == (3, 3)
        assert r.vt.shape == (3, 3)
        assert len(r.sigma) == 3

    def test_wide_shapes(self):
        r = svd(random_matrix_seeded(3, 5, 2))
        assert r.u.shape == (3, 3)
        assert r.s.shape == (3, 3)
        assert r.v.shape == (5, 3)
        assert r.vt.shape == (3, 5)
        assert len(r.sigma) == 3

    def test_vt_is_transpose_of_v(self):
        r = svd(random_matrix_seeded(4, 3, 3))
        assert r.vt == r.v.T

    def test_s_is_diagonal_of_sigma(self):
        r = svd(random_matrix_seeded(4, 3, 4))
        for i in range(3):
            for j in range(3):
                expected = r.sigma[i] if i == j else 0
                assert r.s[i][j] == pytest.approx(expected, abs=0)

    def test_sigma_entries_are_floats(self):
        r = svd(Matrix([[1, 2], [3, 4]]))
        assert all(isinstance(x, float) for x in r.sigma)

    def test_tolerance_stored(self):
        assert svd(Matrix([[1, 2], [3, 4]]), tolerance=1e-10).tolerance == 1e-10


class TestSVDCorrectness:
    """Test reconstruction, orthonormality and ordering on random matrices."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_reconstruction(self, shape):
        assert_reconstructs(svd(random_matrix_seeded(*shape, seed=sum(shape))))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_u_orthonormal(self, shape):
        assert_orthonormal_columns(svd(random_matrix_seeded(*shape, seed=sum(shape))).u)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_v_orthonormal(self, shape):
        assert_orthonormal_columns(svd(random_matrix_seeded(*shape, seed=sum(shape))).v)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sigma_descending_non_negative(self, shape):
        sigma = svd(random_matrix_seeded(*shape, seed=sum(shape))).sigma
        assert all(x >= 0 for x in sigma)
        assert all(sigma[i] >= sigma[i + 1] for i in range(len(sigma) - 1))

    @pytest.mark.parametrize("shape", SHAPES)
    def test_converged(self, shape):
        assert svd(random_matrix_seeded(*shape, seed=sum(shape))).converged

    @pytest.mark.parametrize("shape", SHAPES)
    def test_sign_rule(self, shape):
        v = svd(random_matrix_seeded(*shape, seed=sum(shape))).v
        for j in range(v.cols):
            column = [v[i][j] for i in range(v.rows)]
            first = next(x for x in column if abs(x) > 1e-12)
            assert first > 0


class TestSVDKnownValues:
    """Test singular values and rank on matrices with known answers."""

    def test_diagonal(self):
        r = svd(Matrix([[3, 0], [0, 4]]))
        assert r.sigma == pytest.approx([4, 3], abs=1e-12)
        assert r.rank == 2

    def test_all_ones_rank_one(self):
        r = svd(Matrix([[1, 1], [1, 1]]))
        assert r.sigma == pytest.approx([2, 0], abs=1e-12)
        assert r.rank == 1
        assert_orthonormal_columns(r.u)
        assert_reconstructs(r)

    def test_identity(self):
        r = svd(identity(4))
        assert r.sigma == pytest.approx([1, 1, 1, 1], abs=1e-12)
        assert r.rank == 4

    def test_rank_one_outer_product(self):
        u = [1, 2, 3, 4]
        v = [2, -1, 5]
        r = svd(Matrix([[a * b for b in v] for a in u]))
        norm_product = math.sqrt(sum(a * a for a in u)) * math.sqrt(
            sum(b * b for b in v)
        )
        assert r.sigma[0] == pytest.approx(norm_product, rel=1e-12)
        assert r.sigma[1:] == pytest.approx([0, 0], abs=1e-12)
        assert r.rank == 1
        assert_orthonormal_columns(r.u)
        assert_reconstructs(r)

    def test_zero_matrix(self):
        r = svd(zero_matrix(3, 2))
        assert r.sigma == [0.0, 0.0]
        assert r.rank == 0
        assert_orthonormal_columns(r.u)
        assert_reconstructs(r)

    def test_rectangles_bitmap(self):
        r = svd(rectangles_bitmap(16))
        assert r.rank == 3
        assert r.converged
        assert r.sweeps < 20
        assert_reconstructs(r)
        assert r.error(r.rank) == pytest.approx(0, abs=1e-10)

    def test_repeated_singular_values(self):
        q = qr_decomposition(random_matrix_seeded(3, 3, 11)).q
        r = svd(q @ diagonal([2, 2, 1]))
        assert r.sigma == pytest.approx([2, 2, 1], abs=1e-12)
        assert_orthonormal_columns(r.u)
        assert_orthonormal_columns(r.v)
        assert_reconstructs(r)

    def test_scaled_identity_repeated(self):
        r = svd(Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
        assert r.sigma == pytest.approx([2, 2, 2], abs=1e-12)
        assert_orthonormal_columns(r.u)
        assert_reconstructs(r)


class TestSVDWideVersusTall:
    """Test that a wide matrix and its transpose share the same factorization."""

    @pytest.mark.parametrize("shape", [(3, 6), (2, 5), (1, 4)])
    def test_sigma_matches_transpose(self, shape):
        a = random_matrix_seeded(*shape, seed=sum(shape))
        assert svd(a).sigma == pytest.approx(svd(a.T).sigma, abs=1e-12)

    @pytest.mark.parametrize("shape", [(3, 6), (2, 5)])
    def test_components_match_transpose(self, shape):
        a = random_matrix_seeded(*shape, seed=sum(shape))
        wide = svd(a)
        tall = svd(a.T)
        for i in range(len(wide.sigma)):
            assert_matrix_close(wide.component(i), tall.component(i).T, 1e-10)


class TestSVDCompression:
    """Test the rank-k approximation and compression helpers."""

    def setup_method(self):
        self.a = random_matrix_seeded(6, 4, 21)
        self.r = svd(self.a)

    def test_error_matches_frobenius_of_residual(self):
        for k in range(len(self.r.sigma) + 1):
            residual = self.a - self.r.approximation(k)
            assert self.r.error(k) == pytest.approx(frobenius(residual), abs=1e-10)

    def test_relative_error(self):
        for k in range(len(self.r.sigma) + 1):
            expected = self.r.error(k) / frobenius(self.a)
            assert self.r.relative_error(k) == pytest.approx(expected, abs=1e-12)

    def test_relative_error_zero_matrix(self):
        assert svd(zero_matrix(2, 2)).relative_error(0) == 0.0

    def test_energy_non_decreasing_and_ends_at_one(self):
        values = [self.r.energy(k) for k in range(len(self.r.sigma) + 1)]
        assert values[0] == pytest.approx(0, abs=1e-15)
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        assert values[-1] == pytest.approx(1.0, abs=1e-12)

    def test_energy_zero_matrix_is_one(self):
        assert svd(zero_matrix(2, 3)).energy(0) == 1.0

    def test_error_at_rank_is_zero(self):
        r = svd(rectangles_bitmap(16))
        assert r.error(r.rank) == pytest.approx(0, abs=1e-10)

    def test_approximation_zero_is_zero_matrix(self):
        assert self.r.approximation(0) == zero_matrix(6, 4)

    def test_full_approximation_reconstructs(self):
        assert_matrix_close(self.r.approximation(4), self.a, 1e-12)

    def test_components_sum_to_approximation(self):
        total = zero_matrix(6, 4)
        for i in range(3):
            total = total + self.r.component(i)
        assert_matrix_close(total, self.r.approximation(3), 1e-12)

    def test_storage(self):
        for k in range(5):
            assert self.r.storage(k) == k * (6 + 4 + 1)

    def test_compression_ratio(self):
        for k in range(5):
            assert self.r.compression_ratio(k) == pytest.approx(k * 11 / 24)

    def test_rank_for_energy(self):
        r = svd(Matrix([[3, 0], [0, 4]]))
        assert r.rank_for_energy(0) == 0
        assert r.rank_for_energy(0.6) == 1
        assert r.rank_for_energy(0.7) == 2
        assert r.rank_for_energy(1) == 2

    @pytest.mark.parametrize("k", [-1, 5])
    def test_k_out_of_range(self, k):
        for method in [
            self.r.approximation,
            self.r.error,
            self.r.relative_error,
            self.r.energy,
            self.r.storage,
            self.r.compression_ratio,
        ]:
            with pytest.raises(ValueError):
                method(k)

    @pytest.mark.parametrize("i", [-1, 4])
    def test_component_out_of_range(self, i):
        with pytest.raises(ValueError):
            self.r.component(i)

    @pytest.mark.parametrize("k", [1.5, "2", None])
    def test_k_non_int(self, k):
        for method in [self.r.approximation, self.r.error, self.r.energy]:
            with pytest.raises(TypeError):
                method(k)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_rank_for_energy_out_of_range(self, fraction):
        with pytest.raises(ValueError):
            self.r.rank_for_energy(fraction)


class TestSVDPairs:
    """Test the (sigma, u_i, v_i) pairs view."""

    def test_pairs_structure(self):
        r = svd(random_matrix_seeded(5, 3, 31))
        pairs = r.pairs
        assert len(pairs) == 3
        for i, (sigma, u_i, v_i) in enumerate(pairs):
            assert sigma == r.sigma[i]
            assert u_i.to_list() == pytest.approx([r.u[row][i] for row in range(5)])
            assert v_i.to_list() == pytest.approx([r.v[row][i] for row in range(3)])


class TestSVDRecording:
    """Test the record levels and sweep bookkeeping."""

    def setup_method(self):
        self.a = random_matrix_seeded(5, 4, 41)

    def test_record_none(self):
        assert svd(self.a, record="none").steps == []

    def test_record_sweeps(self):
        r = svd(self.a, record="sweeps")
        assert len(r.steps) == r.sweeps
        for index, sweep in enumerate(r.steps, start=1):
            assert isinstance(sweep, JacobiSweep)
            assert sweep.index == index
            assert sweep.rotations == []
            assert sweep.matrix is None

    def test_default_is_sweeps(self):
        r = svd(self.a)
        assert len(r.steps) == r.sweeps
        assert all(sweep.rotations == [] for sweep in r.steps)

    def test_record_rotations(self):
        r = svd(self.a, record="rotations")
        assert len(r.steps) == r.sweeps
        assert sum(len(sweep.rotations) for sweep in r.steps) > 0
        for sweep in r.steps:
            assert isinstance(sweep.matrix, Matrix)
            assert sweep.rotation_count == len(sweep.rotations)
            for rotation in sweep.rotations:
                assert isinstance(rotation, JacobiRotation)
                assert rotation.p < rotation.q
                assert rotation.sweep == sweep.index

    def test_final_sweep_has_no_rotations(self):
        r = svd(self.a, record="rotations")
        assert r.converged
        assert r.steps[-1].rotation_count == 0

    def test_max_sweeps_one_does_not_converge(self):
        r = svd(self.a, max_sweeps=1)
        assert r.converged is False
        assert r.sweeps == 1


class TestSVDValidation:
    """Test input validation and error messages."""

    def test_non_matrix(self):
        with pytest.raises(TypeError):
            svd([[1, 2], [3, 4]])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_entry(self, bad):
        with pytest.raises(ValueError):
            svd(Matrix([[1, 2], [3, bad]]))

    def test_bad_record(self):
        with pytest.raises(ValueError):
            svd(Matrix([[1, 2], [3, 4]]), record="all")

    @pytest.mark.parametrize("max_sweeps", [0, -3])
    def test_bad_max_sweeps(self, max_sweeps):
        with pytest.raises(ValueError):
            svd(Matrix([[1, 2], [3, 4]]), max_sweeps=max_sweeps)

    @pytest.mark.parametrize("tolerance", [0, -1e-12])
    def test_bad_tolerance(self, tolerance):
        with pytest.raises(ValueError):
            svd(Matrix([[1, 2], [3, 4]]), tolerance=tolerance)


class TestSVDInputTypes:
    """Test that exact and integer inputs are accepted and produce floats."""

    def test_int_input(self):
        r = svd(Matrix([[3, 0], [0, 4]]))
        assert all(isinstance(x, float) for x in r.sigma)

    def test_fraction_input(self):
        r = svd(Matrix([[Fraction(1, 2), 0], [0, Fraction(3, 2)]]))
        assert r.sigma == pytest.approx([1.5, 0.5], abs=1e-12)
        assert all(isinstance(x, float) for x in r.sigma)
        assert_reconstructs(r)


class TestSVDDisplay:
    """Test the terminal representations of the result classes."""

    def test_str_header_and_equation(self):
        r = svd(Matrix([[1, 2], [3, 4], [5, 6]]))
        text = str(r)
        noun = "sweep" if r.sweeps == 1 else "sweeps"
        assert text.startswith(f"SVD of 3×2 matrix — {r.sweeps} {noun}, converged")
        assert "A = U Σ Vᵀ" in text

    def test_str_not_converged_header(self):
        r = svd(random_matrix_seeded(5, 4, 41), max_sweeps=1)
        assert str(r).startswith("SVD of 5×4 matrix — 1 sweep, not converged")

    def test_repr(self):
        r = svd(Matrix([[1, 1], [1, 1]]))
        assert repr(r) == (
            f"SVDResult(shape=2×2, rank=1, sweeps={r.sweeps}, converged=True)"
        )

    def test_sweep_str(self):
        sweep = JacobiSweep(2, 5, 0.00123, [], None)
        assert str(sweep) == (
            "Sweep 2: 5 rotations, max |cos(angle between columns)| = 1.2e-03"
        )


class TestSVDAgainstNumPy:
    """Cross-check singular values against NumPy when it is installed."""

    @pytest.mark.parametrize("shape", [(4, 4), (7, 3), (3, 7), (16, 16), (32, 32)])
    def test_sigma_matches_numpy(self, shape):
        np = pytest.importorskip("numpy")
        a = random_matrix_seeded(*shape, seed=100 + sum(shape))
        expected = np.linalg.svd(np.array(a.to_list()), compute_uv=False)
        assert svd(a).sigma == pytest.approx(list(expected), abs=1e-10)
