import pytest

import panchi as pan
from panchi.algorithms import determinant_lu


class TestDeprecationWarnings:
    """Each name scheduled for removal in 4.0 warns and names its replacement."""

    @pytest.mark.parametrize(
        ("call", "replacement"),
        [
            (
                lambda: determinant_lu(pan.Matrix([[1, 2], [3, 4]])),
                "determinant_by_reduction",
            ),
            (lambda: pan.one_vector(3), "Vector([1] * dims)"),
            (lambda: pan.one_matrix(2, 3), "Matrix([[1] * cols"),
            (
                lambda: pan.Matrix([[1, 2], [3, 4]]).transform(pan.Vector([1, 1])),
                "A @ v",
            ),
            (
                lambda: pan.Matrix([[1, 2, 3], [4, 5, 6]]).left_identity,
                "identity(A.rows)",
            ),
            (
                lambda: pan.Matrix([[1, 2, 3], [4, 5, 6]]).right_identity,
                "identity(A.cols)",
            ),
        ],
    )
    def test_warns_with_replacement(self, call, replacement):
        with pytest.warns(DeprecationWarning, match="removed in panchi 4.0") as record:
            call()
        assert replacement in str(record[0].message)

    def test_warning_points_at_caller(self):
        with pytest.warns(DeprecationWarning) as record:
            pan.one_vector(2)
        assert record[0].filename == __file__

    def test_deprecated_names_still_work(self):
        with pytest.warns(DeprecationWarning):
            assert pan.one_vector(2) == pan.Vector([1, 1])
        with pytest.warns(DeprecationWarning):
            assert determinant_lu(pan.Matrix([[1, 2], [3, 4]])) == pytest.approx(-2)


class TestNoInternalWarnings:
    """The library itself never triggers its own deprecation warnings."""

    @pytest.mark.filterwarnings("error::DeprecationWarning")
    def test_common_operations_do_not_warn(self):
        A = pan.Matrix([[2, 1], [1, 3]])
        A**0
        A**3
        pan.determinant(A)
        pan.determinant_by_reduction(A)
        pan.inverse(A)
        pan.solve(A, pan.Vector([1, 2]))
        pan.eigen(A)
        pan.eigen_qr(A)
        pan.rank(A)
