import pytest

from panchi import Matrix
from panchi.algorithms import ref, rref
from panchi.visualizations.backends.reduction_plan import build_reduction_plan

MATRICES = {
    "2x2 plain": [[1, 2], [3, 4]],
    "2x3 wide": [[1, 2, 3], [3, 4, 5]],
    "needs swap": [[0, 1, 2], [2, 4, 6]],
    "rank deficient": [[1, 2], [2, 4]],
    "3x3": [[1, 2, 3], [2, 5, 7], [0, 1, 2]],
    "3x4 augmented": [[2, 1, 1, 5], [4, 3, 3, 9], [8, 7, 9, 1]],
    "fractions": [["1/2", "1/3"], ["1/4", "1/5"]],
}


def _columns(matrix):
    return [
        tuple(float(matrix[i][j]) for i in range(matrix.rows))
        for j in range(matrix.cols)
    ]


def _close(left, right):
    return len(left) == len(right) and all(
        len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


class TestBuildReductionPlan:
    @pytest.mark.parametrize("rows", MATRICES.values(), ids=MATRICES.keys())
    def test_each_stage_matches_replaying_the_steps(self, rows):
        reduction = rref(Matrix(rows))
        plan = build_reduction_plan(reduction)
        current = reduction.original.copy()
        assert len(plan.stages) == len(reduction.steps)
        for stage, step in zip(plan.stages, reduction.steps, strict=True):
            current = step.apply(current)
            assert _close(stage.columns, _columns(current))

    @pytest.mark.parametrize("rows", MATRICES.values(), ids=MATRICES.keys())
    def test_last_stage_is_the_result(self, rows):
        reduction = rref(Matrix(rows))
        plan = build_reduction_plan(reduction)
        assert _close(
            plan.columns_after(len(plan.stages) - 1), _columns(reduction.result)
        )

    def test_works_for_ref(self):
        reduction = ref(Matrix([[2, 1, 1, 5], [4, 3, 3, 9], [8, 7, 9, 1]]))
        plan = build_reduction_plan(reduction)
        assert plan.form == "REF"
        assert _close(
            plan.columns_after(len(plan.stages) - 1), _columns(reduction.result)
        )

    def test_pivot_columns_land_on_standard_basis_in_rref(self):
        reduction = rref(Matrix([[1, 2, 3], [3, 4, 5]]))
        plan = build_reduction_plan(reduction)
        final = plan.columns_after(len(plan.stages) - 1)
        for row_index, column_index in enumerate(plan.pivot_columns):
            expected = tuple(1.0 if i == row_index else 0.0 for i in range(plan.dims))
            assert _close([final[column_index]], [expected])

    def test_metadata(self):
        plan = build_reduction_plan(rref(Matrix([[1, 2], [2, 4]])))
        assert plan.dims == 2
        assert plan.rank == 1
        assert plan.form == "RREF"
        assert plan.pivot_columns == [0]

    def test_labels_use_row_operation_notation(self):
        reduction = rref(Matrix([[1, 2], [3, 4]]))
        plan = build_reduction_plan(reduction)
        assert [stage.label for stage in plan.stages] == [
            str(s) for s in reduction.steps
        ]

    def test_already_reduced_has_no_stages(self):
        plan = build_reduction_plan(rref(Matrix([[1, 0], [0, 1]])))
        assert plan.stages == []
        assert plan.max_extent() >= 1.0

    def test_rejects_non_reduction(self):
        with pytest.raises(TypeError, match="Reduction"):
            build_reduction_plan(Matrix([[1, 2], [3, 4]]))


class TestTransformAt:
    def test_progress_zero_is_the_state_before_the_step(self):
        plan = build_reduction_plan(rref(Matrix([[1, 2, 3], [3, 4, 5]])))
        for index in range(len(plan.stages)):
            before = plan.cumulative_after(index - 1)
            assert _close(plan.transform_at(index, 0.0), before)

    def test_progress_one_is_the_state_after_the_step(self):
        plan = build_reduction_plan(rref(Matrix([[1, 2, 3], [3, 4, 5]])))
        for index, stage in enumerate(plan.stages):
            assert _close(plan.transform_at(index, 1.0), stage.cumulative)

    def test_max_extent_covers_every_stage(self):
        plan = build_reduction_plan(rref(Matrix([[1, 2, 3], [3, 4, 5]])))
        for index in range(-1, len(plan.stages)):
            for column in plan.columns_after(index):
                assert plan.max_extent() >= max(abs(x) for x in column)