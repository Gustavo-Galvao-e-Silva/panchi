"""Backend-agnostic data prepared from a ``Reduction`` for animation.

A row operation acts on the columns of a matrix as a linear map: if ``E`` is
the elementary matrix of one step, the columns of the matrix after the step
are ``E`` applied to the columns before it. So a reduction of an ``m x n``
matrix is a sequence of linear maps on R^m acting on ``n`` column vectors.

This module turns a ``Reduction`` into plain floats so the drawing backends
never need to know about row operations, fractions, or panchi matrices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from panchi.algorithms.results import Reduction

FloatMatrix = list[list[float]]
FloatPoint = tuple[float, ...]


def _to_float(value: object) -> float:
    """Return ``value`` as a float, or explain why it cannot be drawn."""
    try:
        return float(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"Cannot animate a reduction containing {value!r}: "
            f"visualizations need real entries."
        ) from None


def _identity(n: int) -> FloatMatrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matmul(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
    n = len(left)
    inner = len(right)
    cols = len(right[0])
    result: FloatMatrix = []
    for i in range(n):
        row: list[float] = []
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += left[i][k] * right[k][j]
            row.append(total)
        result.append(row)
    return result


def _apply(matrix: FloatMatrix, point: FloatPoint) -> FloatPoint:
    n = len(matrix)
    return tuple(
        sum(matrix[i][k] * point[k] for k in range(len(point))) for i in range(n)
    )


@dataclass(frozen=True)
class ReductionStage:
    """One row operation, ready to draw.

    Attributes
    ----------
    label : str
        The operation in standard notation, e.g. ``R1 -> R1 + (-3) * R0``.
    step_matrix : list[list[float]]
        The elementary matrix E of this single operation.
    cumulative : list[list[float]]
        The product of every elementary matrix up to and including this one.
    columns : list[tuple[float, ...]]
        The columns of the matrix after this operation.
    """

    label: str
    step_matrix: FloatMatrix
    cumulative: FloatMatrix
    columns: list[FloatPoint]


@dataclass(frozen=True)
class ReductionFrame:
    """What to draw on one frame of a reduction animation.

    Attributes
    ----------
    transform : list[list[float]]
        The total transformation applied to the space on this frame.
    matrix : list[list[float]]
        The matrix (as rows) to show on screen.
    caption : str
        The text describing the current step.
    finished : bool
        True once the last row operation has completed.
    """

    transform: FloatMatrix
    matrix: FloatMatrix
    caption: str
    finished: bool


@dataclass(frozen=True)
class ReductionPlan:
    """Everything a backend needs to animate a reduction.

    Attributes
    ----------
    dims : int
        Number of rows of the matrix, so columns live in R^dims.
    start_columns : list[tuple[float, ...]]
        The columns of the original matrix.
    stages : list[ReductionStage]
        One stage per row operation, in order. Empty if the matrix was
        already in the requested form.
    pivot_columns : list[int]
        Column indices (0-based) that contain a pivot.
    rank : int
        The rank of the original matrix.
    form : str
        ``"REF"`` or ``"RREF"``.
    """

    dims: int
    start_columns: list[FloatPoint]
    stages: list[ReductionStage]
    pivot_columns: list[int]
    rank: int
    form: str

    def columns_after(self, stage_index: int) -> list[FloatPoint]:
        """Columns after ``stage_index`` operations; ``-1`` is the original."""
        if stage_index < 0:
            return self.start_columns
        return self.stages[stage_index].columns

    def cumulative_after(self, stage_index: int) -> FloatMatrix:
        """Product of the first ``stage_index + 1`` elementary matrices."""
        if stage_index < 0:
            return _identity(self.dims)
        return self.stages[stage_index].cumulative

    def matrix_after(self, stage_index: int) -> FloatMatrix:
        """The matrix (as rows) after ``stage_index`` operations."""
        columns = self.columns_after(stage_index)
        return [[column[i] for column in columns] for i in range(self.dims)]

    def transform_at(self, stage_index: int, progress: float) -> FloatMatrix:
        """The total transformation part-way through one row operation.

        Blends the identity into this step's elementary matrix E with
        ``(1 - progress) * I + progress * E`` and applies it on top of every
        earlier step. ``progress`` 0 gives the state before the step and 1
        the state after it.
        """
        identity = _identity(self.dims)
        step = self.stages[stage_index].step_matrix
        blended = [
            [
                (1.0 - progress) * identity[i][j] + progress * step[i][j]
                for j in range(self.dims)
            ]
            for i in range(self.dims)
        ]
        return _matmul(blended, self.cumulative_after(stage_index - 1))

    def frames_per_stage(self, frames: int) -> int:
        """Frames for one row operation plus the pause that follows it."""
        return frames + max(1, frames // 3)

    def total_frames(self, frames: int) -> int:
        """Number of frames in the whole animation."""
        return max(len(self.stages), 1) * self.frames_per_stage(frames)

    def final_caption(self) -> str:
        """The caption shown once the reduction is complete."""
        if self.form == "RREF":
            return (
                f"RREF reached (rank {self.rank}): pivot columns are now "
                f"standard basis vectors"
            )
        return f"REF reached (rank {self.rank}): every pivot has only zeros below it"

    def frame_at(
        self,
        frame: int,
        frames: int,
        ease: Callable[[float], float],
    ) -> ReductionFrame:
        """Describe frame number ``frame`` of the animation.

        Parameters
        ----------
        frame : int
            The frame index, from 0 to ``total_frames(frames) - 1``.
        frames : int
            Frames used to animate each row operation.
        ease : Callable[[float], float]
            Maps linear progress in [0, 1] to eased progress.
        """
        if not self.stages:
            return ReductionFrame(
                transform=self.cumulative_after(-1),
                matrix=self.matrix_after(-1),
                caption=f"No row operations needed: already in {self.form}",
                finished=True,
            )

        stage_count = len(self.stages)
        per_stage = self.frames_per_stage(frames)
        stage_index = min(frame // per_stage, stage_count - 1)
        local = frame - stage_index * per_stage
        progress = ease(min(local / (frames - 1), 1.0))
        step_done = local >= frames - 1
        finished = step_done and stage_index == stage_count - 1

        if finished:
            caption = self.final_caption()
        else:
            label = self.stages[stage_index].label
            caption = f"Step {stage_index + 1}/{stage_count}: {label}"

        return ReductionFrame(
            transform=self.transform_at(stage_index, progress),
            matrix=self.matrix_after(stage_index if step_done else stage_index - 1),
            caption=caption,
            finished=finished,
        )

    def max_extent(self) -> float:
        """Largest absolute coordinate any column reaches during the reduction."""
        values = [1.0]
        for stage_index in range(-1, len(self.stages)):
            for column in self.columns_after(stage_index):
                values.extend(abs(x) for x in column)
        return max(values)


def build_reduction_plan(reduction: Reduction) -> ReductionPlan:
    """Convert a ``Reduction`` into a ``ReductionPlan`` of plain floats.

    Parameters
    ----------
    reduction : Reduction
        The result of ``ref`` or ``rref``.

    Returns
    -------
    ReductionPlan
        The original columns plus one stage per row operation.

    Raises
    ------
    TypeError
        If ``reduction`` is not a ``Reduction``, or holds non-real entries.
    """
    if not isinstance(reduction, Reduction):
        raise TypeError(
            f"animate_reduction expects a Reduction (the result of ref or rref). "
            f"Got {type(reduction).__name__}."
        )

    original = reduction.original
    n = original.rows

    start_columns: list[FloatPoint] = [
        tuple(_to_float(original[i, j]) for i in range(n)) for j in range(original.cols)
    ]

    cumulative = _identity(n)
    stages: list[ReductionStage] = []
    for step in reduction.steps:
        elementary = step.elementary_matrix(n)
        step_matrix = [
            [_to_float(elementary[i, j]) for j in range(n)] for i in range(n)
        ]
        cumulative = _matmul(step_matrix, cumulative)
        columns = [_apply(cumulative, column) for column in start_columns]
        stages.append(
            ReductionStage(
                label=str(step),
                step_matrix=step_matrix,
                cumulative=cumulative,
                columns=columns,
            )
        )

    return ReductionPlan(
        dims=n,
        start_columns=start_columns,
        stages=stages,
        pivot_columns=[column for _, column in reduction.pivots],
        rank=reduction.rank,
        form=reduction.form,
    )
