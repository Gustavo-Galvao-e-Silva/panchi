from __future__ import annotations

from mathrix.primitives.matrix import Matrix
from mathrix.algorithms.row_operations import RowOperation


class Reduction:
    """
    The result of a row reduction performed on a matrix.

    Stores the original matrix, the reduced form, every row operation
    applied as an ordered sequence of Operation objects, the pivot column
    indices, and whether the result is in REF or RREF.

    Parameters
    ----------
    original : Matrix
        The matrix before any row operations were applied.
    result : Matrix
        The matrix after all row operations have been applied.
    steps : list[RowOperation]
        The ordered sequence of elementary row operations that transforms
        original into result.
    pivots : list[tuple[int, int]]
        The (row, col) positions of each pivot, in order of discovery.
    form : str
        Either 'REF' or 'RREF', indicating which reduced form was computed.

    Examples
    --------
    >>> A = Matrix([[2, 4, 6], [1, 2, 4], [0, 0, 1]])
    >>> reduction = rref(A)
    >>> reduction.rank
    3
    >>> reduction.nullity
    0
    >>> print(reduction)
    RREF of 3×3 matrix — 5 steps, rank 3
    ...
    """

    def __init__(
        self,
        original: Matrix,
        result: Matrix,
        steps: list[RowOperation],
        pivots: list[tuple[int, int]],
        form: str,
    ) -> None:
        self.original = original
        self.result = result
        self.steps = steps
        self.pivots = pivots
        self.form = form

    @property
    def rank(self) -> int:
        """
        The rank of the matrix, equal to the number of pivot positions.

        Returns
        -------
        int
            Number of pivot positions found during reduction.
        """
        return len(self.pivots)

    @property
    def nullity(self) -> int:
        """
        The nullity of the matrix, equal to columns minus rank.

        By the rank-nullity theorem, rank + nullity equals the number
        of columns of the original matrix.

        Returns
        -------
        int
            Dimension of the null space.
        """
        return self.original.cols - self.rank

    def __str__(self) -> str:
        """
        Return a step-by-step walkthrough of the reduction.

        Shows the operation label and resulting matrix state after each
        step, followed by a summary of pivot positions, rank, and nullity.

        Returns
        -------
        str
            Human-readable reduction walkthrough.

        Examples
        --------
        >>> print(rref(Matrix([[1, 2], [3, 4]])))
        RREF of 2×2 matrix — 3 steps, rank 2

        Step 1: R1 -> R1 + (-3) * R0
        [[1, 2],
         [0, -2]]
        ...
        """
        header: str = (
            f"{self.form} of {self.original.rows}×{self.original.cols} matrix"
            f" — {len(self.steps)} steps, rank {self.rank}\n"
        )

        current: Matrix = self.original.copy()
        steps_str: str = ""
        for i, step in enumerate(self.steps):
            current = step.apply(current)
            steps_str += f"\nStep {i + 1}: {step}\n{current}\n"

        footer: str = (
            f"\nResult:\n{self.result}\n"
            f"\nPivots: {self.pivots}\n"
            f"Rank: {self.rank}  |  Nullity: {self.nullity}"
        )

        return header + steps_str + footer

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this Reduction.

        Shows form, shape, rank, nullity, pivot positions, number of steps,
        and the result matrix — enough to understand the reduction at a
        glance without the full step walkthrough.

        Returns
        -------
        str
            Compact representation for debugging and inspection.

        Examples
        --------
        >>> rref(Matrix([[1, 2], [3, 4]]))
        Reduction(form=RREF, shape=2×2, rank=2, nullity=0, pivots=[(0, 0), (1, 1)], steps=3)
        [[1, 0],
         [0, 1]]
        """
        summary: str = (
            f"Reduction("
            f"form={self.form}, "
            f"shape={self.original.rows}×{self.original.cols}, "
            f"rank={self.rank}, "
            f"nullity={self.nullity}, "
            f"pivots={self.pivots}, "
            f"steps={len(self.steps)})"
        )

        return f"{summary}\n{self.result}"
