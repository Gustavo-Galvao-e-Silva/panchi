from __future__ import annotations

from mathrix.primitives.matrix import Matrix
from mathrix.primitives.vector import Vector
from mathrix.algorithms.row_operations import RowOperation


class Reduction:
    """
    The result of a row reduction performed on a matrix.

    Stores the original matrix, the reduced form, every row operation
    applied as an ordered sequence of RowOperation objects, the pivot
    positions, and whether the result is in REF or RREF.

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
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> reduction = ref(A)
    >>> reduction.rank
    2
    >>> reduction.nullity
    0
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
        >>> print(ref(Matrix([[1, 2], [3, 4]])))
        REF of 2×2 matrix — 1 steps, rank 2

        Step 1: R1 -> R1 + (-3.0) * R0
        [[1, 2],
         [0, -2.0]]
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

        Returns
        -------
        str
            Compact representation showing form, shape, rank, nullity,
            pivot positions, number of steps, and the result matrix.

        Examples
        --------
        >>> ref(Matrix([[1, 2], [3, 4]]))
        Reduction(form=REF, shape=2×2, rank=2, nullity=0, pivots=[(0, 0), (1, 1)], steps=1)
        [[1, 2],
         [0, -2.0]]
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


class LUDecomposition:
    """
    The result of an LU decomposition with partial pivoting.

    Stores the original matrix, the lower triangular matrix L, the upper
    triangular matrix U, and the permutation matrix P encoding any row
    swaps applied for numerical stability. The decomposition satisfies
    P @ original == L @ U.

    Partial pivoting swaps rows before each elimination step so that the
    largest available entry in the pivot column is used as the pivot.
    This avoids division by small numbers and produces a more numerically
    stable result. The swaps are recorded in P so that the factorisation
    relationship is exact.

    Parameters
    ----------
    original : Matrix
        The square matrix that was decomposed.
    lower : Matrix
        The lower triangular matrix L with ones on the diagonal.
    upper : Matrix
        The upper triangular matrix U produced by Gaussian elimination
        on P @ original.
    permutation : Matrix
        The permutation matrix P encoding all row swaps performed,
        satisfying P @ original == L @ U.
    steps : list[RowOperation]
        The ordered sequence of row operations applied to P @ original
        to produce U.

    Examples
    --------
    >>> A = Matrix([[2, 1], [4, 3]])
    >>> decomp = lu(A)
    >>> decomp.lower @ decomp.upper == decomp.permutation @ A
    True
    """

    def __init__(
        self,
        original: Matrix,
        lower: Matrix,
        upper: Matrix,
        permutation: Matrix,
        steps: list[RowOperation],
    ) -> None:
        self.original = original
        self.lower = lower
        self.upper = upper
        self.permutation = permutation
        self.steps = steps

    def __str__(self) -> str:
        """
        Return a readable summary of the LU decomposition.

        Shows P, L, and U individually and states the factorisation
        relationship P @ A = L @ U.

        Returns
        -------
        str
            Human-readable decomposition summary.

        Examples
        --------
        >>> print(lu(Matrix([[2, 1], [4, 3]])))
        LU decomposition of 2×2 matrix — 1 steps

        P:
        [[1, 0],
         [0, 1]]

        L:
        [[1, 0],
         [2.0, 1]]

        U:
        [[2, 1],
         [0.0, 1.0]]
        """
        header: str = (
            f"LU decomposition of "
            f"{self.original.rows}×{self.original.cols} matrix"
            f" — {len(self.steps)} steps\n"
        )

        body: str = (
            f"\nP:\n{self.permutation}\n" f"\nL:\n{self.lower}\n" f"\nU:\n{self.upper}"
        )

        return header + body

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this LUDecomposition.

        Returns
        -------
        str
            Compact representation showing shape and number of steps.

        Examples
        --------
        >>> lu(Matrix([[2, 1], [4, 3]]))
        LUDecomposition(shape=2×2, steps=1)
        """
        return (
            f"LUDecomposition("
            f"shape={self.original.rows}×{self.original.cols}, "
            f"steps={len(self.steps)})"
        )


class InverseResult:
    """
    The result of a matrix inversion via Gauss-Jordan elimination.

    Stores the original matrix, its inverse, and the row operations applied
    during reduction of the augmented matrix [A | I]. The inverse satisfies
    original @ inverse == identity(n) == inverse @ original.

    Parameters
    ----------
    original : Matrix
        The square invertible matrix that was inverted.
    inverse : Matrix
        The inverse of the original matrix.
    steps : list[RowOperation]
        The ordered sequence of row operations applied to the augmented
        matrix [A | I] to produce [I | A⁻¹].

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> result = inverse(A)
    >>> result.original @ result.inverse == identity(2)
    True
    """

    def __init__(
        self,
        original: Matrix,
        inverse: Matrix,
        steps: list[RowOperation],
    ) -> None:
        self.original = original
        self.inverse = inverse
        self.steps = steps

    def __str__(self) -> str:
        """
        Return a readable summary of the inversion.

        Shows the number of steps taken and the computed inverse matrix.

        Returns
        -------
        str
            Human-readable inversion summary.

        Examples
        --------
        >>> print(inverse(Matrix([[1, 2], [3, 4]])))
        Inverse of 2×2 matrix — 6 steps

        Inverse:
        [[-2.0, 1.0],
         [1.5, -0.5]]
        """
        header: str = (
            f"Inverse of {self.original.rows}×{self.original.cols} matrix"
            f" — {len(self.steps)} steps\n"
        )

        return header + f"\nInverse:\n{self.inverse}"

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this InverseResult.

        Returns
        -------
        str
            Compact representation showing shape, number of steps,
            and the inverse matrix.

        Examples
        --------
        >>> inverse(Matrix([[1, 2], [3, 4]]))
        InverseResult(shape=2×2, steps=6)
        [[-2.0, 1.0],
         [1.5, -0.5]]
        """
        summary: str = (
            f"InverseResult("
            f"shape={self.original.rows}×{self.original.cols}, "
            f"steps={len(self.steps)})"
        )

        return f"{summary}\n{self.inverse}"


class Solution:
    """
    The result of solving a linear system Ax = b.

    Stores the coefficient matrix A, the right-hand side vector b, the
    solution status, the solution vector x if a unique solution exists,
    and the row operations applied during reduction of the augmented
    matrix [A | b].

    The three possible statuses reflect the three fundamentally different
    outcomes a linear system can have:

    - 'unique': exactly one solution exists, stored in solution.
    - 'infinite': infinitely many solutions exist (underdetermined system).
    - 'inconsistent': no solution exists (the system is contradictory).

    Parameters
    ----------
    original : Matrix
        The coefficient matrix A.
    target : Vector
        The right-hand side vector b.
    status : str
        One of 'unique', 'infinite', or 'inconsistent'.
    solution : Vector or None
        The unique solution vector x satisfying A @ x == b, or None if
        the system does not have a unique solution.
    steps : list[RowOperation]
        The ordered sequence of row operations applied to the augmented
        matrix [A | b] during reduction.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> b = Vector([5, 6])
    >>> result = solve(A, b)
    >>> result.status
    'unique'
    >>> A @ result.solution == b
    True
    """

    def __init__(
        self,
        original: Matrix,
        target: Vector,
        status: str,
        solution: Vector | None,
        steps: list[RowOperation],
    ) -> None:
        self.original = original
        self.target = target
        self.status = status
        self.solution = solution
        self.steps = steps

    def __str__(self) -> str:
        """
        Return a readable summary of the solution.

        Shows the system dimensions, the status, and the solution vector
        if one exists.

        Returns
        -------
        str
            Human-readable solution summary.

        Examples
        --------
        >>> print(solve(Matrix([[1, 2], [3, 4]]), Vector([5, 6])))
        Solution to 2×2 system — unique

        x = [-4.0, 4.5]
        """
        header: str = (
            f"Solution to "
            f"{self.original.rows}×{self.original.cols} system"
            f" — {self.status}\n"
        )

        if self.solution is not None:
            return header + f"\nx = {self.solution}"

        return header

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this Solution.

        Returns
        -------
        str
            Compact representation showing shape, status, and solution.

        Examples
        --------
        >>> solve(Matrix([[1, 2], [3, 4]]), Vector([5, 6]))
        Solution(shape=2×2, status=unique, solution=[-4.0, 4.5])
        """
        return (
            f"Solution("
            f"shape={self.original.rows}×{self.original.cols}, "
            f"status={self.status}, "
            f"solution={self.solution})"
        )
