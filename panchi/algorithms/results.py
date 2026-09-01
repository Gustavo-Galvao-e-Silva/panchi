from __future__ import annotations

from typing import TYPE_CHECKING

from panchi._latex import (
    matrix_to_latex,
    row_op_to_latex,
    scalar_to_latex,
    vector_to_latex,
)
from panchi.algorithms.row_operations import RowOperation
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector

if TYPE_CHECKING:
    from panchi.primitives.vector_space import VectorSpace


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

    def _repr_latex_(self) -> str:
        """Render the reduction as a stacked LaTeX arrow sequence."""
        current = self.original.copy()
        lines = [f"& {matrix_to_latex(current)}"]
        for step in self.steps:
            current = step.apply(current)
            lines.append(
                f"&\\xrightarrow{{{row_op_to_latex(step)}}} {matrix_to_latex(current)}"
            )
        body = " \\\\\n".join(lines)
        return f"$$\\begin{{aligned}}\n{body}\n\\end{{aligned}}$$"


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

        P @ A = L @ U

        P:
        [[1, 0],
         [0, 1]]

        A:
        [[2, 1],
         [4, 3]]

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
            f"\nP @ A = L @ U\n"
            f"\nP:\n{self.permutation}\n"
            f"\nA:\n{self.original}\n"
            f"\nL:\n{self.lower}\n"
            f"\nU:\n{self.upper}"
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

    def _repr_latex_(self) -> str:
        """Render the factorization ``A = LU`` (or ``PA = LU`` when pivoted)."""
        n = self.permutation.rows
        is_identity = all(
            self.permutation[i, j] == (1 if i == j else 0)
            for i in range(n)
            for j in range(n)
        )
        rhs = f"{matrix_to_latex(self.lower)}\\,{matrix_to_latex(self.upper)}"
        if is_identity:
            return f"$${matrix_to_latex(self.original)} = {rhs}$$"
        return (
            f"$$P\\,{matrix_to_latex(self.original)} = {rhs}, \\quad "
            f"P = {matrix_to_latex(self.permutation)}$$"
        )


class QRDecomposition:
    """
    The result of a (thin) QR decomposition via Gram-Schmidt.

    Stores the original matrix, the matrix Q with orthonormal columns, the
    upper triangular matrix R, and the ordered list of Gram-Schmidt steps
    used to build Q. The decomposition satisfies original == Q @ R.

    Q is formed by orthonormalizing the columns of the original matrix, so
    its columns are an orthonormal basis for the column space. R = Qᵀ @ A is
    upper triangular and its diagonal entries record how much of each column
    survived orthogonalization. The recorded steps expose the column-by-column
    derivation of Q, mirroring how a Reduction exposes its row operations.

    Parameters
    ----------
    original : Matrix
        The matrix that was decomposed, with linearly independent columns.
    q : Matrix
        The matrix with orthonormal columns.
    r : Matrix
        The upper triangular matrix satisfying original == q @ r.
    steps : list[GramSchmidtStep]
        The ordered Gram-Schmidt steps, one per column, used to build Q.

    Examples
    --------
    >>> A = Matrix([[1, 0], [0, 1]])
    >>> decomp = qr_decomposition(A)
    >>> decomp.q @ decomp.r == A
    True
    """

    def __init__(
        self,
        original: Matrix,
        q: Matrix,
        r: Matrix,
        steps: list,
    ) -> None:
        self.original = original
        self.q = q
        self.r = r
        self.steps = steps

    def __str__(self) -> str:
        """
        Return a readable summary of the QR decomposition.

        Shows the column-by-column Gram-Schmidt walkthrough, then Q and R
        individually, and states the factorisation relationship A = Q @ R.

        Returns
        -------
        str
            Human-readable decomposition summary.
        """
        header: str = (
            f"QR decomposition of "
            f"{self.original.rows}×{self.original.cols} matrix"
            f" — {len(self.steps)} steps\n"
        )

        steps_str: str = ""
        for step in self.steps:
            steps_str += f"\n{step}\n"

        body: str = (
            f"\nA = Q @ R\n"
            f"\nA:\n{self.original}\n"
            f"\nQ:\n{self.q}\n"
            f"\nR:\n{self.r}"
        )

        return header + steps_str + body

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this QRDecomposition.

        Returns
        -------
        str
            Compact representation showing shape and number of steps.

        Examples
        --------
        >>> qr_decomposition(Matrix([[1, 0], [0, 1]]))
        QRDecomposition(shape=2×2, steps=2)
        """
        return (
            f"QRDecomposition("
            f"shape={self.original.rows}×{self.original.cols}, "
            f"steps={len(self.steps)})"
        )

    def _repr_latex_(self) -> str:
        """Render the factorization ``A = QR`` in LaTeX."""
        return (
            f"$${matrix_to_latex(self.original)} = "
            f"{matrix_to_latex(self.q)}\\,{matrix_to_latex(self.r)}$$"
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

    def _repr_latex_(self) -> str:
        """Render ``A^{-1} = ...`` in LaTeX."""
        return (
            f"$${matrix_to_latex(self.original)}^{{-1}} = "
            f"{matrix_to_latex(self.inverse)}$$"
        )


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
    particular : Vector or None
        A particular solution for infinite-solution systems. None for
        unique or inconsistent systems.
    null_space : VectorSpace or None
        A basis for the null space of A for infinite-solution systems.
        None for unique or inconsistent systems.

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
        particular: Vector | None = None,
        null_space: VectorSpace | None = None,
    ) -> None:
        self.original = original
        self.target = target
        self.status = status
        self.solution = solution
        self.steps = steps
        self.particular = particular
        self.null_space = null_space

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

        if self.particular is not None and self.null_space is not None:
            return header + "\n" + self._format_general_solution()

        return header

    def _format_general_solution(self) -> str:
        basis = list(self.null_space)
        n = len(basis)

        if n == 1:
            params = ["t"]
        elif n == 2:
            params = ["s", "t"]
        else:
            params = [f"t{i + 1}" for i in range(n)]

        is_zero = all(self.particular[i] == 0 for i in range(self.particular.dims))

        parts = []
        if not is_zero:
            parts.append(str(self.particular))
        for param, vec in zip(params, basis, strict=False):
            parts.append(f"{param}·{vec}")

        return "x = " + " + ".join(parts)

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

    def _repr_latex_(self) -> str:
        """Render the solution (unique vector or general solution) in LaTeX."""
        if self.solution is not None:
            return f"$x = {vector_to_latex(self.solution)}$"
        if self.particular is not None and self.null_space is not None:
            return f"$${self._general_solution_latex()}$$"
        return "$\\text{No solution (inconsistent system).}$"

    def _general_solution_latex(self) -> str:
        basis = list(self.null_space)
        n = len(basis)
        if n == 1:
            params = ["t"]
        elif n == 2:
            params = ["s", "t"]
        else:
            params = [f"t_{{{i + 1}}}" for i in range(n)]

        is_zero = all(self.particular[i] == 0 for i in range(self.particular.dims))

        parts = []
        if not is_zero:
            parts.append(vector_to_latex(self.particular))
        for param, vec in zip(params, basis, strict=False):
            parts.append(f"{param}\\,{vector_to_latex(vec)}")

        return "x = " + " + ".join(parts)


class EigenResult:
    """
    The result of an eigenvalue computation via the QR algorithm.

    Stores the original matrix, the computed eigenvalues, the corresponding
    eigenvectors, and metadata about the iterative process: how many
    iterations were run, whether the iteration converged, and the final
    (near) upper-triangular matrix the QR algorithm produced.

    Eigenvalues are read off the diagonal of the QR iterate, and each
    eigenvector is paired with the eigenvalue at the same index. The values
    are numerical approximations, not exact or symbolic results.

    Only real eigenvalues are supported. Matrices with complex eigenvalues
    (or eigenvalues of equal magnitude) may not converge; in that case
    ``converged`` is False and ``eigenvectors`` is empty.

    Parameters
    ----------
    original : Matrix
        The square matrix whose spectrum was computed.
    eigenvalues : list[float]
        The computed eigenvalues, in the order they appear on the diagonal
        of the final iterate.
    eigenvectors : list[Vector]
        The eigenvectors, paired by index with eigenvalues. Empty when the
        iteration did not converge.
    iterations : int
        The number of QR iterations performed.
    converged : bool
        Whether the below-diagonal mass fell below the tolerance before the
        iteration limit was reached.
    triangular : Matrix
        The final (near) upper-triangular matrix produced by the iteration.

    Examples
    --------
    >>> result = eigen(Matrix([[2, 1], [1, 2]]))
    >>> sorted(round(v, 6) for v in result.eigenvalues)
    [1.0, 3.0]
    """

    def __init__(
        self,
        original: Matrix,
        eigenvalues: list[float],
        eigenvectors: list[Vector],
        iterations: int,
        converged: bool,
        triangular: Matrix,
    ) -> None:
        self.original = original
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.iterations = iterations
        self.converged = converged
        self.triangular = triangular

    @property
    def pairs(self) -> list[tuple[float, Vector]]:
        """
        The eigenvalue/eigenvector pairs, zipped by index.

        Returns
        -------
        list[tuple[float, Vector]]
            One (eigenvalue, eigenvector) tuple per computed eigenvector.
            Empty when no eigenvectors were computed.
        """
        return list(zip(self.eigenvalues, self.eigenvectors, strict=False))

    def __str__(self) -> str:
        """
        Return a readable summary of the eigenvalue computation.

        Shows the iteration count and convergence status, then each
        eigenvalue together with its eigenvector when available.

        Returns
        -------
        str
            Human-readable eigendecomposition summary.
        """
        header: str = (
            f"Eigendecomposition of "
            f"{self.original.rows}×{self.original.cols} matrix"
            f" — {self.iterations} iterations (converged: {self.converged})\n"
        )

        if self.eigenvectors:
            body = "\n" + "\n".join(
                f"λ = {value}\n  v = {vector}" for value, vector in self.pairs
            )
        else:
            body = "\nEigenvalues: " + ", ".join(str(v) for v in self.eigenvalues)

        return header + body

    def __repr__(self) -> str:
        """
        Return a concise data inspection string for this EigenResult.

        Returns
        -------
        str
            Compact representation showing shape, iteration count, and
            convergence status.

        Examples
        --------
        >>> eigen(Matrix([[2, 1], [1, 2]]))
        EigenResult(shape=2×2, iterations=..., converged=True)
        """
        return (
            f"EigenResult("
            f"shape={self.original.rows}×{self.original.cols}, "
            f"iterations={self.iterations}, "
            f"converged={self.converged})"
        )

    def _repr_latex_(self) -> str:
        """Render the eigenvalue/eigenvector pairs in LaTeX."""
        values = ",\\; ".join(scalar_to_latex(round(v, 6)) for v in self.eigenvalues)
        if not self.eigenvectors:
            body = f"\\lambda \\in \\left\\{{ {values} \\right\\}}"
            if not self.converged:
                body = "\\text{did not converge; } " + body
            return f"${body}$"
        lines = []
        for i, (value, vector) in enumerate(self.pairs):
            lines.append(
                f"\\lambda_{{{i + 1}}} = {scalar_to_latex(round(value, 6))}, "
                f"\\quad v_{{{i + 1}}} = {vector_to_latex(vector)}"
            )
        body = " \\\\\n".join(lines)
        return f"$$\\begin{{aligned}}\n{body}\n\\end{{aligned}}$$"
