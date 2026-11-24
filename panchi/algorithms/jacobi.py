from __future__ import annotations

import math
import sys

from panchi.algorithms.results import JacobiRotation, JacobiSweep
from panchi.primitives.matrix import Matrix


def _rotation(zeta: float) -> tuple[float, float, float]:
    """
    Compute the Jacobi rotation (cos, sin, tan) for a given zeta.

    Uses the smaller of the two roots of t² + 2ζt − 1 = 0, so the
    rotation angle satisfies |θ| <= π/4.

    Parameters
    ----------
    zeta : float
        The value (β − α) / (2γ) for the column pair being rotated.

    Returns
    -------
    tuple[float, float, float]
        The cosine c, sine s and tangent t of the rotation angle.

    Examples
    --------
    >>> c, s, t = _rotation(0.0)
    >>> round(c, 6), round(s, 6), t
    (0.707107, 0.707107, 1.0)
    """
    sign = 1.0 if zeta >= 0 else -1.0
    t = sign / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = c * t
    return c, s, t


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


def _columns_to_matrix(columns: list[list[float]]) -> Matrix:
    rows = len(columns[0])
    return Matrix([[column[i] for column in columns] for i in range(rows)])


def _one_sided_jacobi(
    columns: list[list[float]],
    tolerance: float,
    max_sweeps: int,
    record: str,
) -> tuple[list[list[float]], list[list[float]], int, bool, list[JacobiSweep]]:
    """
    Orthogonalize the columns of a matrix with one-sided Jacobi rotations.

    Repeatedly sweeps over every column pair (p, q) with p < q and rotates
    the pair so the two columns become orthogonal. The same rotations are
    accumulated into V, which starts as the identity, so that the working
    columns W satisfy W = A V throughout.

    Parameters
    ----------
    columns : list[list[float]]
        The columns of A, one list per column.
    tolerance : float
        A pair is rotated when |γ| > tolerance · √(αβ).
    max_sweeps : int
        The maximum number of sweeps to perform.
    record : str
        "none", "sweeps" or "rotations": how much history to keep.

    Returns
    -------
    tuple
        The orthogonalized columns W, the columns of V, the number of sweeps
        performed, whether the last sweep made no rotations, and the recorded
        list of JacobiSweep objects.

    Examples
    --------
    >>> w, v, sweeps, converged, steps = _one_sided_jacobi(
    ...     [[1.0, 1.0], [1.0, 1.0]], 1e-15, 60, "none"
    ... )
    >>> converged, steps
    (True, [])
    """
    n = len(columns)
    work = [list(column) for column in columns]
    v_columns = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]

    frobenius_squared = 0.0
    for column in work:
        frobenius_squared += _dot(column, column)
    floor = (sys.float_info.epsilon * frobenius_squared) ** 2

    steps: list[JacobiSweep] = []
    sweeps = 0
    converged = False

    while sweeps < max_sweeps:
        sweeps += 1
        rotation_count = 0
        max_off = 0.0
        rotations: list[JacobiRotation] = []

        for p in range(n - 1):
            for q in range(p + 1, n):
                a_p = work[p]
                a_q = work[q]
                alpha = _dot(a_p, a_p)
                beta = _dot(a_q, a_q)
                gamma = _dot(a_p, a_q)
                if alpha * beta <= floor or gamma == 0.0:
                    continue

                off = abs(gamma) / math.sqrt(alpha * beta)
                max_off = max(max_off, off)
                if off <= tolerance:
                    continue

                c, s, t = _rotation((beta - alpha) / (2.0 * gamma))
                work[p] = [c * x - s * y for x, y in zip(a_p, a_q, strict=True)]
                work[q] = [s * x + c * y for x, y in zip(a_p, a_q, strict=True)]
                v_p = v_columns[p]
                v_q = v_columns[q]
                v_columns[p] = [c * x - s * y for x, y in zip(v_p, v_q, strict=True)]
                v_columns[q] = [s * x + c * y for x, y in zip(v_p, v_q, strict=True)]
                rotation_count += 1

                if record == "rotations":
                    rotations.append(
                        JacobiRotation(sweeps, p, q, c, s, t, alpha, beta, gamma)
                    )

        if record == "sweeps":
            steps.append(JacobiSweep(sweeps, rotation_count, max_off, [], None))
        elif record == "rotations":
            snapshot = _columns_to_matrix(work)
            steps.append(
                JacobiSweep(sweeps, rotation_count, max_off, rotations, snapshot)
            )

        if rotation_count == 0:
            converged = True
            break

    return work, v_columns, sweeps, converged, steps
