"""LaTeX rendering helpers for rich display in Jupyter and Colab.

These build the LaTeX fragments used by the ``_repr_latex_`` hooks on the
primitives and result objects. They are pure string builders: the library's
terminal ``__str__``/``__repr__`` output is never affected, and nothing here is
imported at runtime by the core, so the LaTeX layer is a strictly additive,
surface-agnostic upgrade that only renders when a notebook asks for it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panchi.primitives.matrix import Matrix
    from panchi.primitives.vector import Vector
    from panchi.utils.types import Scalar


def scalar_to_latex(value: Scalar) -> str:
    """Render a scalar as LaTeX.

    ``Fraction`` values become ``\\frac{p}{q}`` (a bare integer when the
    denominator is 1, with the sign kept outside the fraction); ``int`` and
    ``float`` fall back to ``str``.
    """
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        sign = "-" if value < 0 else ""
        return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"
    return str(value)


def matrix_to_latex(matrix: Matrix) -> str:
    """Render a Matrix as a ``bmatrix`` (no surrounding math delimiters)."""
    rows = " \\\\ ".join(
        " & ".join(scalar_to_latex(x) for x in row) for row in matrix.data
    )
    return f"\\begin{{bmatrix}} {rows} \\end{{bmatrix}}"


def vector_to_latex(vector: Vector) -> str:
    """Render a Vector as a column ``bmatrix`` (no surrounding delimiters)."""
    entries = " \\\\ ".join(scalar_to_latex(x) for x in vector.data)
    return f"\\begin{{bmatrix}} {entries} \\end{{bmatrix}}"


def row_op_to_latex(op) -> str:
    """Render a row operation as its standard LaTeX notation (0-based rows)."""
    from panchi.algorithms.row_operations import RowAdd, RowScale, RowSwap

    if isinstance(op, RowSwap):
        return f"R_{{{op.a}}} \\leftrightarrow R_{{{op.b}}}"
    if isinstance(op, RowScale):
        return f"R_{{{op.row}}} \\to {scalar_to_latex(op.scalar)}\\,R_{{{op.row}}}"
    if isinstance(op, RowAdd):
        return (
            f"R_{{{op.target}}} \\to R_{{{op.target}}} "
            f"+ ({scalar_to_latex(op.scalar)})\\,R_{{{op.source}}}"
        )
    return str(op)
