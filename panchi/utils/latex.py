"""LaTeX rendering helpers for rich display in Jupyter and Colab.

These build the LaTeX fragments used by the ``_repr_latex_`` hooks on the
primitives and result objects. They are pure string builders: the library's
terminal ``__str__``/``__repr__`` output is never affected.

This module is a leaf: it imports nothing from panchi at runtime, so any module
can depend on it without creating an import cycle. It only knows how to render
the built-in number types. Any other object renders itself by defining a
``_latex()`` method (see ``SupportsLatex``), which ``scalar_to_latex`` calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from panchi.primitives.matrix import Matrix
    from panchi.primitives.vector import Vector
    from panchi.utils.types import Scalar


@runtime_checkable
class SupportsLatex(Protocol):
    """An object that renders itself as a LaTeX fragment (no math delimiters)."""

    def _latex(self) -> str: ...


def scalar_to_latex(value: Scalar) -> str:
    """Render a scalar as LaTeX.

    ``Fraction`` values become ``\\frac{p}{q}`` (a bare integer when the
    denominator is 1, with the sign kept outside the fraction); values that
    define ``_latex()``, such as ``QuadraticNumber``, render themselves; ``int``
    and ``float`` fall back to ``str``.
    """
    if isinstance(value, SupportsLatex):
        return value._latex()
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


def span_to_latex(vectors: Iterable[Vector]) -> str:
    """Render ``span{v1, v2, ...}`` (no surrounding math delimiters)."""
    generators = r",\; ".join(vector_to_latex(v) for v in vectors)
    return f"\\operatorname{{span}}\\left\\{{ {generators} \\right\\}}"


def aligned_to_latex(lines: Iterable[str]) -> str:
    """Stack LaTeX lines in a display-math ``aligned`` block."""
    body = " \\\\\n".join(lines)
    return f"$$\\begin{{aligned}}\n{body}\n\\end{{aligned}}$$"
