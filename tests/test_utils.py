"""Tests for panchi.utils package structure and exports (issue #114)."""

from __future__ import annotations

import panchi.utils
from panchi.utils.latex import (
    matrix_to_latex,
    row_op_to_latex,
    scalar_to_latex,
    vector_to_latex,
)
from panchi.utils.types import SCALAR_TYPES, Scalar, parse_scalar


def test_utils_package_exports() -> None:
    """Verify that panchi.utils re-exports core utility functions and types."""
    assert panchi.utils.Scalar is Scalar
    assert panchi.utils.SCALAR_TYPES is SCALAR_TYPES
    assert panchi.utils.parse_scalar is parse_scalar
    assert panchi.utils.scalar_to_latex is scalar_to_latex
    assert panchi.utils.matrix_to_latex is matrix_to_latex
    assert panchi.utils.vector_to_latex is vector_to_latex
    assert panchi.utils.row_op_to_latex is row_op_to_latex


def test_parse_scalar_from_utils() -> None:
    """Verify parse_scalar functionality via panchi.utils.types."""
    assert parse_scalar(5) == 5
    assert parse_scalar("1/2") == 0.5 or parse_scalar("1/2").numerator == 1


def test_latex_helpers_from_utils() -> None:
    """Verify latex helpers via panchi.utils.latex."""
    assert scalar_to_latex(10) == "10"
