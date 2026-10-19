"""Deprecation warnings for names scheduled for removal in panchi 4.0."""

from __future__ import annotations

import warnings


def warn_deprecated(old: str, replacement: str) -> None:
    """
    Warn that old is deprecated and will be removed in panchi 4.0.

    Call it first thing in the deprecated function or property, so the
    warning points at the user's line rather than at panchi.

    Parameters
    ----------
    old : str
        The deprecated name, as the user would write it.
    replacement : str
        What to use instead.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in panchi 4.0. "
        f"Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
