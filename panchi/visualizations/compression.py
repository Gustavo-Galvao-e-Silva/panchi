from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from panchi.algorithms.results import SVDResult
from panchi.primitives.matrix import Matrix

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _to_rows(matrix: Matrix) -> list[list[float]]:
    """Convert a Matrix into nested lists of floats for ``imshow``."""
    return [[float(entry) for entry in row] for row in matrix.to_list()]


def plot_compression(
    result: SVDResult,
    ranks: list[int],
    *,
    cmap: str = "gray",
    save_path: str | Path | None = None,
) -> Figure:
    """
    Show how well low-rank SVD approximations reproduce a matrix image.

    The top row draws the original matrix as an image, followed by the
    rank-k approximation ``A_k = σ₁u₁v₁ᵀ + ... + σₖuₖvₖᵀ`` for each requested
    k. Every panel shares the same colour scale, so brightness is
    comparable. The bottom panel plots the relative error and the captured
    energy for every k from 0 to ``len(result.sigma)``, with the chosen
    ranks marked.

    Parameters
    ----------
    result : SVDResult
        The singular value decomposition of the image matrix, as returned by
        ``svd``.
    ranks : list[int]
        The ranks k to display. Each must satisfy
        ``0 <= k <= len(result.sigma)``.
    cmap : str, default="gray"
        A matplotlib colormap name used for every image panel.
    save_path : str or Path, optional
        File to write the figure to. If omitted, the figure is shown with
        ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the image panels and the error/energy curves.

    Raises
    ------
    TypeError
        If ``result`` is not an ``SVDResult``, or a rank is not an int.
    ValueError
        If ``ranks`` is empty or any rank is out of range.

    Examples
    --------
    >>> from panchi import Matrix, svd
    >>> from panchi.visualizations import plot_compression
    >>> image = Matrix([[1 if 2 <= c <= 5 else 0 for c in range(8)]
    ...                 for r in range(8)])
    >>> plot_compression(svd(image), [1, 2])  # doctest: +SKIP

    See Also
    --------
    panchi.algorithms.svd : Compute the singular value decomposition.
    """
    if not isinstance(result, SVDResult):
        raise TypeError(
            f"plot_compression expects an SVDResult from svd(), "
            f"got {type(result).__name__}."
        )
    if len(ranks) == 0:
        raise ValueError(
            "ranks must list at least one rank k to display, e.g. [1, 3, 8]."
        )

    errors = [result.relative_error(k) for k in ranks]
    energies = [result.energy(k) for k in ranks]

    import matplotlib.pyplot as plt

    original = _to_rows(result.original)
    entries = [entry for row in original for entry in row]
    vmin = min(entries)
    vmax = max(entries)

    panels = len(ranks) + 1
    fig = plt.figure(figsize=(max(2.6 * panels, 8), 6.5))
    grid = fig.add_gridspec(2, panels, height_ratios=[1.3, 1])

    image_ax = fig.add_subplot(grid[0, 0])
    image_ax.imshow(original, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    image_ax.set_title(f"Original\nrank {result.rank}\n", fontsize=9)
    image_ax.axis("off")

    for column, k in enumerate(ranks, start=1):
        image_ax = fig.add_subplot(grid[0, column])
        image_ax.imshow(
            _to_rows(result.approximation(k)),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        image_ax.set_title(
            f"k = {k}\nerror {errors[column - 1]:.1%}\n"
            f"energy {energies[column - 1]:.1%}",
            fontsize=9,
        )
        image_ax.axis("off")

    all_ranks = list(range(len(result.sigma) + 1))
    curve_ax = fig.add_subplot(grid[1, :])
    curve_ax.plot(
        all_ranks,
        [result.relative_error(k) for k in all_ranks],
        marker="o",
        markersize=3,
        label="relative error",
    )
    curve_ax.plot(
        all_ranks,
        [result.energy(k) for k in all_ranks],
        marker="o",
        markersize=3,
        label="energy captured",
    )
    for k in ranks:
        curve_ax.axvline(k, color="gray", linestyle="--", linewidth=0.8)
    curve_ax.set_xlabel("rank k")
    curve_ax.set_ylim(-0.05, 1.05)
    curve_ax.grid(alpha=0.3)
    curve_ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
    return fig
