from __future__ import annotations

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.matplotlib_base import (
    DEFAULT_COLORS,
    _MatplotlibBackendBase,
    _calculate_axis_range,
)

AXIS_LINE_COLOR = "#666666"


def _setup_coordinate_space_3d(
    ax,
    axis_range: tuple[float, float],
    grid: bool = True,
) -> None:
    lo, hi = axis_range
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_zlim(axis_range)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)

    ax.plot([lo, hi], [0, 0], [0, 0], color=AXIS_LINE_COLOR, linewidth=1, alpha=0.7)
    ax.plot([0, 0], [lo, hi], [0, 0], color=AXIS_LINE_COLOR, linewidth=1, alpha=0.7)
    ax.plot([0, 0], [0, 0], [lo, hi], color=AXIS_LINE_COLOR, linewidth=1, alpha=0.7)

    ax.grid(grid)


class _MatplotlibBackend3D(_MatplotlibBackendBase):
    """Matplotlib-based 3D visualization backend."""

    def plot_vectors(
        self,
        vectors: list[Vector],
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None:
        if colors is None:
            colors = DEFAULT_COLORS[: len(vectors)]
        if labels is None:
            labels = [None] * len(vectors)

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")

        axis_range = _calculate_axis_range(vectors)
        _setup_coordinate_space_3d(ax, axis_range, grid)

        for vec, color, label in zip(vectors, colors, labels, strict=False):
            ax.quiver(
                0,
                0,
                0,
                vec[0],
                vec[1],
                vec[2],
                color=color,
                arrow_length_ratio=0.15,
                linewidth=2.5,
            )

            if label:
                ax.text(
                    vec[0] * 1.05,
                    vec[1] * 1.05,
                    vec[2] * 1.05,
                    label,
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

        self._finalize_figure(fig, name)

    def animate_addition(
        self,
        v1: Vector,
        v2: Vector,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("animate_addition is not yet implemented for 3D.")

    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("animate_scaling is not yet implemented for 3D.")

    def animate_transform(
        self,
        matrix: Matrix,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("animate_transform is not yet implemented for 3D.")

    def plot_span(
        self,
        space: VectorSpace,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
        span_color: str | None = None,
    ) -> None:
        raise NotImplementedError("plot_span is not yet implemented for 3D.")
