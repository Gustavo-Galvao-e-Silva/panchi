from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.base import _BaseBackend

DEFAULT_COLORS = ["#E63946", "#F77F00", "#06FFA5", "#118AB2", "#073B4C"]

# Per-method default color roles. Any of these can be overridden by passing
# ``colors=`` (animations) or ``span_color=`` (spans) to the matching method.
ADDITION_COLORS = ("#E63946", "#F77F00", "#06FFA5")  # v1, v2, result
SCALING_COLORS = ("#E63946", "#118AB2")  # original, scaled
TRANSFORM_COLORS = ("#E63946", "#118AB2")  # e1, e2
SPAN_COLOR = "#7C3AED"

GRID_COLOR = "#CCCCCC"

AXIS_PADDING = 1.3
MIN_AXIS_RANGE = 0.5
_EPSILON = 1e-12

_QUALITY_DPI = {
    "low": 72,
    "medium": 100,
    "high": 150,
}


def _resolve_colors(colors: list[str] | None, defaults: tuple[str, ...]) -> list[str]:
    """Fill in per-role colors, keeping defaults for any not supplied.

    A partial list is honored positionally; missing trailing roles fall back
    to ``defaults``. ``None`` reproduces the default palette exactly.
    """
    colors = colors or []
    return [colors[i] if i < len(colors) else defaults[i] for i in range(len(defaults))]


def _setup_coordinate_plane(
    ax: plt.Axes,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: bool = True,
) -> None:
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    if grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax.set_xlabel("x", fontsize=12, loc="right")
    ax.set_ylabel("y", fontsize=12, loc="top", rotation=0)


def _calculate_axis_range(vectors: list[Vector]) -> tuple[float, float]:
    all_x = [v[0] for v in vectors]
    all_y = [v[1] for v in vectors]
    max_coord = max(abs(min(all_x + all_y)), abs(max(all_x + all_y))) * AXIS_PADDING
    max_coord = max(max_coord, MIN_AXIS_RANGE)
    return (-max_coord, max_coord)


def _create_vector_arrow(
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float = 2.5,
) -> FancyArrowPatch:
    return FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_width=0.4,head_length=0.8",
        color=color,
        linewidth=linewidth,
        zorder=3,
        mutation_scale=20,
    )


def _add_vector_label(
    ax: plt.Axes,
    position: tuple[float, float],
    label: str,
    color: str,
    offset: tuple[float, float] = (0.15, 0.15),
) -> plt.Text:
    return ax.text(
        position[0] + offset[0],
        position[1] + offset[1],
        label,
        fontsize=12,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
    )


def _smooth_step(t: float) -> float:
    return t * t * (3 - 2 * t)


class _MatplotlibBackend(_BaseBackend):
    """Matplotlib-based 2D visualization backend."""

    def __init__(
        self,
        save_path: Path | None,
        quality: str,
        figsize: tuple[int, int],
    ) -> None:
        super().__init__(save_path=save_path, quality=quality)
        self.figsize = figsize
        self._dpi = _QUALITY_DPI.get(quality, 100)

    def _finalize_figure(self, fig: plt.Figure, name: str) -> None:
        plt.tight_layout()
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path / f"{name}.png", dpi=self._dpi)
            plt.close(fig)
        else:
            plt.show()

    def _run_animation(
        self,
        fig: plt.Figure,
        frame_fn,
        frames: int,
        interval: int,
        name: str,
    ) -> None:
        """Build a blitting ``FuncAnimation`` from ``frame_fn`` and finalize it."""
        anim = animation.FuncAnimation(
            fig,
            frame_fn,
            frames=frames,
            interval=interval,
            blit=True,
            repeat=True,
        )
        self._finalize_animation(anim, name, interval)

    def _finalize_animation(
        self,
        anim: animation.FuncAnimation,
        name: str,
        interval: int,
    ) -> None:
        plt.tight_layout()
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            anim.save(
                str(self.save_path / f"{name}.gif"),
                writer="pillow",
                fps=1000 // interval,
            )
            plt.close(anim._fig)
        else:
            plt.show()

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

        fig, ax = plt.subplots(figsize=self.figsize)
        axis_range = _calculate_axis_range(vectors)
        _setup_coordinate_plane(ax, axis_range, axis_range, grid)

        for vec, color, label in zip(vectors, colors, labels, strict=False):
            arrow = _create_vector_arrow((0, 0), (vec[0], vec[1]), color)
            ax.add_patch(arrow)

            if label:
                offset_x = 0.15 if vec[0] >= 0 else -0.15
                offset_y = 0.15 if vec[1] >= 0 else -0.15
                _add_vector_label(
                    ax, (vec[0], vec[1]), label, color, (offset_x, offset_y)
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
        color_v1, color_v2, color_result = _resolve_colors(colors, ADDITION_COLORS)

        fig, ax = plt.subplots(figsize=self.figsize)

        result = v1 + v2
        axis_range = _calculate_axis_range([v1, v2, result])
        _setup_coordinate_plane(ax, axis_range, axis_range, grid=True)

        arrow_v1 = _create_vector_arrow((0, 0), (v1[0], v1[1]), color_v1)
        arrow_v2_from_origin = _create_vector_arrow((0, 0), (0, 0), color_v2)
        arrow_v2_from_origin.set_alpha(0.4)
        arrow_v2_from_v1 = _create_vector_arrow(
            (v1[0], v1[1]), (v1[0], v1[1]), color_v2
        )
        arrow_result = _create_vector_arrow((0, 0), (0, 0), color_result, linewidth=3.5)
        arrow_result.set_alpha(0)

        ax.add_patch(arrow_v1)
        ax.add_patch(arrow_v2_from_origin)
        ax.add_patch(arrow_v2_from_v1)
        ax.add_patch(arrow_result)

        ax.text(
            v1[0] / 2 - 0.2,
            v1[1] / 2 + 0.3,
            "v₁",
            fontsize=14,
            fontweight="bold",
            color=color_v1,
        )
        text_v2 = ax.text(
            0, 0, "v₂", fontsize=14, fontweight="bold", color=color_v2, alpha=0
        )
        text_result = ax.text(
            0,
            0,
            "v₁ + v₂",
            fontsize=14,
            fontweight="bold",
            color=color_result,
            alpha=0,
        )

        half_frames = frames // 2

        def animate_frame(frame: int) -> tuple:
            if frame < half_frames:
                t = frame / half_frames
                current_x = v2[0] * t
                current_y = v2[1] * t

                arrow_v2_from_origin.set_positions((0, 0), (current_x, current_y))
                arrow_v2_from_v1.set_positions(
                    (v1[0], v1[1]), (v1[0] + current_x, v1[1] + current_y)
                )

                text_v2.set_position((current_x / 2 + 0.2, current_y / 2 - 0.3))
                text_v2.set_alpha(t)
            else:
                t = (frame - half_frames) / half_frames

                arrow_v2_from_origin.set_positions((0, 0), (v2[0], v2[1]))
                arrow_v2_from_v1.set_positions((v1[0], v1[1]), (result[0], result[1]))

                text_v2.set_position((v2[0] / 2 + 0.2, v2[1] / 2 - 0.3))
                text_v2.set_alpha(1)

                result_x = result[0] * t
                result_y = result[1] * t
                arrow_result.set_positions((0, 0), (result_x, result_y))
                arrow_result.set_alpha(t)

                text_result.set_position((result_x / 2 - 0.3, result_y / 2 + 0.5))
                text_result.set_alpha(t)

            return (
                arrow_v2_from_origin,
                arrow_v2_from_v1,
                arrow_result,
                text_v2,
                text_result,
            )

        self._run_animation(fig, animate_frame, frames, interval, name)

    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        color_start, color_end = _resolve_colors(colors, SCALING_COLORS)

        fig, ax = plt.subplots(figsize=self.figsize)

        scaled = scale_factor * vector
        axis_range = _calculate_axis_range([vector, scaled])
        _setup_coordinate_plane(ax, axis_range, axis_range, grid=True)

        arrow = _create_vector_arrow((0, 0), (vector[0], vector[1]), color_start)
        ax.add_patch(arrow)

        text = ax.text(
            vector[0] + 0.2,
            vector[1] + 0.2,
            "v",
            fontsize=14,
            fontweight="bold",
            color=color_start,
        )

        def animate_frame(frame: int) -> tuple:
            t = frame / frames
            current_scale = 1 + (scale_factor - 1) * t
            current_x = vector[0] * current_scale
            current_y = vector[1] * current_scale

            arrow.set_positions((0, 0), (current_x, current_y))

            if t > 0.5:
                arrow.set_color(color_end)
                text.set_text(f"{scale_factor}v")
                text.set_color(color_end)
                text.set_alpha((t - 0.5) * 2)

            text.set_position((current_x + 0.2, current_y + 0.2))

            return arrow, text

        self._run_animation(fig, animate_frame, frames, interval, name)

    def animate_transform(
        self,
        matrix: Matrix,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        color_e1, color_e2 = _resolve_colors(colors, TRANSFORM_COLORS)

        fig, ax = plt.subplots(figsize=self.figsize)

        a, b = float(matrix[0][0]), float(matrix[0][1])
        c, d = float(matrix[1][0]), float(matrix[1][1])

        max_coord = max(abs(a), abs(b), abs(c), abs(d), 1.0) * 1.5
        grid_range = max(int(max_coord) + 2, 6)

        _setup_coordinate_plane(
            ax,
            (-grid_range, grid_range),
            (-grid_range, grid_range),
            grid=False,
        )

        subdivisions = 50
        grid_lines_init = []

        for i in range(-grid_range, grid_range + 1):
            ts = [
                -grid_range + k * (2 * grid_range) / subdivisions
                for k in range(subdivisions + 1)
            ]
            vertical = [(i, t) for t in ts]
            horizontal = [(t, i) for t in ts]
            grid_lines_init.append(vertical)
            grid_lines_init.append(horizontal)

        grid_collection = LineCollection(
            grid_lines_init,
            colors=GRID_COLOR,
            linewidths=0.8,
            alpha=0.6,
            zorder=1,
        )
        ax.add_collection(grid_collection)

        arrow_e1 = _create_vector_arrow((0, 0), (1, 0), color_e1, linewidth=3.5)
        arrow_e2 = _create_vector_arrow((0, 0), (0, 1), color_e2, linewidth=3.5)
        ax.add_patch(arrow_e1)
        ax.add_patch(arrow_e2)

        label_e1 = ax.text(
            1.15, 0.15, "e₁", fontsize=13, fontweight="bold", color=color_e1
        )
        label_e2 = ax.text(
            0.15, 1.15, "e₂", fontsize=13, fontweight="bold", color=color_e2
        )

        def animate_frame(frame: int) -> tuple:
            t = _smooth_step(frame / frames)

            mt_a = 1 + (a - 1) * t
            mt_b = b * t
            mt_c = c * t
            mt_d = 1 + (d - 1) * t

            transformed_lines = []
            for line in grid_lines_init:
                transformed = [
                    (mt_a * x + mt_b * y, mt_c * x + mt_d * y) for x, y in line
                ]
                transformed_lines.append(transformed)

            grid_collection.set_segments(transformed_lines)

            e1_x = mt_a * 1 + mt_b * 0
            e1_y = mt_c * 1 + mt_d * 0
            e2_x = mt_a * 0 + mt_b * 1
            e2_y = mt_c * 0 + mt_d * 1

            arrow_e1.set_positions((0, 0), (e1_x, e1_y))
            arrow_e2.set_positions((0, 0), (e2_x, e2_y))

            label_e1.set_position((e1_x + 0.15, e1_y + 0.15))
            label_e2.set_position((e2_x + 0.15, e2_y + 0.15))

            if t > 0.9:
                label_e1.set_text(f"[{a:.3g}, {c:.3g}]")
                label_e2.set_text(f"[{b:.3g}, {d:.3g}]")

            return grid_collection, arrow_e1, arrow_e2, label_e1, label_e2

        self._run_animation(fig, animate_frame, frames, interval, name)

    def plot_span(
        self,
        space: VectorSpace,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
        span_color: str | None = None,
    ) -> None:
        basis = space.basis
        dim = space.dims

        fig, ax = plt.subplots(figsize=self.figsize)

        if dim == 0:
            _setup_coordinate_plane(ax, (-3, 3), (-3, 3), grid)
            ax.plot(0, 0, "ko", markersize=8, zorder=5)
            self._finalize_figure(fig, name)
            return

        axis_range = _calculate_axis_range(basis)
        _setup_coordinate_plane(ax, axis_range, axis_range, grid)

        span_color = span_color or SPAN_COLOR

        if dim == 1:
            bx, by = float(basis[0][0]), float(basis[0][1])
            extent = max(abs(axis_range[0]), abs(axis_range[1])) * 2
            if abs(bx) > _EPSILON or abs(by) > _EPSILON:
                scale = extent / max(abs(bx), abs(by))
                ax.plot(
                    [-bx * scale, bx * scale],
                    [-by * scale, by * scale],
                    color=span_color,
                    linewidth=3,
                    alpha=0.4,
                    zorder=1,
                    label="span",
                )
        elif dim == 2:
            xlo, xhi = axis_range
            ylo, yhi = axis_range
            rect = plt.Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                color=span_color,
                alpha=0.12,
                zorder=1,
                label="span = R²",
            )
            ax.add_patch(rect)

        if colors is None:
            colors = DEFAULT_COLORS[: len(basis)]
        if labels is None:
            labels = [f"b{i + 1}" for i in range(len(basis))]

        for vec, color, label in zip(basis, colors, labels, strict=False):
            arrow = _create_vector_arrow((0, 0), (vec[0], vec[1]), color, linewidth=3)
            ax.add_patch(arrow)
            offset_x = 0.15 if vec[0] >= 0 else -0.15
            offset_y = 0.15 if vec[1] >= 0 else -0.15
            _add_vector_label(ax, (vec[0], vec[1]), label, color, (offset_x, offset_y))

        ax.legend(loc="upper left", fontsize=10)
        self._finalize_figure(fig, name)
