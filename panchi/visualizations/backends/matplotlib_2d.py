from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from panchi.algorithms.matrix_operations import rank
from panchi.algorithms.vector_space_operations import basis as compute_basis
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.matplotlib_base import (
    _EPSILON,
    ADDITION_COLORS,
    DEFAULT_COLORS,
    GRID_COLOR,
    SCALING_COLORS,
    SPAN_COLOR,
    TRANSFORM_COLORS,
    _calculate_axis_range,
    _InlineAnimation,
    _MatplotlibBackendBase,
    _resolve_colors,
    _resolve_n_colors,
    _smooth_step,
)


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


class _MatplotlibBackend2D(_MatplotlibBackendBase):
    """Matplotlib-based 2D visualization backend."""

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
    ) -> _InlineAnimation | None:
        color_v1, color_v2, color_result = _resolve_colors(colors, ADDITION_COLORS)

        fig, ax = plt.subplots(figsize=self.figsize)

        result = v1 + v2
        axis_range = _calculate_axis_range([v1, v2, result])
        _setup_coordinate_plane(ax, axis_range, axis_range, grid=True)

        arrow_v1 = _create_vector_arrow((0, 0), (v1[0], v1[1]), color_v1)
        arrow_v2_from_origin = _create_vector_arrow((0, 0), (v2[0], v2[1]), color_v2)
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
        ax.text(
            v2[0] / 2 + 0.2,
            v2[1] / 2 - 0.3,
            "v₂",
            fontsize=14,
            fontweight="bold",
            color=color_v2,
            alpha=0.5,
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
                arrow_v2_from_v1.set_positions(
                    (v1[0], v1[1]), (v1[0] + v2[0] * t, v1[1] + v2[1] * t)
                )
            else:
                t = (frame - half_frames) / half_frames

                arrow_v2_from_v1.set_positions((v1[0], v1[1]), (result[0], result[1]))

                result_x = result[0] * t
                result_y = result[1] * t
                arrow_result.set_positions((0, 0), (result_x, result_y))
                arrow_result.set_alpha(t)

                text_result.set_position((result_x / 2 - 0.3, result_y / 2 + 0.5))
                text_result.set_alpha(t)

            return arrow_v2_from_v1, arrow_result, text_result

        return self._run_animation(fig, animate_frame, frames, interval, name)

    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> _InlineAnimation | None:
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

        return self._run_animation(fig, animate_frame, frames, interval, name)

    def animate_transform(
        self,
        matrix: Matrix,
        vectors: list[Vector] | None = None,
        frames: int = 60,
        interval: int = 30,
        name: str = "animate_transform",
        colors: list[str] | None = None,
    ) -> _InlineAnimation | None:
        vecs = (
            [(float(v[0]), float(v[1])) for v in vectors]
            if vectors
            else [(1.0, 0.0), (0.0, 1.0)]
        )
        vec_colors = _resolve_n_colors(colors, len(vecs), TRANSFORM_COLORS)

        fig, ax = plt.subplots(figsize=self.figsize)

        a, b = float(matrix[0][0]), float(matrix[0][1])
        c, d = float(matrix[1][0]), float(matrix[1][1])

        coords = [abs(x) for v in vecs for x in v]
        max_coord = max(abs(a), abs(b), abs(c), abs(d), *coords, 1.0) * 1.5
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

        arrows = []
        labels = []
        for (vx, vy), color in zip(vecs, vec_colors, strict=False):
            arrow = _create_vector_arrow((0, 0), (vx, vy), color, linewidth=3.5)
            ax.add_patch(arrow)
            label = ax.text(
                vx + 0.15,
                vy + 0.15,
                f"[{vx:.3g}, {vy:.3g}]",
                fontsize=12,
                fontweight="bold",
                color=color,
            )
            arrows.append(arrow)
            labels.append(label)

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

            for (vx, vy), arrow, label in zip(vecs, arrows, labels, strict=False):
                tx = mt_a * vx + mt_b * vy
                ty = mt_c * vx + mt_d * vy
                arrow.set_positions((0, 0), (tx, ty))
                label.set_position((tx + 0.15, ty + 0.15))
                if t > 0.9:
                    label.set_text(f"[{tx:.3g}, {ty:.3g}]")

            return (grid_collection, *arrows, *labels)

        return self._run_animation(fig, animate_frame, frames, interval, name)

    def plot_span(
        self,
        spaces: list[VectorSpace],
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
        span_color: str | None = None,
    ) -> None:
        per_space = [(compute_basis(s), rank(s)) for s in spaces]
        all_basis = [v for basis, _ in per_space for v in basis]

        fig, ax = plt.subplots(figsize=self.figsize)
        axis_range = _calculate_axis_range(all_basis) if all_basis else (-3.0, 3.0)
        _setup_coordinate_plane(ax, axis_range, axis_range, grid)

        if len(per_space) == 1:
            basis, dim = per_space[0]
            dim_labels = {0: "span = {0}", 1: "span = line (R¹)", 2: "span = R²"}
            arrow_colors = colors or DEFAULT_COLORS[: len(basis)]
            arrow_labels = labels or [f"b{i + 1}" for i in range(len(basis))]
            self._draw_span_2d(
                ax,
                basis,
                dim,
                axis_range,
                fill_color=span_color or SPAN_COLOR,
                arrow_colors=arrow_colors,
                arrow_labels=arrow_labels,
                shade_label=dim_labels.get(dim, "span"),
            )
        else:
            span_colors = _resolve_n_colors(colors, len(per_space), DEFAULT_COLORS)
            for i, (basis, dim) in enumerate(per_space):
                label = labels[i] if labels and i < len(labels) else f"span {i + 1}"
                self._draw_span_2d(
                    ax,
                    basis,
                    dim,
                    axis_range,
                    fill_color=span_colors[i],
                    arrow_colors=[span_colors[i]] * len(basis),
                    arrow_labels=[None] * len(basis),
                    shade_label=label,
                )

        ax.legend(loc="upper left", fontsize=10)
        self._finalize_figure(fig, name)

    def _draw_span_2d(
        self,
        ax: plt.Axes,
        basis: list[Vector],
        dim: int,
        axis_range: tuple[float, float],
        *,
        fill_color: str,
        arrow_colors: list[str],
        arrow_labels: list[str | None],
        shade_label: str,
    ) -> None:
        if dim == 0:
            ax.plot(
                0, 0, "o", color=fill_color, markersize=8, zorder=5, label=shade_label
            )
            return

        if dim == 1:
            bx, by = float(basis[0][0]), float(basis[0][1])
            extent = max(abs(axis_range[0]), abs(axis_range[1])) * 2
            if abs(bx) > _EPSILON or abs(by) > _EPSILON:
                scale = extent / max(abs(bx), abs(by))
                ax.plot(
                    [-bx * scale, bx * scale],
                    [-by * scale, by * scale],
                    color=fill_color,
                    linewidth=3,
                    alpha=0.4,
                    zorder=1,
                    label=shade_label,
                )
        elif dim == 2:
            xlo, xhi = axis_range
            ylo, yhi = axis_range
            rect = plt.Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                color=fill_color,
                alpha=0.12,
                zorder=1,
                label=shade_label,
            )
            ax.add_patch(rect)

        for vec, color, label in zip(basis, arrow_colors, arrow_labels, strict=False):
            arrow = _create_vector_arrow((0, 0), (vec[0], vec[1]), color, linewidth=3)
            ax.add_patch(arrow)
            if label:
                offset_x = 0.15 if vec[0] >= 0 else -0.15
                offset_y = 0.15 if vec[1] >= 0 else -0.15
                _add_vector_label(
                    ax, (vec[0], vec[1]), label, color, (offset_x, offset_y)
                )
