from __future__ import annotations

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.geometry import (
    CUBE_EDGES,
    CUBE_VERTS,
    apply_3x3,
)
from panchi.visualizations.backends.matplotlib_base import (
    _EPSILON,
    ADDITION_COLORS,
    DEFAULT_COLORS,
    GRID_COLOR,
    SCALING_COLORS,
    SPAN_COLOR,
    TRANSFORM_COLORS_3D,
    _calculate_axis_range,
    _MatplotlibBackendBase,
    _resolve_colors,
    _smooth_step,
)

AXIS_LINE_COLOR = "#666666"
HUD_COLOR = "#333333"


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


def _span_annotation(ax, text: str) -> None:
    """Label the span dimension as a fixed 2D overlay (doesn't tumble with the view)."""
    ax.text2D(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=HUD_COLOR,
    )


def _draw_arrow_3d(ax, start, disp, color, linewidth: float = 2.5):
    """Draw a 3D arrow (quiver) from ``start`` along displacement ``disp``."""
    return ax.quiver(
        start[0],
        start[1],
        start[2],
        disp[0],
        disp[1],
        disp[2],
        color=color,
        arrow_length_ratio=0.15,
        linewidth=linewidth,
    )


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
            _draw_arrow_3d(ax, (0, 0, 0), (vec[0], vec[1], vec[2]), color)

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
        color_v1, color_v2, color_result = _resolve_colors(colors, ADDITION_COLORS)

        result = v1 + v2
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")
        axis_range = _calculate_axis_range([v1, v2, result])
        _setup_coordinate_space_3d(ax, axis_range, grid=True)

        _draw_arrow_3d(ax, (0, 0, 0), (v1[0], v1[1], v1[2]), color_v1)
        faded_v2 = _draw_arrow_3d(ax, (0, 0, 0), (v2[0], v2[1], v2[2]), color_v2)
        faded_v2.set_alpha(0.4)
        ax.text(
            v1[0] * 1.05,
            v1[1] * 1.05,
            v1[2] * 1.05,
            "v₁",
            color=color_v1,
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            v2[0] * 1.05,
            v2[1] * 1.05,
            v2[2] * 1.05,
            "v₂",
            color=color_v2,
            alpha=0.6,
            fontsize=12,
            fontweight="bold",
        )
        parallelogram = Poly3DCollection(
            [
                [
                    (0.0, 0.0, 0.0),
                    (v1[0], v1[1], v1[2]),
                    (result[0], result[1], result[2]),
                    (v2[0], v2[1], v2[2]),
                ]
            ],
            facecolor=color_result,
            alpha=0.1,
        )
        ax.add_collection3d(parallelogram)

        half_frames = frames // 2
        state = {"tiptail": None, "result": None, "text": None}

        def animate_frame(frame: int) -> tuple:
            if frame < half_frames:
                t = frame / half_frames
                end = (
                    v1[0] + v2[0] * t,
                    v1[1] + v2[1] * t,
                    v1[2] + v2[2] * t,
                )
            else:
                end = (result[0], result[1], result[2])

            if state["tiptail"] is not None:
                state["tiptail"].remove()
            state["tiptail"] = _draw_arrow_3d(
                ax,
                (v1[0], v1[1], v1[2]),
                (end[0] - v1[0], end[1] - v1[1], end[2] - v1[2]),
                color_v2,
            )

            if frame >= half_frames:
                t2 = (frame - half_frames) / half_frames
                rx, ry, rz = result[0] * t2, result[1] * t2, result[2] * t2

                if state["result"] is not None:
                    state["result"].remove()
                state["result"] = _draw_arrow_3d(
                    ax, (0, 0, 0), (rx, ry, rz), color_result, linewidth=3.5
                )

                if state["text"] is not None:
                    state["text"].remove()
                state["text"] = ax.text(
                    rx * 1.05,
                    ry * 1.05,
                    rz * 1.05,
                    "v₁+v₂",
                    color=color_result,
                    fontsize=12,
                    fontweight="bold",
                )

            return ()

        self._run_animation(fig, animate_frame, frames, interval, name, blit=False)

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

        scaled = scale_factor * vector
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")
        axis_range = _calculate_axis_range([vector, scaled])
        _setup_coordinate_space_3d(ax, axis_range, grid=True)

        state = {"arrow": None, "text": None}

        def animate_frame(frame: int) -> tuple:
            t = frame / frames
            scale = 1 + (scale_factor - 1) * t
            vx, vy, vz = vector[0] * scale, vector[1] * scale, vector[2] * scale

            color = color_end if t > 0.5 else color_start
            label = f"{scale_factor}v" if t > 0.5 else "v"

            if state["arrow"] is not None:
                state["arrow"].remove()
            state["arrow"] = _draw_arrow_3d(ax, (0, 0, 0), (vx, vy, vz), color)

            if state["text"] is not None:
                state["text"].remove()
            state["text"] = ax.text(
                vx * 1.1,
                vy * 1.1,
                vz * 1.1,
                label,
                fontsize=12,
                fontweight="bold",
                color=color,
            )

            return ()

        self._run_animation(fig, animate_frame, frames, interval, name, blit=False)

    def animate_transform(
        self,
        matrix: Matrix,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> None:
        basis_colors = _resolve_colors(colors, TRANSFORM_COLORS_3D)
        target = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")

        image = [apply_3x3(target, v) for v in CUBE_VERTS]
        coords = [c for corner in image for c in corner]
        rng = max(max((abs(c) for c in coords), default=1.0) * 1.3, 1.5)
        _setup_coordinate_space_3d(ax, (-rng, rng), grid=False)

        def cube_segments(m):
            pts = [apply_3x3(m, v) for v in CUBE_VERTS]
            return [[pts[i], pts[j]] for i, j in CUBE_EDGES]

        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cube = Line3DCollection(
            cube_segments(identity),
            colors=GRID_COLOR,
            linewidths=1.4,
            alpha=0.8,
        )
        ax.add_collection3d(cube)

        matrix_text = "\n".join(
            "[" + "  ".join(f"{target[i][j]:+.2g}" for j in range(3)) + "]"
            for i in range(3)
        )
        ax.text2D(
            0.02,
            0.98,
            matrix_text,
            transform=ax.transAxes,
            va="top",
            family="monospace",
            fontsize=11,
            color=HUD_COLOR,
        )

        basis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        state = {"arrows": []}

        def animate_frame(frame: int) -> tuple:
            t = _smooth_step(frame / frames)
            current = [
                [
                    (1.0 if i == j else 0.0) * (1 - t) + target[i][j] * t
                    for j in range(3)
                ]
                for i in range(3)
            ]

            cube.set_segments(cube_segments(current))

            for arrow in state["arrows"]:
                arrow.remove()
            state["arrows"] = [
                _draw_arrow_3d(ax, (0, 0, 0), apply_3x3(current, b), color, linewidth=3)
                for b, color in zip(basis, basis_colors, strict=False)
            ]

            return ()

        self._run_animation(fig, animate_frame, frames, interval, name, blit=False)

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
        span_color = span_color or SPAN_COLOR

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")

        if dim == 0:
            _setup_coordinate_space_3d(ax, (-3, 3), grid)
            ax.scatter([0], [0], [0], color="black", s=40, zorder=5)
            _span_annotation(ax, "span = {0}")
            self._finalize_figure(fig, name)
            return

        axis_range = _calculate_axis_range(basis)
        _setup_coordinate_space_3d(ax, axis_range, grid)
        hi = axis_range[1]

        if dim == 1:
            b = basis[0]
            scale = hi / max(abs(b[0]), abs(b[1]), abs(b[2]), _EPSILON) * 1.4
            ax.plot(
                [-b[0] * scale, b[0] * scale],
                [-b[1] * scale, b[1] * scale],
                [-b[2] * scale, b[2] * scale],
                color=span_color,
                linewidth=3,
                alpha=0.55,
            )
            span_label = "span = line"
        elif dim == 2:
            b1, b2 = basis[0], basis[1]
            max_comp = max(abs(c) for c in (b1[0], b1[1], b1[2], b2[0], b2[1], b2[2]))
            f = hi / max(max_comp, _EPSILON) * 1.3

            def corner(sa, sb):
                return tuple(sa * f * b1[k] + sb * f * b2[k] for k in range(3))

            patch = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
            ax.add_collection3d(
                Poly3DCollection([patch], facecolor=span_color, alpha=0.22)
            )
            span_label = "span = plane"
        else:  # dim == 3
            h = hi * 0.7
            verts = [tuple((2 * c - 1) * h for c in v) for v in CUBE_VERTS]
            segments = [[verts[i], verts[j]] for i, j in CUBE_EDGES]
            ax.add_collection3d(
                Line3DCollection(segments, colors=span_color, alpha=0.4, linewidths=1.2)
            )
            span_label = "span = R³"

        _span_annotation(ax, span_label)

        if colors is None:
            colors = DEFAULT_COLORS[: len(basis)]
        if labels is None:
            labels = [f"b{i + 1}" for i in range(len(basis))]

        for vec, color, label in zip(basis, colors, labels, strict=False):
            _draw_arrow_3d(ax, (0, 0, 0), (vec[0], vec[1], vec[2]), color, linewidth=3)
            ax.text(
                vec[0] * 1.05,
                vec[1] * 1.05,
                vec[2] * 1.05,
                label,
                color=color,
                fontsize=12,
                fontweight="bold",
            )

        self._finalize_figure(fig, name)
