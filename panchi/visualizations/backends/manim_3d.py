from __future__ import annotations

import manim
from manim import (
    DEGREES,
    Arrow3D,
    Create,
    Dot3D,
    FadeIn,
    Line3D,
    MathTex,
    Polygon,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
)

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.geometry import (
    CUBE_EDGES,
    CUBE_VERTS,
    apply_3x3,
)
from panchi.visualizations.backends.manim_base import (
    _EPSILON,
    ADDITION_COLORS,
    AXIS_COLOR,
    DEFAULT_COLORS,
    GUIDE_COLOR,
    MATRIX_HUD_COLOR,
    PARALLELOGRAM_COLOR,
    SCALING_COLORS,
    SPAN_COLOR,
    TRANSFORM_COLORS_3D,
    _axis_range,
    _BuilderSceneMixin,
    _ManimBackendBase,
    _resolve_colors,
)

CAMERA_PHI = 70 * DEGREES
CAMERA_THETA = -45 * DEGREES


class _VectorScene3D(ThreeDScene):
    """Base 3D scene with a tilted camera and a vector-drawing helper."""

    def setup_axes(
        self,
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        z_range: tuple[int, int],
    ) -> None:
        self.axes = ThreeDAxes(
            x_range=[*x_range, 1],
            y_range=[*y_range, 1],
            z_range=[*z_range, 1],
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 2,
                "include_tip": True,
            },
        )

        self.set_camera_orientation(phi=CAMERA_PHI, theta=CAMERA_THETA)
        self.play(Create(self.axes))

        # Labels sit just past each axis tip and face the camera (fixed
        # orientation), so they stay legible and anchored to their own axis
        # instead of lying flat in the world plane.
        ends = ((x_range[1], 0, 0), (0, y_range[1], 0), (0, 0, z_range[1]))
        for name, end in zip(("x", "y", "z"), ends, strict=False):
            label = MathTex(name, color=AXIS_COLOR).scale(0.8)
            label.move_to(self.axes.c2p(*end) * 1.08)
            self.add_fixed_orientation_mobjects(label)
        self.wait(0.3)

    def add_vector(
        self,
        coords,
        color,
        label: str | None = None,
        *,
        start=(0, 0, 0),
        opacity: float = 1.0,
        run_time: float = 0.8,
        wait: float = 0.0,
    ) -> tuple[Arrow3D, MathTex | None]:
        """Draw a 3D arrow (optionally from a non-origin ``start``) with a label.

        The label is a fixed-orientation mobject: it sits just past the arrow
        tip in space but always faces the camera, so it stays legible under the
        tilted 3D view instead of lying flat in the world plane.
        """
        tip = self.axes.c2p(*coords)
        arrow = Arrow3D(self.axes.c2p(*start), tip, color=color)
        if opacity != 1.0:
            arrow.set_opacity(opacity)
        self.play(Create(arrow), run_time=run_time)

        label_mob = None
        if label is not None:
            label_mob = MathTex(label, color=color).scale(0.9)
            label_mob.move_to(tip * 1.15)
            if opacity != 1.0:
                label_mob.set_opacity(opacity)
            self.add_fixed_orientation_mobjects(label_mob)

        if wait:
            self.wait(wait)
        return arrow, label_mob


class _FunctionScene3D(_BuilderSceneMixin, _VectorScene3D):
    """A 3D scene whose ``construct`` delegates to a supplied builder callable."""


class _ManimBackend3D(_ManimBackendBase):
    """Manim-based 3D visualization backend."""

    _function_scene_cls = _FunctionScene3D

    def plot_vectors(
        self,
        vectors: list[Vector],
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None:
        color_list = colors or DEFAULT_COLORS[: len(vectors)]
        label_list = labels or [f"v_{{{i + 1}}}" for i in range(len(vectors))]

        def build(scene: _VectorScene3D) -> None:
            range_val = _axis_range(c for v in vectors for c in (v[0], v[1], v[2]))
            axis_range = (-range_val, range_val)
            scene.setup_axes(axis_range, axis_range, axis_range)

            for vec, color, label in zip(vectors, color_list, label_list, strict=False):
                scene.add_vector((vec[0], vec[1], vec[2]), color, label)

            scene.wait(1.5)

        self._render(name, build)

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

        v1c = (v1[0], v1[1], v1[2])
        v2c = (v2[0], v2[1], v2[2])
        result = (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

        def build(scene: _VectorScene3D) -> None:
            range_val = _axis_range((*v1c, *v2c, *result))
            axis_range = (-range_val, range_val)
            scene.setup_axes(axis_range, axis_range, axis_range)

            scene.add_vector(v1c, color_v1, r"v_1", run_time=1, wait=0.4)
            scene.add_vector(v2c, color_v2, r"v_2", opacity=0.35, run_time=1, wait=0.4)
            scene.add_vector(result, color_v2, start=v1c, run_time=1.2, wait=0.4)

            parallelogram = Polygon(
                scene.axes.c2p(0, 0, 0),
                scene.axes.c2p(*v1c),
                scene.axes.c2p(*result),
                scene.axes.c2p(*v2c),
                color=PARALLELOGRAM_COLOR,
                fill_opacity=0.15,
                stroke_width=0,
            )
            scene.play(FadeIn(parallelogram), run_time=1)
            scene.wait(0.3)

            scene.add_vector(result, color_result, r"v_1 + v_2", run_time=1.5, wait=2)

        self._render(name, build)

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

        v_coords = (vector[0], vector[1], vector[2])
        scaled_coords = (
            vector[0] * scale_factor,
            vector[1] * scale_factor,
            vector[2] * scale_factor,
        )

        def build(scene: _VectorScene3D) -> None:
            range_val = _axis_range((*v_coords, *scaled_coords))
            axis_range = (-range_val, range_val)
            scene.setup_axes(axis_range, axis_range, axis_range)

            arrow, label = scene.add_vector(
                v_coords, color_start, r"v", run_time=1, wait=0.5
            )

            scaled_arrow = Arrow3D(
                scene.axes.c2p(0, 0, 0),
                scene.axes.c2p(*scaled_coords),
                color=color_end,
            )
            scaled_label = MathTex(f"{scale_factor} \\cdot v", color=color_end).scale(
                0.9
            )
            scaled_label.move_to(scene.axes.c2p(*scaled_coords) * 1.15)

            scene.play(Transform(arrow, scaled_arrow), run_time=2)
            if label is not None:
                scene.remove(label)
            scene.add_fixed_orientation_mobjects(scaled_label)
            scene.wait(1.5)

        self._render(name, build)

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
        basis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        def build(scene: _VectorScene3D) -> None:
            image = [apply_3x3(target, v) for v in CUBE_VERTS]
            coords = [c for corner in image for c in corner]
            range_val = max(_axis_range(coords) if coords else 2, 2)
            axis_range = (-range_val, range_val)
            scene.setup_axes(axis_range, axis_range, axis_range)

            matrix_tex = MathTex(
                r"\begin{bmatrix}"
                + r"\\".join(
                    " & ".join(f"{target[i][j]:.2g}" for j in range(3))
                    for i in range(3)
                )
                + r"\end{bmatrix}",
                color=MATRIX_HUD_COLOR,
            ).scale(0.7)
            matrix_tex.to_corner(manim.UR, buff=0.4)
            scene.add_fixed_in_frame_mobjects(matrix_tex)
            scene.play(Write(matrix_tex), run_time=0.6)

            def cube_group(m):
                group = VGroup()
                for i, j in CUBE_EDGES:
                    p1 = apply_3x3(m, CUBE_VERTS[i])
                    p2 = apply_3x3(m, CUBE_VERTS[j])
                    group.add(
                        Line3D(
                            scene.axes.c2p(*p1),
                            scene.axes.c2p(*p2),
                            color=GUIDE_COLOR,
                            thickness=0.015,
                        )
                    )
                return group

            identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            cube = cube_group(identity)
            arrows = [
                Arrow3D(scene.axes.c2p(0, 0, 0), scene.axes.c2p(*b), color=color)
                for b, color in zip(basis, basis_colors, strict=False)
            ]
            scene.play(Create(cube), *[Create(a) for a in arrows], run_time=1.2)
            scene.wait(0.4)

            cube_target = cube_group(target)
            arrow_targets = [
                Arrow3D(
                    scene.axes.c2p(0, 0, 0),
                    scene.axes.c2p(*apply_3x3(target, b)),
                    color=color,
                )
                for b, color in zip(basis, basis_colors, strict=False)
            ]
            scene.play(
                Transform(cube, cube_target),
                *[
                    Transform(a, at)
                    for a, at in zip(arrows, arrow_targets, strict=False)
                ],
                run_time=3,
            )
            scene.wait(2)

        self._render(name, build)

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
        color_list = colors or DEFAULT_COLORS[: len(basis)]
        label_list = labels or [f"b_{{{i + 1}}}" for i in range(len(basis))]
        span_color = span_color or SPAN_COLOR

        def build(scene: _VectorScene3D) -> None:
            if dim == 0:
                scene.setup_axes((-3, 3), (-3, 3), (-3, 3))
                dot = Dot3D(scene.axes.c2p(0, 0, 0), color=manim.WHITE)
                scene.play(Create(dot))
                scene.wait(1)
                return

            range_val = _axis_range(c for v in basis for c in (v[0], v[1], v[2]))
            axis_range = (-range_val, range_val)
            scene.setup_axes(axis_range, axis_range, axis_range)

            if dim == 1:
                b = basis[0]
                scale = range_val / max(abs(b[0]), abs(b[1]), abs(b[2]), _EPSILON) * 1.4
                line = Line3D(
                    scene.axes.c2p(-b[0] * scale, -b[1] * scale, -b[2] * scale),
                    scene.axes.c2p(b[0] * scale, b[1] * scale, b[2] * scale),
                    color=span_color,
                )
                scene.play(Create(line))
            elif dim == 2:
                b1, b2 = basis[0], basis[1]
                max_comp = max(
                    abs(c) for c in (b1[0], b1[1], b1[2], b2[0], b2[1], b2[2])
                )
                f = range_val / max(max_comp, _EPSILON) * 1.3

                def corner(sa, sb):
                    return scene.axes.c2p(
                        *(sa * f * b1[k] + sb * f * b2[k] for k in range(3))
                    )

                plane = Polygon(
                    corner(-1, -1),
                    corner(1, -1),
                    corner(1, 1),
                    corner(-1, 1),
                    color=span_color,
                    fill_opacity=0.25,
                    stroke_opacity=0.4,
                )
                scene.play(FadeIn(plane))
            else:  # dim == 3
                h = range_val * 0.7
                verts = [tuple((2 * c - 1) * h for c in v) for v in CUBE_VERTS]
                edges = VGroup(
                    *[
                        Line3D(
                            scene.axes.c2p(*verts[i]),
                            scene.axes.c2p(*verts[j]),
                            color=span_color,
                            thickness=0.01,
                        )
                        for i, j in CUBE_EDGES
                    ]
                )
                scene.play(Create(edges))

            for vec, color, label in zip(basis, color_list, label_list, strict=False):
                scene.add_vector((vec[0], vec[1], vec[2]), color, label)

            scene.wait(1.5)

        self._render(name, build)
