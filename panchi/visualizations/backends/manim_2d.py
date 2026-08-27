from __future__ import annotations

import manim
from manim import (
    Arrow,
    Axes,
    Create,
    DashedLine,
    FadeIn,
    GrowArrow,
    Line,
    MathTex,
    NumberPlane,
    Polygon,
    Rectangle,
    Scene,
    Transform,
    Write,
)

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.manim_base import (
    ADDITION_COLORS,
    AXIS_COLOR,
    DEFAULT_COLORS,
    GUIDE_COLOR,
    MATRIX_HUD_COLOR,
    PARALLELOGRAM_COLOR,
    SCALING_COLORS,
    SPAN_COLOR,
    TRANSFORM_COLORS,
    _axis_range,
    _BuilderSceneMixin,
    _ManimBackendBase,
    _resolve_colors,
)

_EPSILON = 1e-12

_USABLE_WIDTH = 12.0
_USABLE_HEIGHT = 6.8


def _label_direction(vx: float, vy: float) -> manim.Vector:
    ax, ay = abs(vx), abs(vy)
    if ax < 0.3 and ay < 0.3:
        return manim.UP + manim.RIGHT
    if ax < 0.3:
        return manim.RIGHT if vy > 0 else manim.RIGHT
    if ay < 0.3:
        return manim.UP if vx > 0 else manim.UP
    if vx > 0 and vy > 0:
        return manim.UP + manim.LEFT
    if vx < 0 and vy > 0:
        return manim.UP + manim.RIGHT
    if vx < 0 and vy < 0:
        return manim.DOWN + manim.RIGHT
    return manim.DOWN + manim.LEFT


def _arrow(
    coords,
    start,
    end,
    color,
    stroke_width: float = 6,
    tip_ratio: float = 0.15,
) -> Arrow:
    """Build an arrow between two data-space points on ``coords`` (Axes/NumberPlane)."""
    return Arrow(
        coords.c2p(*start),
        coords.c2p(*end),
        buff=0,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=tip_ratio,
    )


def _label(
    text: str,
    color,
    arrow: Arrow,
    direction_coords,
    scale: float = 0.9,
    buff: float = 0.2,
) -> MathTex:
    """Build a ``MathTex`` label placed just past an arrow's tip."""
    mob = MathTex(text, color=color).scale(scale)
    mob.next_to(
        arrow.get_end(),
        _label_direction(direction_coords[0], direction_coords[1]),
        buff=buff,
    )
    return mob


class _VectorScene(Scene):
    """Shared base scene with coordinate-plane setup and a vector-drawing helper."""

    def setup_axes(
        self,
        x_range: tuple[int, int] = (-5, 5),
        y_range: tuple[int, int] = (-5, 5),
    ) -> None:
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        unit = min(_USABLE_WIDTH / x_span, _USABLE_HEIGHT / y_span)

        self.axes = Axes(
            x_range=[*x_range, 1],
            y_range=[*y_range, 1],
            x_length=unit * x_span,
            y_length=unit * y_span,
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 2,
                "include_tip": True,
                "tip_width": 0.2,
                "tip_height": 0.2,
            },
            x_axis_config={
                "numbers_to_include": range(x_range[0], x_range[1] + 1, 1),
            },
            y_axis_config={
                "numbers_to_include": range(y_range[0], y_range[1] + 1, 1),
            },
        ).add_coordinates()

        self.axes.x_axis.set_opacity(0.8)
        self.axes.y_axis.set_opacity(0.8)

        x_label = MathTex("x").next_to(self.axes.x_axis.get_end(), manim.RIGHT)
        y_label = MathTex("y").next_to(self.axes.y_axis.get_end(), manim.UP)

        self.play(Create(self.axes), Write(x_label), Write(y_label))
        self.wait(0.3)

    def add_vector(
        self,
        coords,
        color,
        label: str | None = None,
        *,
        start=(0, 0),
        label_dir=None,
        stroke_width: float = 6,
        label_scale: float = 0.9,
        run_time: float = 0.8,
        opacity: float = 1.0,
        wait: float = 0.0,
    ) -> tuple[Arrow, MathTex | None]:
        """Grow an arrow (optionally from a non-origin ``start``) with a label.

        ``label_dir`` chooses the quadrant the label is nudged toward; it
        defaults to ``coords`` but can differ when an arrow is drawn shifted
        (e.g. the tip-to-tail second vector in an addition).
        """
        arrow = _arrow(self.axes, start, coords, color, stroke_width)
        label_mob = None
        if label is not None:
            label_mob = _label(label, color, arrow, label_dir or coords, label_scale)

        if opacity != 1.0:
            arrow.set_opacity(opacity)
            if label_mob is not None:
                label_mob.set_opacity(opacity)

        anims = [GrowArrow(arrow)]
        if label_mob is not None:
            anims.append(Write(label_mob))
        self.play(*anims, run_time=run_time)
        if wait:
            self.wait(wait)
        return arrow, label_mob


class _FunctionScene(_BuilderSceneMixin, _VectorScene):
    """A 2D scene whose ``construct`` delegates to a supplied builder callable."""


class _ManimBackend2D(_ManimBackendBase):
    """Manim-based 2D visualization backend."""

    _function_scene_cls = _FunctionScene

    def plot_vectors(
        self,
        vectors: list[Vector],
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None:
        vec_data = vectors
        color_list = colors or DEFAULT_COLORS[: len(vec_data)]
        label_list = labels or [f"v_{{{i + 1}}}" for i in range(len(vec_data))]

        def build(scene: _VectorScene) -> None:
            range_val = _axis_range(c for v in vec_data for c in (v[0], v[1]))
            scene.setup_axes((-range_val, range_val), (-range_val, range_val))

            for vec, color, label in zip(
                vec_data, color_list, label_list, strict=False
            ):
                scene.add_vector((vec[0], vec[1]), color, label)

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

        v1c = (v1[0], v1[1])
        v2c = (v2[0], v2[1])
        result = (v1[0] + v2[0], v1[1] + v2[1])

        def build(scene: _VectorScene) -> None:
            range_val = _axis_range((*v1c, *v2c, *result))
            scene.setup_axes((-range_val, range_val), (-range_val, range_val))

            scene.add_vector(
                v1c, color_v1, r"v_1", stroke_width=7, run_time=1, wait=0.5
            )
            scene.add_vector(
                v2c, color_v2, r"v_2", stroke_width=7, run_time=1, opacity=0.3, wait=0.5
            )
            scene.add_vector(
                result,
                color_v2,
                r"v_2",
                start=v1c,
                label_dir=v2c,
                stroke_width=7,
                run_time=1.2,
                wait=0.5,
            )

            dashed_line_1 = DashedLine(
                scene.axes.c2p(*v2c),
                scene.axes.c2p(*result),
                color=GUIDE_COLOR,
                stroke_width=2,
                dash_length=0.1,
            )
            dashed_line_2 = DashedLine(
                scene.axes.c2p(0, 0),
                scene.axes.c2p(*v2c),
                color=GUIDE_COLOR,
                stroke_width=2,
                dash_length=0.1,
            )
            parallelogram = Polygon(
                scene.axes.c2p(0, 0),
                scene.axes.c2p(*v1c),
                scene.axes.c2p(*result),
                scene.axes.c2p(*v2c),
                color=PARALLELOGRAM_COLOR,
                fill_opacity=0.15,
                stroke_width=0,
            )
            scene.play(
                Create(dashed_line_1),
                Create(dashed_line_2),
                FadeIn(parallelogram),
                run_time=1,
            )
            scene.wait(0.5)

            scene.add_vector(
                result,
                color_result,
                r"v_1 + v_2",
                stroke_width=8,
                label_scale=1.0,
                run_time=1.5,
                wait=2,
            )

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

        v_coords = (vector[0], vector[1])
        scaled_coords = (vector[0] * scale_factor, vector[1] * scale_factor)

        def build(scene: _VectorScene) -> None:
            range_val = _axis_range((*v_coords, *scaled_coords))
            scene.setup_axes((-range_val, range_val), (-range_val, range_val))

            arrow, label = scene.add_vector(
                v_coords, color_start, r"v", stroke_width=7, run_time=1, wait=0.5
            )

            scaled_arrow = _arrow(
                scene.axes, (0, 0), scaled_coords, color_end, stroke_width=7
            )
            scale_text = _label(
                f"{scale_factor} \\cdot v", color_end, scaled_arrow, scaled_coords
            )
            scene.play(
                Transform(arrow, scaled_arrow),
                Transform(label, scale_text),
                run_time=2,
            )
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
        color_e1, color_e2 = _resolve_colors(colors, TRANSFORM_COLORS)

        mat = [
            [float(matrix[0][0]), float(matrix[0][1])],
            [float(matrix[1][0]), float(matrix[1][1])],
        ]

        def build(scene: _VectorScene) -> None:
            plane = NumberPlane(
                x_range=[-5, 5, 1],
                y_range=[-5, 5, 1],
                background_line_style={
                    "stroke_color": GUIDE_COLOR,
                    "stroke_width": 1,
                    "stroke_opacity": 0.6,
                },
            )
            scene.play(Create(plane), run_time=1)

            matrix_tex = MathTex(
                r"\begin{bmatrix}"
                f" {mat[0][0]:.3g} & {mat[0][1]:.3g} "
                r"\\"
                f" {mat[1][0]:.3g} & {mat[1][1]:.3g} "
                r"\end{bmatrix}",
                color=MATRIX_HUD_COLOR,
            ).scale(0.9)
            matrix_tex.to_corner(manim.UR, buff=0.5)
            scene.play(Write(matrix_tex), run_time=0.8)

            arrow_e1 = _arrow(plane, (0, 0), (1, 0), color_e1, 7, tip_ratio=0.2)
            arrow_e2 = _arrow(plane, (0, 0), (0, 1), color_e2, 7, tip_ratio=0.2)

            label_e1 = MathTex(r"\hat{e}_1", color=color_e1).scale(0.8)
            label_e1.next_to(arrow_e1.get_end(), manim.DOWN + manim.RIGHT, buff=0.2)
            label_e2 = MathTex(r"\hat{e}_2", color=color_e2).scale(0.8)
            label_e2.next_to(arrow_e2.get_end(), manim.UP + manim.LEFT, buff=0.2)

            scene.play(
                GrowArrow(arrow_e1),
                GrowArrow(arrow_e2),
                Write(label_e1),
                Write(label_e2),
                run_time=1,
            )
            scene.wait(0.5)

            label_e1.add_updater(
                lambda m: m.next_to(
                    arrow_e1.get_end(), manim.DOWN + manim.RIGHT, buff=0.2
                )
            )
            label_e2.add_updater(
                lambda m: m.next_to(arrow_e2.get_end(), manim.UP + manim.LEFT, buff=0.2)
            )

            scene.play(
                plane.animate.apply_matrix(mat),
                arrow_e1.animate.put_start_and_end_on(
                    plane.c2p(0, 0),
                    plane.c2p(mat[0][0], mat[1][0]),
                ),
                arrow_e2.animate.put_start_and_end_on(
                    plane.c2p(0, 0),
                    plane.c2p(mat[0][1], mat[1][1]),
                ),
                run_time=3,
            )

            label_e1.clear_updaters()
            label_e2.clear_updaters()
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

        def build(scene: _VectorScene) -> None:
            if dim == 0:
                scene.setup_axes((-3, 3), (-3, 3))
                dot = manim.Dot(scene.axes.c2p(0, 0), color=manim.WHITE, radius=0.08)
                scene.play(Create(dot))
                scene.wait(1)
                return

            range_val = _axis_range(c for v in basis for c in (v[0], v[1]))
            scene.setup_axes((-range_val, range_val), (-range_val, range_val))

            if dim == 1:
                bx, by = float(basis[0][0]), float(basis[0][1])
                extent = range_val * 2
                scale = (
                    extent / max(abs(bx), abs(by))
                    if max(abs(bx), abs(by)) > _EPSILON
                    else 1
                )
                span_line = Line(
                    scene.axes.c2p(-bx * scale, -by * scale),
                    scene.axes.c2p(bx * scale, by * scale),
                    color=span_color,
                    stroke_width=3,
                    stroke_opacity=0.5,
                )
                scene.play(Create(span_line), run_time=0.8)

            elif dim == 2:
                rect = Rectangle(
                    width=range_val * 2 * scene.axes.x_axis.get_unit_size(),
                    height=range_val * 2 * scene.axes.y_axis.get_unit_size(),
                    color=span_color,
                    fill_opacity=0.1,
                    stroke_width=0,
                )
                rect.move_to(scene.axes.c2p(0, 0))
                scene.play(FadeIn(rect), run_time=0.8)

            for vec, color, label in zip(basis, color_list, label_list, strict=False):
                scene.add_vector((vec[0], vec[1]), color, label)

            scene.wait(1.5)

        self._render(name, build)
