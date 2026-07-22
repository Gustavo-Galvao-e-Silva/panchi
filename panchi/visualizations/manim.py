from __future__ import annotations

from pathlib import Path

try:
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

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    raise

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.base import _BaseBackend

DEFAULT_COLORS = [manim.RED, manim.ORANGE, manim.GREEN, manim.BLUE, manim.PURPLE]

# Per-method default color roles, overridable via ``colors=`` / ``span_color=``.
ADDITION_COLORS = (manim.RED, manim.ORANGE, manim.GREEN)  # v1, v2, result
SCALING_COLORS = (manim.RED, manim.BLUE)  # original, scaled
TRANSFORM_COLORS = (manim.RED, manim.BLUE)  # e1, e2
SPAN_COLOR = manim.PURPLE


def _resolve_colors(colors: list[str] | None, defaults: tuple) -> list:
    """Fill in per-role colors positionally, keeping defaults for omitted roles."""
    colors = colors or []
    return [colors[i] if i < len(colors) else defaults[i] for i in range(len(defaults))]


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


_QUALITY_MAP = {
    "low": "low_quality",
    "medium": "medium_quality",
    "high": "high_quality",
    "production": "production_quality",
}


class _VectorScene(Scene):
    """Shared base scene with coordinate plane setup."""

    def setup_axes(
        self,
        x_range: tuple[int, int] = (-5, 5),
        y_range: tuple[int, int] = (-5, 5),
    ) -> None:
        self.axes = Axes(
            x_range=[*x_range, 1],
            y_range=[*y_range, 1],
            axis_config={
                "color": manim.GREY_B,
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


class _ManimBackend(_BaseBackend):
    """Manim-based 2D visualization backend."""

    def __init__(self, save_path: Path | None, quality: str) -> None:
        super().__init__(save_path=save_path, quality=quality)

    def _configure(self, name: str) -> None:
        manim.config.quality = _QUALITY_MAP.get(self.quality, "medium_quality")
        manim.config.disable_caching = True
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            manim.config.media_dir = str(self.save_path)
            manim.config.video_dir = "{media_dir}"
            manim.config.output_file = name

    def plot_vectors(
        self,
        *vectors: Vector,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None:
        vec_data = list(vectors)
        color_list = colors or DEFAULT_COLORS[: len(vec_data)]
        label_list = labels or [f"v_{{{i + 1}}}" for i in range(len(vec_data))]

        class _PlotScene(_VectorScene):
            def construct(inner_self) -> None:
                all_coords = [(v[0], v[1]) for v in vec_data]
                max_coord = max(abs(c) for vec in all_coords for c in vec)
                range_val = int(max_coord * 1.3) + 1

                inner_self.setup_axes(
                    x_range=(-range_val, range_val),
                    y_range=(-range_val, range_val),
                )

                for vec, color, label in zip(
                    vec_data, color_list, label_list, strict=False
                ):
                    arrow = Arrow(
                        inner_self.axes.c2p(0, 0),
                        inner_self.axes.c2p(vec[0], vec[1]),
                        buff=0,
                        color=color,
                        stroke_width=6,
                        max_tip_length_to_length_ratio=0.15,
                    )
                    label_mob = MathTex(label, color=color).scale(0.9)
                    label_mob.next_to(
                        arrow.get_end(), _label_direction(vec[0], vec[1]), buff=0.2
                    )
                    inner_self.play(GrowArrow(arrow), Write(label_mob), run_time=0.8)

                inner_self.wait(1.5)

        self._configure(name)
        _PlotScene().render()

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

        class _AdditionScene(_VectorScene):
            def construct(inner_self) -> None:
                v1_coords = (v1[0], v1[1])
                v2_coords = (v2[0], v2[1])
                result_coords = (v1[0] + v2[0], v1[1] + v2[1])

                max_coord = max(
                    abs(c) for c in (*v1_coords, *v2_coords, *result_coords)
                )
                range_val = int(max_coord * 1.4) + 1

                inner_self.setup_axes(
                    x_range=(-range_val, range_val),
                    y_range=(-range_val, range_val),
                )

                arrow_v1 = Arrow(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*v1_coords),
                    buff=0,
                    color=color_v1,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.15,
                )
                label_v1 = MathTex(r"v_1", color=color_v1).scale(0.9)
                label_v1.next_to(
                    arrow_v1.get_end(),
                    _label_direction(v1_coords[0], v1_coords[1]),
                    buff=0.2,
                )

                inner_self.play(GrowArrow(arrow_v1), Write(label_v1), run_time=1)
                inner_self.wait(0.5)

                arrow_v2_origin = Arrow(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*v2_coords),
                    buff=0,
                    color=color_v2,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.15,
                ).set_opacity(0.3)

                label_v2_origin = MathTex(r"v_2", color=color_v2).scale(0.9)
                label_v2_origin.next_to(
                    arrow_v2_origin.get_end(),
                    _label_direction(v2_coords[0], v2_coords[1]),
                    buff=0.2,
                )
                label_v2_origin.set_opacity(0.3)

                inner_self.play(
                    GrowArrow(arrow_v2_origin),
                    Write(label_v2_origin),
                    run_time=1,
                )
                inner_self.wait(0.5)

                arrow_v2_shifted = Arrow(
                    inner_self.axes.c2p(*v1_coords),
                    inner_self.axes.c2p(*result_coords),
                    buff=0,
                    color=color_v2,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.15,
                )

                label_v2_shifted = MathTex(r"v_2", color=color_v2).scale(0.9)
                label_v2_shifted.next_to(
                    arrow_v2_shifted.get_end(),
                    _label_direction(v2_coords[0], v2_coords[1]),
                    buff=0.2,
                )

                inner_self.play(
                    GrowArrow(arrow_v2_shifted),
                    Write(label_v2_shifted),
                    run_time=1.2,
                )
                inner_self.wait(0.5)

                dashed_line_1 = DashedLine(
                    inner_self.axes.c2p(*v2_coords),
                    inner_self.axes.c2p(*result_coords),
                    color=manim.GREY,
                    stroke_width=2,
                    dash_length=0.1,
                )
                dashed_line_2 = DashedLine(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*v2_coords),
                    color=manim.GREY,
                    stroke_width=2,
                    dash_length=0.1,
                )

                parallelogram = Polygon(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*v1_coords),
                    inner_self.axes.c2p(*result_coords),
                    inner_self.axes.c2p(*v2_coords),
                    color=manim.BLUE_E,
                    fill_opacity=0.15,
                    stroke_width=0,
                )

                inner_self.play(
                    Create(dashed_line_1),
                    Create(dashed_line_2),
                    FadeIn(parallelogram),
                    run_time=1,
                )
                inner_self.wait(0.5)

                arrow_result = Arrow(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*result_coords),
                    buff=0,
                    color=color_result,
                    stroke_width=8,
                    max_tip_length_to_length_ratio=0.15,
                )
                label_result = MathTex(r"v_1 + v_2", color=color_result).scale(1.0)
                label_result.next_to(
                    arrow_result.get_end(),
                    _label_direction(result_coords[0], result_coords[1]),
                    buff=0.2,
                )

                inner_self.play(
                    GrowArrow(arrow_result), Write(label_result), run_time=1.5
                )
                inner_self.wait(2)

        self._configure(name)
        _AdditionScene().render()

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

        class _ScalingScene(_VectorScene):
            def construct(inner_self) -> None:
                v_coords = (vector[0], vector[1])
                scaled_coords = (
                    vector[0] * scale_factor,
                    vector[1] * scale_factor,
                )

                max_coord = max(abs(c) for c in (*v_coords, *scaled_coords))
                range_val = int(max_coord * 1.4) + 1

                inner_self.setup_axes(
                    x_range=(-range_val, range_val),
                    y_range=(-range_val, range_val),
                )

                arrow = Arrow(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*v_coords),
                    buff=0,
                    color=color_start,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.15,
                )
                label = MathTex(r"v", color=color_start).scale(0.9)
                label.next_to(
                    arrow.get_end(),
                    _label_direction(v_coords[0], v_coords[1]),
                    buff=0.2,
                )

                inner_self.play(GrowArrow(arrow), Write(label), run_time=1)
                inner_self.wait(0.5)

                scaled_arrow = Arrow(
                    inner_self.axes.c2p(0, 0),
                    inner_self.axes.c2p(*scaled_coords),
                    buff=0,
                    color=color_end,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.15,
                )

                scale_text = MathTex(
                    f"{scale_factor}", r"\cdot", r"v", color=color_end
                ).scale(0.9)
                scale_text.next_to(
                    scaled_arrow.get_end(),
                    _label_direction(scaled_coords[0], scaled_coords[1]),
                    buff=0.2,
                )

                inner_self.play(
                    Transform(arrow, scaled_arrow),
                    Transform(label, scale_text),
                    run_time=2,
                )
                inner_self.wait(1.5)

        self._configure(name)
        _ScalingScene().render()

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

        class _TransformScene(Scene):
            def construct(inner_self) -> None:
                plane = NumberPlane(
                    x_range=[-5, 5, 1],
                    y_range=[-5, 5, 1],
                    background_line_style={
                        "stroke_color": manim.GREY,
                        "stroke_width": 1,
                        "stroke_opacity": 0.6,
                    },
                )
                inner_self.play(Create(plane), run_time=1)

                matrix_tex = MathTex(
                    r"\begin{bmatrix}"
                    f" {mat[0][0]:.3g} & {mat[0][1]:.3g} "
                    r"\\"
                    f" {mat[1][0]:.3g} & {mat[1][1]:.3g} "
                    r"\end{bmatrix}",
                    color=manim.YELLOW,
                ).scale(0.9)
                matrix_tex.to_corner(manim.UR, buff=0.5)
                inner_self.play(Write(matrix_tex), run_time=0.8)

                arrow_e1 = Arrow(
                    plane.c2p(0, 0),
                    plane.c2p(1, 0),
                    buff=0,
                    color=color_e1,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.2,
                )
                arrow_e2 = Arrow(
                    plane.c2p(0, 0),
                    plane.c2p(0, 1),
                    buff=0,
                    color=color_e2,
                    stroke_width=7,
                    max_tip_length_to_length_ratio=0.2,
                )

                label_e1 = MathTex(r"\hat{e}_1", color=color_e1).scale(0.8)
                label_e1.next_to(arrow_e1.get_end(), manim.DOWN + manim.RIGHT, buff=0.2)

                label_e2 = MathTex(r"\hat{e}_2", color=color_e2).scale(0.8)
                label_e2.next_to(arrow_e2.get_end(), manim.UP + manim.LEFT, buff=0.2)

                inner_self.play(
                    GrowArrow(arrow_e1),
                    GrowArrow(arrow_e2),
                    Write(label_e1),
                    Write(label_e2),
                    run_time=1,
                )
                inner_self.wait(0.5)

                label_e1.add_updater(
                    lambda m: m.next_to(
                        arrow_e1.get_end(), manim.DOWN + manim.RIGHT, buff=0.2
                    )
                )
                label_e2.add_updater(
                    lambda m: m.next_to(
                        arrow_e2.get_end(), manim.UP + manim.LEFT, buff=0.2
                    )
                )

                inner_self.play(
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
                inner_self.wait(2)

        self._configure(name)
        _TransformScene().render()

    def plot_span(
        self,
        vectors: list[Vector],
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

        class _SpanScene(_VectorScene):
            def construct(inner_self) -> None:
                if dim == 0:
                    inner_self.setup_axes(x_range=(-3, 3), y_range=(-3, 3))
                    dot = manim.Dot(
                        inner_self.axes.c2p(0, 0), color=manim.WHITE, radius=0.08
                    )
                    inner_self.play(Create(dot))
                    inner_self.wait(1)
                    return

                max_coord = max(abs(c) for v in basis for c in (v[0], v[1]))
                range_val = int(max_coord * 1.5) + 1

                inner_self.setup_axes(
                    x_range=(-range_val, range_val),
                    y_range=(-range_val, range_val),
                )

                if dim == 1:
                    bx, by = float(basis[0][0]), float(basis[0][1])
                    extent = range_val * 2
                    scale = (
                        extent / max(abs(bx), abs(by))
                        if max(abs(bx), abs(by)) > 1e-12
                        else 1
                    )
                    span_line = Line(
                        inner_self.axes.c2p(-bx * scale, -by * scale),
                        inner_self.axes.c2p(bx * scale, by * scale),
                        color=span_color,
                        stroke_width=3,
                        stroke_opacity=0.5,
                    )
                    inner_self.play(Create(span_line), run_time=0.8)

                elif dim == 2:
                    rect = Rectangle(
                        width=range_val * 2 * inner_self.axes.x_axis.get_unit_size(),
                        height=range_val * 2 * inner_self.axes.y_axis.get_unit_size(),
                        color=span_color,
                        fill_opacity=0.1,
                        stroke_width=0,
                    )
                    rect.move_to(inner_self.axes.c2p(0, 0))
                    inner_self.play(FadeIn(rect), run_time=0.8)

                for vec, color, label in zip(
                    basis, color_list, label_list, strict=False
                ):
                    arrow = Arrow(
                        inner_self.axes.c2p(0, 0),
                        inner_self.axes.c2p(vec[0], vec[1]),
                        buff=0,
                        color=color,
                        stroke_width=6,
                        max_tip_length_to_length_ratio=0.15,
                    )
                    label_mob = MathTex(label, color=color).scale(0.9)
                    label_mob.next_to(
                        arrow.get_end(),
                        _label_direction(vec[0], vec[1]),
                        buff=0.2,
                    )
                    inner_self.play(GrowArrow(arrow), Write(label_mob), run_time=0.8)

                inner_self.wait(1.5)

        self._configure(name)
        _SpanScene().render()
