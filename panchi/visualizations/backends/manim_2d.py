from __future__ import annotations
from pathlib import Path

try:
    import manim
    from manim import (
        Scene,
        Axes,
        Arrow,
        MathTex,
        Create,
        Write,
        GrowArrow,
        Transform,
        FadeIn,
        DashedLine,
        Polygon,
    )

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    manim = None


if MANIM_AVAILABLE:
    DEFAULT_COLORS = [manim.RED, manim.ORANGE, manim.GREEN, manim.BLUE, manim.PURPLE]
else:
    DEFAULT_COLORS = []


class VectorScene(Scene):
    """Base scene for vector visualizations with consistent styling."""

    def setup_axes(self, x_range=(-5, 5), y_range=(-5, 5)):
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


class PlotVectorsScene(VectorScene):
    """Scene for plotting multiple static vectors."""

    def __init__(self, vectors, colors=None, labels=None, **kwargs):
        super().__init__(**kwargs)
        self.vectors_data = vectors
        self.colors_data = colors or DEFAULT_COLORS[: len(vectors)]
        self.labels_data = labels or [f"v_{{{i+1}}}" for i in range(len(vectors))]

    def construct(self):
        all_coords = [(v[0], v[1]) for v in self.vectors_data]
        max_coord = max(abs(coord) for vec in all_coords for coord in vec)
        range_val = int(max_coord * 1.3) + 1

        self.setup_axes(
            x_range=(-range_val, range_val), y_range=(-range_val, range_val)
        )

        vector_mobjects = []
        label_mobjects = []

        for vec, color, label in zip(
            self.vectors_data, self.colors_data, self.labels_data
        ):
            arrow = Arrow(
                self.axes.c2p(0, 0),
                self.axes.c2p(vec[0], vec[1]),
                buff=0,
                color=color,
                stroke_width=6,
                max_tip_length_to_length_ratio=0.15,
            )

            label_mob = MathTex(label, color=color).scale(0.9)
            label_mob.next_to(arrow.get_end(), manim.UP + manim.RIGHT, buff=0.1)

            vector_mobjects.append(arrow)
            label_mobjects.append(label_mob)

        for arrow, label in zip(vector_mobjects, label_mobjects):
            self.play(GrowArrow(arrow), Write(label), run_time=0.8)

        self.wait(1.5)


class VectorAdditionScene(VectorScene):
    """Scene for animating vector addition."""

    def __init__(self, v1, v2, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.v2 = v2

    def construct(self):
        v1_coords = (self.v1[0], self.v1[1])
        v2_coords = (self.v2[0], self.v2[1])
        result_coords = (self.v1[0] + self.v2[0], self.v1[1] + self.v2[1])

        max_coord = max(abs(c) for c in (*v1_coords, *v2_coords, *result_coords))
        range_val = int(max_coord * 1.4) + 1

        self.setup_axes(
            x_range=(-range_val, range_val), y_range=(-range_val, range_val)
        )

        arrow_v1 = Arrow(
            self.axes.c2p(0, 0),
            self.axes.c2p(*v1_coords),
            buff=0,
            color=manim.RED,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )
        label_v1 = MathTex(r"v_1", color=manim.RED).scale(0.9)
        label_v1.next_to(arrow_v1.get_center(), manim.UP + manim.LEFT, buff=0.15)

        self.play(GrowArrow(arrow_v1), Write(label_v1), run_time=1)
        self.wait(0.5)

        arrow_v2_origin = Arrow(
            self.axes.c2p(0, 0),
            self.axes.c2p(*v2_coords),
            buff=0,
            color=manim.ORANGE,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        ).set_opacity(0.3)

        label_v2_origin = MathTex(r"v_2", color=manim.ORANGE).scale(0.9)
        label_v2_origin.next_to(
            arrow_v2_origin.get_center(), manim.DOWN + manim.RIGHT, buff=0.15
        )
        label_v2_origin.set_opacity(0.3)

        self.play(GrowArrow(arrow_v2_origin), Write(label_v2_origin), run_time=1)
        self.wait(0.5)

        arrow_v2_shifted = Arrow(
            self.axes.c2p(*v1_coords),
            self.axes.c2p(*result_coords),
            buff=0,
            color=manim.ORANGE,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )

        label_v2_shifted = MathTex(r"v_2", color=manim.ORANGE).scale(0.9)
        label_v2_shifted.next_to(
            arrow_v2_shifted.get_center(), manim.UP + manim.RIGHT, buff=0.15
        )

        self.play(GrowArrow(arrow_v2_shifted), Write(label_v2_shifted), run_time=1.2)
        self.wait(0.5)

        dashed_line_1 = DashedLine(
            self.axes.c2p(*v2_coords),
            self.axes.c2p(*result_coords),
            color=manim.GREY,
            stroke_width=2,
            dash_length=0.1,
        )

        dashed_line_2 = DashedLine(
            self.axes.c2p(0, 0),
            self.axes.c2p(*v2_coords),
            color=manim.GREY,
            stroke_width=2,
            dash_length=0.1,
        )

        parallelogram = Polygon(
            self.axes.c2p(0, 0),
            self.axes.c2p(*v1_coords),
            self.axes.c2p(*result_coords),
            self.axes.c2p(*v2_coords),
            color=manim.BLUE_E,
            fill_opacity=0.15,
            stroke_width=0,
        )

        self.play(
            Create(dashed_line_1),
            Create(dashed_line_2),
            FadeIn(parallelogram),
            run_time=1,
        )
        self.wait(0.5)

        arrow_result = Arrow(
            self.axes.c2p(0, 0),
            self.axes.c2p(*result_coords),
            buff=0,
            color=manim.GREEN,
            stroke_width=8,
            max_tip_length_to_length_ratio=0.15,
        )

        label_result = MathTex(r"v_1 + v_2", color=manim.GREEN).scale(1.0)
        label_result.next_to(arrow_result.get_center(), manim.LEFT, buff=0.2)

        self.play(GrowArrow(arrow_result), Write(label_result), run_time=1.5)
        self.wait(2)


class VectorScalingScene(VectorScene):
    """Scene for animating vector scaling."""

    def __init__(self, vector, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.vector = vector
        self.scale_factor = scale_factor

    def construct(self):
        v_coords = (self.vector[0], self.vector[1])
        scaled_coords = (
            self.vector[0] * self.scale_factor,
            self.vector[1] * self.scale_factor,
        )

        max_coord = max(abs(c) for c in (*v_coords, *scaled_coords))
        range_val = int(max_coord * 1.4) + 1

        self.setup_axes(
            x_range=(-range_val, range_val), y_range=(-range_val, range_val)
        )

        arrow = Arrow(
            self.axes.c2p(0, 0),
            self.axes.c2p(*v_coords),
            buff=0,
            color=manim.RED,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )

        label = MathTex(r"v", color=manim.RED).scale(0.9)
        label.next_to(arrow.get_end(), manim.UP + manim.RIGHT, buff=0.1)

        self.play(GrowArrow(arrow), Write(label), run_time=1)
        self.wait(0.5)

        scaled_arrow = Arrow(
            self.axes.c2p(0, 0),
            self.axes.c2p(*scaled_coords),
            buff=0,
            color=manim.BLUE,
            stroke_width=7,
            max_tip_length_to_length_ratio=0.15,
        )

        scale_text = MathTex(
            f"{self.scale_factor}", r"\cdot", r"v", color=manim.BLUE
        ).scale(0.9)
        scale_text.next_to(scaled_arrow.get_end(), manim.UP + manim.RIGHT, buff=0.1)

        self.play(
            Transform(arrow, scaled_arrow), Transform(label, scale_text), run_time=2
        )
        self.wait(1.5)


def plot_vectors_static(
    vectors, colors=None, labels=None, quality="medium_quality", output_path=None
):
    """
    Plot multiple 2D vectors using Manim.

    Parameters
    ----------
    vectors : list[Vector]
        List of 2D vectors to plot.
    colors : list, optional
        Manim colors for each vector.
    labels : list[str], optional
        LaTeX labels for each vector.
    quality : str, optional
        Render quality.
    output_path : Path or str, optional
        Path to save the video.

    Returns
    -------
    Path
        Path to the rendered video.
    """
    if not MANIM_AVAILABLE:
        raise ImportError("Manim is not installed. Use backend='matplotlib' instead.")

    manim.config.quality = quality
    if output_path:
        manim.config.output_file = str(output_path)

    scene = PlotVectorsScene(vectors=vectors, colors=colors, labels=labels)
    scene.render()

    return Path(manim.config.get_dir("video_dir")) / f"{PlotVectorsScene.__name__}.mp4"


def animate_addition(v1, v2, quality="medium_quality", output_path=None):
    """
    Animate vector addition using Manim.

    Parameters
    ----------
    v1 : Vector
        First vector.
    v2 : Vector
        Second vector.
    quality : str, optional
        Render quality.
    output_path : Path or str, optional
        Path to save the video.

    Returns
    -------
    Path
        Path to the rendered video.
    """
    if not MANIM_AVAILABLE:
        raise ImportError("Manim is not installed. Use backend='matplotlib' instead.")

    manim.config.quality = quality
    if output_path:
        manim.config.output_file = str(output_path)

    scene = VectorAdditionScene(v1=v1, v2=v2)
    scene.render()

    return (
        Path(manim.config.get_dir("video_dir")) / f"{VectorAdditionScene.__name__}.mp4"
    )


def animate_scaling(vector, scale_factor, quality="medium_quality", output_path=None):
    """
    Animate vector scaling using Manim.

    Parameters
    ----------
    vector : Vector
        Vector to scale.
    scale_factor : float
        Scaling factor.
    quality : str, optional
        Render quality.
    output_path : Path or str, optional
        Path to save the video.

    Returns
    -------
    Path
        Path to the rendered video.
    """
    if not MANIM_AVAILABLE:
        raise ImportError("Manim is not installed. Use backend='matplotlib' instead.")

    manim.config.quality = quality
    if output_path:
        manim.config.output_file = str(output_path)

    scene = VectorScalingScene(vector=vector, scale_factor=scale_factor)
    scene.render()

    return (
        Path(manim.config.get_dir("video_dir")) / f"{VectorScalingScene.__name__}.mp4"
    )
