from __future__ import annotations

from manim import (
    DEGREES,
    Arrow3D,
    Create,
    MathTex,
    ThreeDAxes,
    ThreeDScene,
)

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.manim_base import (
    AXIS_COLOR,
    DEFAULT_COLORS,
    _axis_range,
    _BuilderSceneMixin,
    _ManimBackendBase,
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
        for name, end in zip(("x", "y", "z"), ends):
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
        run_time: float = 0.8,
        wait: float = 0.0,
    ) -> tuple[Arrow3D, MathTex | None]:
        """Draw a 3D arrow from the origin with an optional label near its tip.

        The label is a fixed-orientation mobject: it sits just past the arrow
        tip in space but always faces the camera, so it stays legible under the
        tilted 3D view instead of lying flat in the world plane.
        """
        tip = self.axes.c2p(*coords)
        arrow = Arrow3D(self.axes.c2p(0, 0, 0), tip, color=color)
        self.play(Create(arrow), run_time=run_time)

        label_mob = None
        if label is not None:
            label_mob = MathTex(label, color=color).scale(0.9)
            label_mob.move_to(tip * 1.15)
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
