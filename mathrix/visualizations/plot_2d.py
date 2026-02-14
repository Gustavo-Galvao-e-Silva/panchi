from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

from mathrix.primitives.vector import Vector

DEFAULT_COLORS = ["#E63946", "#F77F00", "#06FFA5", "#118AB2", "#073B4C"]


def _setup_coordinate_plane(ax, x_range, y_range, grid=True):
    """Configure axes to look like a mathematical coordinate plane."""
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


def _calculate_axis_range(vectors):
    """Calculate appropriate axis range to fit all vectors with padding."""
    all_x = [v[0] for v in vectors]
    all_y = [v[1] for v in vectors]
    max_coord = max(abs(min(all_x + all_y)), abs(max(all_x + all_y))) * 1.2
    return (-max_coord, max_coord)


def _create_vector_arrow(start, end, color, linewidth=2.5):
    """Create a properly styled arrow for vector visualization."""
    return FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_width=0.4,head_length=0.8",
        color=color,
        linewidth=linewidth,
        zorder=3,
        mutation_scale=20,
    )


def _add_vector_label(ax, vector, label, color, offset=(0.15, 0.15)):
    """Add a label near the tip of a vector."""
    offset_x = offset[0] if vector[0] >= 0 else -offset[0]
    offset_y = offset[1] if vector[1] >= 0 else -offset[1]

    ax.text(
        vector[0] + offset_x,
        vector[1] + offset_y,
        label,
        fontsize=12,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
    )


def plot_vectors(*vectors: Vector, colors=None, labels=None, grid=True, figsize=(8, 8)):
    """
    Plot multiple 2D vectors originating from the origin.

    Parameters
    ----------
    *vectors : Vector
        Variable number of 2D vectors to plot.
    colors : list[str], optional
        Colors for each vector. If None, uses a default palette.
    labels : list[str], optional
        Labels for each vector. If None, no labels are shown.
    grid : bool, optional
        Whether to show grid lines. Default is True.
    figsize : tuple[int, int], optional
        Figure size in inches. Default is (8, 8).

    Raises
    ------
    ValueError
        If any vector is not 2-dimensional.
    """
    if not all(v.dims == 2 for v in vectors):
        raise ValueError("All vectors must be 2-dimensional for 2D plotting")

    if colors is None:
        colors = DEFAULT_COLORS[: len(vectors)]

    if labels is None:
        labels = [None] * len(vectors)

    fig, ax = plt.subplots(figsize=figsize)

    x_range, y_range = _calculate_axis_range(vectors), _calculate_axis_range(vectors)
    _setup_coordinate_plane(ax, x_range, y_range, grid)

    for vec, color, label in zip(vectors, colors, labels):
        arrow = _create_vector_arrow((0, 0), (vec[0], vec[1]), color)
        ax.add_patch(arrow)

        if label:
            _add_vector_label(ax, vec, label, color)

    plt.tight_layout()
    plt.show()


def animate_vector_addition(
    v1: Vector, v2: Vector, frames=60, interval=30, figsize=(10, 10)
):
    """
    Animate the addition of two 2D vectors using parallelogram construction.

    The animation shows:
    1. Vector v1 (red) from the origin
    2. Vector v2 (orange) growing from the origin and from the tip of v1
    3. The resultant vector v1 + v2 (green) appearing

    Parameters
    ----------
    v1 : Vector
        First 2D vector.
    v2 : Vector
        Second 2D vector.
    frames : int, optional
        Total number of animation frames. Default is 60.
    interval : int, optional
        Milliseconds between frames. Default is 30.
    figsize : tuple[int, int], optional
        Figure size in inches. Default is (10, 10).

    Returns
    -------
    animation.FuncAnimation
        The matplotlib animation object.

    Raises
    ------
    ValueError
        If either vector is not 2-dimensional.
    """
    if v1.dims != 2 or v2.dims != 2:
        raise ValueError("Both vectors must be 2-dimensional for 2D plotting")

    fig, ax = plt.subplots(figsize=figsize)

    result = v1 + v2
    max_coord = (
        max(
            abs(v1[0]),
            abs(v1[1]),
            abs(v2[0]),
            abs(v2[1]),
            abs(result[0]),
            abs(result[1]),
        )
        * 1.3
    )

    _setup_coordinate_plane(
        ax, (-max_coord, max_coord), (-max_coord, max_coord), grid=True
    )

    arrow_v1 = _create_vector_arrow((0, 0), (v1[0], v1[1]), "#E63946")
    arrow_v2_from_origin = _create_vector_arrow((0, 0), (0, 0), "#F77F00")
    arrow_v2_from_origin.set_alpha(0.4)
    arrow_v2_from_v1 = _create_vector_arrow((v1[0], v1[1]), (v1[0], v1[1]), "#F77F00")
    arrow_result = _create_vector_arrow((0, 0), (0, 0), "#06FFA5", linewidth=3.5)
    arrow_result.set_alpha(0)

    ax.add_patch(arrow_v1)
    ax.add_patch(arrow_v2_from_origin)
    ax.add_patch(arrow_v2_from_v1)
    ax.add_patch(arrow_result)

    text_v1 = ax.text(
        v1[0] / 2 - 0.2,
        v1[1] / 2 + 0.3,
        "v₁",
        fontsize=14,
        fontweight="bold",
        color="#E63946",
    )
    text_v2 = ax.text(
        0, 0, "v₂", fontsize=14, fontweight="bold", color="#F77F00", alpha=0
    )
    text_result = ax.text(
        0, 0, "v₁ + v₂", fontsize=14, fontweight="bold", color="#06FFA5", alpha=0
    )

    half_frames = frames // 2

    def animate(frame):
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

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim


def animate_vector_scaling(
    vector: Vector, scale_factor: float, frames=60, interval=30, figsize=(10, 10)
):
    """
    Animate scalar multiplication of a vector.

    Shows a vector smoothly scaling from its original size to the scaled size.

    Parameters
    ----------
    vector : Vector
        The 2D vector to scale.
    scale_factor : float
        The scaling factor to apply.
    frames : int, optional
        Total number of animation frames. Default is 60.
    interval : int, optional
        Milliseconds between frames. Default is 30.
    figsize : tuple[int, int], optional
        Figure size in inches. Default is (10, 10).

    Returns
    -------
    animation.FuncAnimation
        The matplotlib animation object.

    Raises
    ------
    ValueError
        If vector is not 2-dimensional.
    """
    if vector.dims != 2:
        raise ValueError("Vector must be 2-dimensional for 2D plotting")

    fig, ax = plt.subplots(figsize=figsize)

    scaled = scale_factor * vector
    max_coord = (
        max(abs(vector[0]), abs(vector[1]), abs(scaled[0]), abs(scaled[1])) * 1.3
    )

    _setup_coordinate_plane(
        ax, (-max_coord, max_coord), (-max_coord, max_coord), grid=True
    )

    arrow = _create_vector_arrow((0, 0), (vector[0], vector[1]), "#E63946")
    ax.add_patch(arrow)

    text = ax.text(
        vector[0] + 0.2,
        vector[1] + 0.2,
        "v",
        fontsize=14,
        fontweight="bold",
        color="#E63946",
    )

    def animate(frame):
        t = frame / frames
        current_scale = 1 + (scale_factor - 1) * t
        current_x = vector[0] * current_scale
        current_y = vector[1] * current_scale

        arrow.set_positions((0, 0), (current_x, current_y))

        if t > 0.5:
            alpha = (t - 0.5) * 2
            arrow.set_color("#118AB2")
            text.set_text(f"{scale_factor}v")
            text.set_color("#118AB2")
            text.set_alpha(alpha)

        text.set_position((current_x + 0.2, current_y + 0.2))

        return arrow, text

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim
