from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

DEFAULT_COLORS = ["#E63946", "#F77F00", "#06FFA5", "#118AB2", "#073B4C"]


def setup_coordinate_plane(ax, x_range, y_range, grid=True):
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


def calculate_axis_range(vectors):
    """Calculate appropriate axis range to fit all vectors with padding."""
    all_x = [v[0] for v in vectors]
    all_y = [v[1] for v in vectors]
    max_coord = max(abs(min(all_x + all_y)), abs(max(all_x + all_y))) * 1.2
    return (-max_coord, max_coord)


def create_vector_arrow(start, end, color, linewidth=2.5):
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


def add_vector_label(ax, position, label, color, offset=(0.15, 0.15)):
    """Add a label at a specific position."""
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


def plot_vectors_static(vectors, colors=None, labels=None, grid=True, figsize=(8, 8)):
    """
    Plot multiple 2D vectors using matplotlib.

    Parameters
    ----------
    vectors : list[Vector]
        List of 2D vectors to plot.
    colors : list[str], optional
        Colors for each vector.
    labels : list[str], optional
        Labels for each vector.
    grid : bool, optional
        Whether to show grid.
    figsize : tuple[int, int], optional
        Figure size.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    if colors is None:
        colors = DEFAULT_COLORS[: len(vectors)]

    if labels is None:
        labels = [None] * len(vectors)

    fig, ax = plt.subplots(figsize=figsize)

    x_range, y_range = calculate_axis_range(vectors), calculate_axis_range(vectors)
    setup_coordinate_plane(ax, x_range, y_range, grid)

    for vec, color, label in zip(vectors, colors, labels):
        arrow = create_vector_arrow((0, 0), (vec[0], vec[1]), color)
        ax.add_patch(arrow)

        if label:
            offset_x = 0.15 if vec[0] >= 0 else -0.15
            offset_y = 0.15 if vec[1] >= 0 else -0.15
            add_vector_label(ax, (vec[0], vec[1]), label, color, (offset_x, offset_y))

    plt.tight_layout()
    return fig, ax


def animate_addition(v1, v2, frames=60, interval=30, figsize=(10, 10), save_path=None):
    """
    Animate vector addition using matplotlib.

    Parameters
    ----------
    v1 : Vector
        First vector.
    v2 : Vector
        Second vector.
    frames : int, optional
        Number of frames.
    interval : int, optional
        Milliseconds between frames.
    figsize : tuple[int, int], optional
        Figure size.
    save_path : Path or str, optional
        Path to save animation. If None, displays instead.

    Returns
    -------
    animation.FuncAnimation
        The matplotlib animation object.
    """
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

    setup_coordinate_plane(
        ax, (-max_coord, max_coord), (-max_coord, max_coord), grid=True
    )

    arrow_v1 = create_vector_arrow((0, 0), (v1[0], v1[1]), "#E63946")
    arrow_v2_from_origin = create_vector_arrow((0, 0), (0, 0), "#F77F00")
    arrow_v2_from_origin.set_alpha(0.4)
    arrow_v2_from_v1 = create_vector_arrow((v1[0], v1[1]), (v1[0], v1[1]), "#F77F00")
    arrow_result = create_vector_arrow((0, 0), (0, 0), "#06FFA5", linewidth=3.5)
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

    def animate_frame(frame):
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
        fig, animate_frame, frames=frames, interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(str(save_path), writer="pillow", fps=1000 // interval)

    return anim


def animate_scaling(
    vector, scale_factor, frames=60, interval=30, figsize=(10, 10), save_path=None
):
    """
    Animate vector scaling using matplotlib.

    Parameters
    ----------
    vector : Vector
        Vector to scale.
    scale_factor : float
        Scaling factor.
    frames : int, optional
        Number of frames.
    interval : int, optional
        Milliseconds between frames.
    figsize : tuple[int, int], optional
        Figure size.
    save_path : Path or str, optional
        Path to save animation. If None, displays instead.

    Returns
    -------
    animation.FuncAnimation
        The matplotlib animation object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    scaled = scale_factor * vector
    max_coord = (
        max(abs(vector[0]), abs(vector[1]), abs(scaled[0]), abs(scaled[1])) * 1.3
    )

    setup_coordinate_plane(
        ax, (-max_coord, max_coord), (-max_coord, max_coord), grid=True
    )

    arrow = create_vector_arrow((0, 0), (vector[0], vector[1]), "#E63946")
    ax.add_patch(arrow)

    text = ax.text(
        vector[0] + 0.2,
        vector[1] + 0.2,
        "v",
        fontsize=14,
        fontweight="bold",
        color="#E63946",
    )

    def animate_frame(frame):
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
        fig, animate_frame, frames=frames, interval=interval, blit=True, repeat=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(str(save_path), writer="pillow", fps=1000 // interval)

    return anim
