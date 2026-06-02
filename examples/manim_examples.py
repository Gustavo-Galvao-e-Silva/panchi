"""
Generate example videos using the Manim backend.

Renders two high-quality animations:
1. Vector addition: (-2, 4) + (5, -1)
2. Linear transformation: rotation by 30° followed by non-uniform scaling
"""

import math

from panchi import Matrix, Vector
from panchi.visualizations import Animator2D


def main():
    output_dir = "./examples/output"
    animator = Animator2D(backend="manim", save_path=output_dir, quality="high")

    print("Rendering example 1: vector addition (-2, 4) + (5, -1)")
    animator.animate_addition(
        Vector([-2, 4]),
        Vector([5, -1]),
        name="addition_example",
    )
    print("  Done.")

    print("Rendering example 2: rotation (30°) + non-uniform scale (2x, 0.5y)")
    angle = math.radians(30)
    rotation = [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ]
    scale = [[2, 0], [0, 0.5]]
    combined = [
        [
            rotation[0][0] * scale[0][0] + rotation[0][1] * scale[1][0],
            rotation[0][0] * scale[0][1] + rotation[0][1] * scale[1][1],
        ],
        [
            rotation[1][0] * scale[0][0] + rotation[1][1] * scale[1][0],
            rotation[1][0] * scale[0][1] + rotation[1][1] * scale[1][1],
        ],
    ]
    animator.animate_transform(
        Matrix(combined),
        name="rotation_scale_transform",
    )
    print("  Done.")

    print(f"\nAll videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
