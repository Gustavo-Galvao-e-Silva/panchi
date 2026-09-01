"""Backend-agnostic 3D geometry helpers shared by the matplotlib and manim
3D backends."""

from __future__ import annotations

# Unit cube [0, 1]^3: the 8 vertices and the 12 edges (vertex-index pairs that
# differ in exactly one coordinate).
CUBE_VERTS = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
CUBE_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if sum(a != b for a, b in zip(CUBE_VERTS[i], CUBE_VERTS[j], strict=False)) == 1
]


def apply_3x3(m, p):
    """Apply a 3x3 matrix (list of rows) to a 3-point, returning a tuple."""
    return tuple(sum(m[i][k] * p[k] for k in range(3)) for i in range(3))
