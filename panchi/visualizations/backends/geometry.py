"""Backend-agnostic 3D geometry helpers shared by the matplotlib and manim
3D backends."""

from __future__ import annotations

from collections.abc import Sequence
from math import asin, atan2, cos, degrees, radians, sin

from panchi.primitives.vector import Vector

Point = Sequence[float]

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


def coordinates(vector: Vector) -> list[float]:
    """Return the components of ``vector`` as plain floats."""
    return [float(component) for component in vector.data]


def _dot(u: Point, v: Point) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _cross(u: Point, v: Point) -> tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _unit(v: Point) -> tuple[float, ...]:
    length = _dot(v, v) ** 0.5
    return tuple(c / length for c in v)


def plane_in_cube(u: Point, v: Point, half_size: float) -> list[tuple[float, ...]]:
    """Return the polygon where the plane spanned by ``u, v`` meets a cube.

    The cube is centered on the origin with side ``2 * half_size``. Filling the
    cube, rather than scaling ``u`` and ``v``, keeps the plane's shape the same
    however close to parallel the two spanning vectors are.
    """
    normal = _unit(_cross(u, v))
    corners = [tuple((2 * c - 1) * half_size for c in vert) for vert in CUBE_VERTS]

    crossings = []
    for i, j in CUBE_EDGES:
        a, b = corners[i], corners[j]
        side_a, side_b = _dot(normal, a), _dot(normal, b)
        if side_a * side_b > 0 or side_a == side_b:
            continue
        t = side_a / (side_a - side_b)
        crossings.append(tuple(p + t * (q - p) for p, q in zip(a, b, strict=True)))

    across = _unit(u)
    up = _cross(normal, across)
    crossings.sort(key=lambda p: atan2(_dot(p, up), _dot(p, across)))
    return crossings


def edge_view(
    normal: Point, along: Point, height: float = 15.0
) -> tuple[float, float, float]:
    """Return ``(elevation, azimuth, roll)`` looking across a plane, in degrees.

    The camera sits ``height`` degrees above the plane, on the side ``normal``
    points to, and is rolled so the plane reads as the horizon with ``along``
    pointing right. Anything off the plane then rises straight up the screen.
    """
    normal = _unit(normal)
    along = _unit(along)
    sideways = _cross(along, normal)
    tilt = radians(height)
    camera = tuple(
        cos(tilt) * s + sin(tilt) * n for s, n in zip(sideways, normal, strict=True)
    )

    elevation = degrees(asin(max(-1.0, min(1.0, camera[2]))))
    azimuth = degrees(atan2(camera[1], camera[0]))

    az = radians(azimuth)
    el = radians(elevation)
    screen_up = (-sin(el) * cos(az), -sin(el) * sin(az), cos(el))
    screen_right = _cross(screen_up, camera)
    roll = degrees(atan2(_dot(normal, screen_right), _dot(normal, screen_up)))
    return (elevation, azimuth, roll)
