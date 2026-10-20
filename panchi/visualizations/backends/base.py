from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace
from panchi.visualizations.backends.reduction_plan import ReductionPlan

DEFAULT_COLORS = ["#805B49", "#FFB592", "#3E7C7B", "#D4A15E", "#45423F"]

# Per-method default color roles, shared across dimensions. Overridable by
# passing ``colors=`` (animations) or ``span_color=`` (spans) to the method.
ADDITION_COLORS = ("#805B49", "#FFB592", "#3E7C7B")  # v1, v2, result
SCALING_COLORS = ("#805B49", "#FFB592")  # original, scaled
TRANSFORM_COLORS = ("#805B49", "#FFB592")  # e1, e2 (2D)
TRANSFORM_COLORS_3D = ("#805B49", "#FFB592", "#3E7C7B")  # e1, e2, e3 (3D)
SPAN_COLOR = "#805B49"


def _resolve_colors(colors: list[str] | None, defaults: tuple[str, ...]) -> list[str]:
    """Fill in per-role colors, keeping defaults for any not supplied.

    A partial list is honored positionally; missing trailing roles fall back
    to ``defaults``. ``None`` reproduces the default palette exactly.
    """
    colors = colors or []
    return [colors[i] if i < len(colors) else defaults[i] for i in range(len(defaults))]


def _resolve_n_colors(
    colors: list[str] | None, n: int, defaults: tuple[str, ...]
) -> list[str]:
    """Return ``n`` colors, cycling the palette when more are needed than given.

    Supplied colors take precedence; ``defaults`` (then ``DEFAULT_COLORS``) fill
    the rest, cycling so an arbitrary number of vectors each get a color.
    """
    palette = list(colors) if colors else list(defaults) or list(DEFAULT_COLORS)
    return [palette[i % len(palette)] for i in range(n)]


class _BaseBackend(ABC):
    """Abstract base defining the contract for 2D visualization backends."""

    def __init__(self, save_path: Path | None, quality: str) -> None:
        self.save_path = save_path
        self.quality = quality

    @abstractmethod
    def plot_vectors(
        self,
        vectors: list[Vector],
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> object | None: ...

    @abstractmethod
    def animate_addition(
        self,
        v1: Vector,
        v2: Vector,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> object | None: ...

    @abstractmethod
    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> object | None: ...

    @abstractmethod
    def animate_transform(
        self,
        matrix: Matrix,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> object | None: ...

    @abstractmethod
    def plot_span(
        self,
        space: VectorSpace,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
        span_color: str | None = None,
    ) -> object | None: ...

    def animate_reduction(
        self,
        plan: ReductionPlan,
        frames: int,
        interval: int,
        name: str,
        colors: list[str] | None = None,
    ) -> object | None:
        raise NotImplementedError(
            f"animate_reduction is not available for the {type(self).__name__} "
            f"backend yet. Use the matplotlib backend: Animator2D() or Animator3D()."
        )
