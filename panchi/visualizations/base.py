from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace


class _BaseBackend(ABC):
    """Abstract base defining the contract for 2D visualization backends."""

    def __init__(self, save_path: Path | None, quality: str) -> None:
        self.save_path = save_path
        self.quality = quality

    @abstractmethod
    def plot_vectors(
        self,
        *vectors: Vector,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None: ...

    @abstractmethod
    def animate_addition(
        self,
        v1: Vector,
        v2: Vector,
        frames: int,
        interval: int,
        name: str,
    ) -> None: ...

    @abstractmethod
    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int,
        interval: int,
        name: str,
    ) -> None: ...

    @abstractmethod
    def animate_transform(
        self,
        matrix: Matrix,
        frames: int,
        interval: int,
        name: str,
    ) -> None: ...

    @abstractmethod
    def plot_span(
        self,
        vectors: list[Vector],
        space: VectorSpace,
        colors: list[str] | None,
        labels: list[str] | None,
        grid: bool,
        name: str,
    ) -> None: ...
