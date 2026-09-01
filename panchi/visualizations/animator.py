from __future__ import annotations

from pathlib import Path

from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace


def _build_backend(
    backend: str,
    save_path: Path | None,
    quality: str,
    figsize: tuple[int, int],
    include_extra_files: bool,
):
    """Construct a 2D backend from the facade's constructor arguments."""
    if backend == "matplotlib":
        from panchi.visualizations.backends.matplotlib_2d import _MatplotlibBackend2D

        return _MatplotlibBackend2D(
            save_path=save_path,
            quality=quality,
            figsize=figsize,
        )
    if backend == "manim":
        from panchi.visualizations.backends.manim_2d import _ManimBackend2D

        return _ManimBackend2D(
            save_path=save_path or Path("./media"),
            quality=quality,
            include_extra_files=include_extra_files,
        )
    raise ValueError(f"Unknown backend '{backend}'. Choose 'matplotlib' or 'manim'.")


def _resolve_single_span(
    item: list[Vector] | VectorSpace,
) -> tuple[list[Vector], VectorSpace]:
    """Resolve one span (a ``VectorSpace`` or a list of ``Vector``) to a pair."""
    if isinstance(item, VectorSpace):
        return list(item.data), item
    space = VectorSpace(item)
    return list(space.data), space


def _resolve_span_inputs(
    arg: list[Vector] | VectorSpace | list[list[Vector] | VectorSpace],
) -> list[tuple[list[Vector], VectorSpace]]:
    """Resolve a ``plot_span`` argument into a list of ``(vectors, VectorSpace)``.

    Accepts a single span (a ``VectorSpace`` or a flat list of ``Vector``) or
    several spans (a list whose items are each a ``VectorSpace`` or a list of
    ``Vector``). A single span yields a one-element list.
    """
    if isinstance(arg, VectorSpace):
        return [_resolve_single_span(arg)]

    if not isinstance(arg, list):
        raise TypeError(
            f"plot_span expects a list of vectors, a VectorSpace, or a list of "
            f"those. Got {type(arg).__name__}."
        )

    if len(arg) == 0:
        raise ValueError("plot_span requires at least one vector or span.")

    if all(isinstance(item, (VectorSpace, list)) for item in arg):
        return [_resolve_single_span(item) for item in arg]

    return [_resolve_single_span(arg)]


class Animator2D:
    """2D visualization of vectors, matrices, and linear transformations.

    Parameters
    ----------
    backend : str
        Visualization backend: ``"matplotlib"`` (default) or ``"manim"``.
    save_path : str or Path, optional
        Directory for saving output. If ``None``, matplotlib displays
        interactively and manim saves to ``./media``.
    quality : str
        Render quality: ``"low"``, ``"medium"`` (default), ``"high"``,
        or ``"production"`` (manim only).
    figsize : tuple[int, int]
        Figure size in inches (matplotlib only). Default ``(8, 8)``.

    Examples
    --------
    >>> from panchi import Vector, Matrix
    >>> from panchi.visualizations import Animator2D
    >>>
    >>> animator = Animator2D()
    >>> animator.plot_vectors([Vector([3, 2]), Vector([1, 3])])
    >>> animator.animate_transform(Matrix([[0, -1], [1, 0]]))
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        save_path: str | Path | None = None,
        quality: str = "medium",
        figsize: tuple[int, int] = (8, 8),
        include_extra_files: bool = False,
    ) -> None:
        resolved_path = Path(save_path) if save_path else None
        self._backend = _build_backend(
            backend, resolved_path, quality, figsize, include_extra_files
        )
        self.backend = backend

    def plot_vectors(
        self,
        vectors: list[Vector],
        *,
        colors: list[str] | None = None,
        labels: list[str] | None = None,
        grid: bool = True,
        name: str | None = None,
    ) -> None:
        """Plot a list of 2D vectors.

        Parameters
        ----------
        vectors : list[Vector]
            2D vectors to plot.
        colors : list[str], optional
            Colors for each vector.
        labels : list[str], optional
            Labels for each vector.
        grid : bool
            Whether to show grid lines. Default ``True``.
        name : str, optional
            Output filename (without extension) when saving.
        """
        self._validate_2d(vectors)
        self._backend.plot_vectors(
            vectors,
            colors=colors,
            labels=labels,
            grid=grid,
            name=name or "plot_vectors",
        )

    def animate_addition(
        self,
        v1: Vector,
        v2: Vector,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate the addition of two 2D vectors.

        Parameters
        ----------
        v1 : Vector
            First vector.
        v2 : Vector
            Second vector.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            Up to three colors for ``[v1, v2, v1 + v2]``. Any omitted role
            keeps its default. ``None`` uses the default palette.
        """
        self._validate_2d([v1, v2])
        return self._backend.animate_addition(
            v1,
            v2,
            frames=frames,
            interval=interval,
            name=name or "animate_addition",
            colors=colors,
        )

    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate scalar multiplication of a 2D vector.

        Parameters
        ----------
        vector : Vector
            Vector to scale.
        scale_factor : float
            Scaling factor.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            Up to two colors for ``[original, scaled]``. Any omitted role
            keeps its default. ``None`` uses the default palette.
        """
        self._validate_2d([vector])
        return self._backend.animate_scaling(
            vector,
            scale_factor,
            frames=frames,
            interval=interval,
            name=name or "animate_scaling",
            colors=colors,
        )

    def animate_transform(
        self,
        matrix: Matrix,
        vectors: list[Vector] | None = None,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate a 2x2 linear transformation with full grid deformation.

        The coordinate grid morphs smoothly from the identity to the given
        matrix. Each vector in ``vectors`` slides from ``v`` to ``matrix @ v``;
        when ``vectors`` is ``None`` the standard basis ``[e1, e2]`` is shown.

        Parameters
        ----------
        matrix : Matrix
            A 2x2 transformation matrix.
        vectors : list[Vector], optional
            2D vectors to transform. ``None`` uses the standard basis.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            One color per transformed vector, cycling if fewer are supplied.
            ``None`` uses the default palette.
        """
        self._validate_2x2(matrix)
        if vectors is not None:
            self._validate_2d(vectors)
        return self._backend.animate_transform(
            matrix,
            vectors=vectors,
            frames=frames,
            interval=interval,
            name=name or "animate_transform",
            colors=colors,
        )

    def plot_span(
        self,
        vectors_or_space: list[Vector] | VectorSpace | list[list[Vector] | VectorSpace],
        *,
        colors: list[str] | None = None,
        labels: list[str] | None = None,
        grid: bool = True,
        name: str | None = None,
        span_color: str | None = None,
    ) -> None:
        """Visualize the span of vectors with a shaded region and basis arrows.

        Accepts a single span (a ``VectorSpace`` or a list of ``Vector``) or
        several spans to compare (a list whose items are each a ``VectorSpace``
        or a list of ``Vector``). A 1D subspace is drawn as a line through the
        origin; a 2D subspace shades the entire plane. With multiple spans each
        is drawn in its own color with a legend.

        Parameters
        ----------
        vectors_or_space : list[Vector], VectorSpace, or list of those
            One span, or a list of spans to compare.
        colors : list[str], optional
            For a single span, colors for the basis vectors; for multiple
            spans, one color per span.
        labels : list[str], optional
            For a single span, labels for the basis vectors; for multiple
            spans, one legend label per span.
        grid : bool
            Whether to show grid lines. Default ``True``.
        name : str, optional
            Output filename (without extension) when saving.
        span_color : str, optional
            Color of the shaded region (single span only). ``None`` uses the
            default.
        """
        spans = _resolve_span_inputs(vectors_or_space)
        for vectors, _ in spans:
            self._validate_2d(vectors)
        self._backend.plot_span(
            [space for _, space in spans],
            colors=colors,
            labels=labels,
            grid=grid,
            name=name or "plot_span",
            span_color=span_color,
        )

    @staticmethod
    def _validate_2d(vectors: list[Vector]) -> None:
        if not isinstance(vectors, list):
            raise TypeError(
                f"Vectors must be passed as a list. Wrap a single vector as [v]. "
                f"Got {type(vectors).__name__}."
            )
        for v in vectors:
            if v.dims != 2:
                raise ValueError(
                    f"Only 2D vectors are supported for visualization. "
                    f"Got {v.dims}D vector: {v}"
                )

    @staticmethod
    def _validate_2x2(matrix: Matrix) -> None:
        if matrix.shape != (2, 2):
            raise ValueError(
                f"Only 2x2 matrices are supported for transformation "
                f"visualization. Got shape {matrix.rows}x{matrix.cols}."
            )


class Animator3D:
    """3D visualization of vectors in R³.

    Mirrors :class:`Animator2D` for three-dimensional vectors.

    Parameters
    ----------
    backend : str
        Visualization backend: ``"matplotlib"`` (default) or ``"manim"``.
    save_path : str or Path, optional
        Directory for saving output. If ``None``, matplotlib displays
        interactively and manim saves to ``./media``.
    quality : str
        Render quality: ``"low"``, ``"medium"`` (default), or ``"high"``.
    figsize : tuple[int, int]
        Figure size in inches. Default ``(8, 8)``.

    Examples
    --------
    >>> from panchi import Vector
    >>> from panchi.visualizations import Animator3D
    >>>
    >>> animator = Animator3D()
    >>> animator.plot_vectors([Vector([3, 2, 1]), Vector([1, 3, 2])])
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        save_path: str | Path | None = None,
        quality: str = "medium",
        figsize: tuple[int, int] = (8, 8),
        include_extra_files: bool = False,
    ) -> None:
        resolved_path = Path(save_path) if save_path else None

        if backend == "matplotlib":
            from panchi.visualizations.backends.matplotlib_3d import (
                _MatplotlibBackend3D,
            )

            self._backend = _MatplotlibBackend3D(
                save_path=resolved_path,
                quality=quality,
                figsize=figsize,
            )
        elif backend == "manim":
            from panchi.visualizations.backends.manim_3d import _ManimBackend3D

            self._backend = _ManimBackend3D(
                save_path=resolved_path or Path("./media"),
                quality=quality,
                include_extra_files=include_extra_files,
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'matplotlib' or 'manim'."
            )

        self.backend = backend

    def plot_vectors(
        self,
        vectors: list[Vector],
        *,
        colors: list[str] | None = None,
        labels: list[str] | None = None,
        grid: bool = True,
        name: str | None = None,
    ) -> None:
        """Plot a list of 3D vectors.

        Parameters
        ----------
        vectors : list[Vector]
            3D vectors to plot.
        colors : list[str], optional
            Colors for each vector.
        labels : list[str], optional
            Labels for each vector.
        grid : bool
            Whether to show grid lines. Default ``True``.
        name : str, optional
            Output filename (without extension) when saving.
        """
        self._validate_3d(vectors)
        self._backend.plot_vectors(
            vectors,
            colors=colors,
            labels=labels,
            grid=grid,
            name=name or "plot_vectors",
        )

    def animate_addition(
        self,
        v1: Vector,
        v2: Vector,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate the addition of two 3D vectors.

        Parameters
        ----------
        v1 : Vector
            First vector.
        v2 : Vector
            Second vector.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            Up to three colors for ``[v1, v2, v1 + v2]``. Any omitted role
            keeps its default. ``None`` uses the default palette.
        """
        self._validate_3d([v1, v2])
        return self._backend.animate_addition(
            v1,
            v2,
            frames=frames,
            interval=interval,
            name=name or "animate_addition",
            colors=colors,
        )

    def animate_scaling(
        self,
        vector: Vector,
        scale_factor: float,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate scalar multiplication of a 3D vector.

        Parameters
        ----------
        vector : Vector
            Vector to scale.
        scale_factor : float
            Scaling factor.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            Up to two colors for ``[original, scaled]``. Any omitted role
            keeps its default. ``None`` uses the default palette.
        """
        self._validate_3d([vector])
        return self._backend.animate_scaling(
            vector,
            scale_factor,
            frames=frames,
            interval=interval,
            name=name or "animate_scaling",
            colors=colors,
        )

    def animate_transform(
        self,
        matrix: Matrix,
        vectors: list[Vector] | None = None,
        frames: int = 60,
        interval: int = 30,
        name: str | None = None,
        colors: list[str] | None = None,
    ) -> object | None:
        """Animate a 3x3 linear transformation of R³.

        Each vector in ``vectors`` slides smoothly from ``v`` to ``matrix @ v``;
        when ``vectors`` is ``None`` the standard basis ``[e1, e2, e3]`` is
        shown (its image is the parallelepiped spanned by the matrix columns).

        Parameters
        ----------
        matrix : Matrix
            A 3x3 transformation matrix.
        vectors : list[Vector], optional
            3D vectors to transform. ``None`` uses the standard basis.
        frames : int
            Number of animation frames. Default ``60``.
        interval : int
            Milliseconds between frames. Default ``30``.
        name : str, optional
            Output filename (without extension) when saving.
        colors : list[str], optional
            One color per transformed vector, cycling if fewer are supplied.
            ``None`` uses the default palette.
        """
        self._validate_3x3(matrix)
        if vectors is not None:
            self._validate_3d(vectors)
        return self._backend.animate_transform(
            matrix,
            vectors=vectors,
            frames=frames,
            interval=interval,
            name=name or "animate_transform",
            colors=colors,
        )

    def plot_span(
        self,
        vectors_or_space: list[Vector] | VectorSpace | list[list[Vector] | VectorSpace],
        *,
        colors: list[str] | None = None,
        labels: list[str] | None = None,
        grid: bool = True,
        name: str | None = None,
        span_color: str | None = None,
    ) -> None:
        """Visualize the span of 3D vectors with a shaded region and basis arrows.

        Accepts a single span (a ``VectorSpace`` or a list of ``Vector``) or
        several spans to compare (a list whose items are each a ``VectorSpace``
        or a list of ``Vector``). A 1D subspace is drawn as a line through the
        origin, a 2D subspace as a plane patch, and a 3D subspace as a
        translucent volume. With multiple spans each is drawn in its own color
        with a legend.

        Parameters
        ----------
        vectors_or_space : list[Vector], VectorSpace, or list of those
            One span, or a list of spans to compare.
        colors : list[str], optional
            For a single span, colors for the basis vectors; for multiple
            spans, one color per span.
        labels : list[str], optional
            For a single span, labels for the basis vectors; for multiple
            spans, one legend label per span.
        grid : bool
            Whether to show grid lines. Default ``True``.
        name : str, optional
            Output filename (without extension) when saving.
        span_color : str, optional
            Color of the shaded region (single span only). ``None`` uses the
            default.
        """
        spans = _resolve_span_inputs(vectors_or_space)
        for vectors, _ in spans:
            self._validate_3d(vectors)
        self._backend.plot_span(
            [space for _, space in spans],
            colors=colors,
            labels=labels,
            grid=grid,
            name=name or "plot_span",
            span_color=span_color,
        )

    @staticmethod
    def _validate_3d(vectors: list[Vector]) -> None:
        if not isinstance(vectors, list):
            raise TypeError(
                f"Vectors must be passed as a list. Wrap a single vector as [v]. "
                f"Got {type(vectors).__name__}."
            )
        for v in vectors:
            if v.dims != 3:
                raise ValueError(
                    f"Only 3D vectors are supported for Animator3D. "
                    f"Got {v.dims}D vector: {v}"
                )

    @staticmethod
    def _validate_3x3(matrix: Matrix) -> None:
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Only 3x3 matrices are supported for transformation "
                f"visualization. Got shape {matrix.rows}x{matrix.cols}."
            )
