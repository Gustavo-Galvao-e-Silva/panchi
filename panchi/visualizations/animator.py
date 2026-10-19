from __future__ import annotations

from pathlib import Path

from panchi.algorithms.vector_space_operations import basis
from panchi.primitives.matrix import Matrix
from panchi.primitives.vector import Vector
from panchi.primitives.vector_space import VectorSpace


def _build_backend(
    backend: str,
    dims: int,
    save_path: Path | None,
    quality: str,
    figsize: tuple[int, int],
    include_extra_files: bool,
    output_format: str | None,
):
    """Construct a 2D or 3D backend from the facade's constructor arguments."""
    if backend == "matplotlib":
        if output_format is not None:
            raise ValueError(
                "output_format only applies to the manim backend. matplotlib "
                "already saves animations as .gif and still plots as .png."
            )
        if dims == 2:
            from panchi.visualizations.backends.matplotlib_2d import (
                _MatplotlibBackend2D as backend_class,
            )
        else:
            from panchi.visualizations.backends.matplotlib_3d import (
                _MatplotlibBackend3D as backend_class,
            )
        return backend_class(save_path=save_path, quality=quality, figsize=figsize)
    if backend == "manim":
        if output_format not in (None, "mp4", "gif"):
            raise ValueError(
                f"Unknown output_format {output_format!r}. Choose 'mp4' or 'gif'."
            )
        if dims == 2:
            from panchi.visualizations.backends.manim_2d import (
                _ManimBackend2D as backend_class,
            )
        else:
            from panchi.visualizations.backends.manim_3d import (
                _ManimBackend3D as backend_class,
            )
        return backend_class(
            save_path=save_path or Path("./media"),
            quality=quality,
            include_extra_files=include_extra_files,
            output_format=output_format or "mp4",
        )
    raise ValueError(f"Unknown backend '{backend}'. Choose 'matplotlib' or 'manim'.")


def _to_float_vector(vector: Vector) -> Vector:
    """Return a float copy of ``vector`` for drawing."""
    try:
        return Vector([float(x) for x in vector.data])
    except TypeError:
        raise TypeError(
            f"Cannot plot {vector}: visualizations need real components."
        ) from None


def _to_float_space(space: VectorSpace) -> VectorSpace:
    """Return a float ``VectorSpace`` spanned by the exact basis of ``space``."""
    vectors = basis(space) or list(space.data)
    return VectorSpace([_to_float_vector(v) for v in vectors])


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


class _Animator:
    """Shared facade for Animator2D and Animator3D.

    Validates and converts inputs, then delegates drawing to the backend.
    Subclasses set ``_dims`` to the dimension of the space they draw.
    """

    _dims: int

    def __init__(
        self,
        backend: str = "matplotlib",
        save_path: str | Path | None = None,
        quality: str = "medium",
        figsize: tuple[int, int] = (8, 8),
        include_extra_files: bool = False,
        output_format: str | None = None,
    ) -> None:
        resolved_path = Path(save_path) if save_path else None
        self._backend = _build_backend(
            backend,
            self._dims,
            resolved_path,
            quality,
            figsize,
            include_extra_files,
            output_format,
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
    ) -> object | None:
        """Plot a list of vectors.

        Parameters
        ----------
        vectors : list[Vector]
            Vectors to plot, with 2 components for Animator2D or 3 for
            Animator3D.
        colors : list[str], optional
            Colors for each vector.
        labels : list[str], optional
            Labels for each vector.
        grid : bool
            Whether to show grid lines. Default ``True``.
        name : str, optional
            Output filename (without extension) when saving.
        """
        self._validate_vectors(vectors)
        return self._backend.plot_vectors(
            [_to_float_vector(v) for v in vectors],
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
        """Animate the addition of two vectors.

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
        self._validate_vectors([v1, v2])
        return self._backend.animate_addition(
            _to_float_vector(v1),
            _to_float_vector(v2),
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
        """Animate scalar multiplication of a vector.

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
        self._validate_vectors([vector])
        return self._backend.animate_scaling(
            _to_float_vector(vector),
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
        """Animate a linear transformation (2x2 for Animator2D, 3x3 for Animator3D).

        Each vector in ``vectors`` slides smoothly from ``v`` to ``matrix @ v``;
        when ``vectors`` is ``None`` the standard basis is shown. In 2D the
        coordinate grid deforms along with them; in 3D the image of the
        standard basis is the parallelepiped spanned by the matrix columns.

        Parameters
        ----------
        matrix : Matrix
            The transformation matrix.
        vectors : list[Vector], optional
            Vectors to transform. ``None`` uses the standard basis.
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
        self._validate_matrix(matrix)
        if vectors is not None:
            self._validate_vectors(vectors)
        return self._backend.animate_transform(
            matrix,
            vectors=None if vectors is None else [_to_float_vector(v) for v in vectors],
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
    ) -> object | None:
        """Visualize the span of vectors with a shaded region and basis arrows.

        Accepts a single span (a ``VectorSpace`` or a list of ``Vector``) or
        several spans to compare (a list whose items are each a ``VectorSpace``
        or a list of ``Vector``). A 1D subspace is drawn as a line through the
        origin, a 2D subspace as a plane (in 2D it shades the whole view), and
        a 3D subspace as a translucent volume. With multiple spans each is
        drawn in its own color with a legend.

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
            self._validate_vectors(vectors)
        return self._backend.plot_span(
            [_to_float_space(space) for _, space in spans],
            colors=colors,
            labels=labels,
            grid=grid,
            name=name or "plot_span",
            span_color=span_color,
        )

    def _validate_vectors(self, vectors: list[Vector]) -> None:
        if not isinstance(vectors, list):
            raise TypeError(
                f"Vectors must be passed as a list. Wrap a single vector as [v]. "
                f"Got {type(vectors).__name__}."
            )
        for v in vectors:
            if v.dims != self._dims:
                raise ValueError(
                    f"Only {self._dims}D vectors are supported for "
                    f"{type(self).__name__}. Got {v.dims}D vector: {v}"
                )

    def _validate_matrix(self, matrix: Matrix) -> None:
        n = self._dims
        if matrix.shape != (n, n):
            raise ValueError(
                f"Only {n}x{n} matrices are supported for transformation "
                f"visualization. Got shape {matrix.rows}x{matrix.cols}."
            )


class Animator2D(_Animator):
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
    include_extra_files : bool
        Keep manim's scratch directories next to the output (manim only).
        Default ``False``.
    output_format : str, optional
        ``"mp4"`` (default) or ``"gif"`` (manim only). With the manim
        backend every method returns the saved file, which plays inline
        when it is the last value in a Jupyter cell.

    Examples
    --------
    >>> from panchi import Vector, Matrix
    >>> from panchi.visualizations import Animator2D
    >>>
    >>> animator = Animator2D()
    >>> animator.plot_vectors([Vector([3, 2]), Vector([1, 3])])
    >>> animator.animate_transform(Matrix([[0, -1], [1, 0]]))
    """

    _dims = 2


class Animator3D(_Animator):
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
        Figure size in inches (matplotlib only). Default ``(8, 8)``.
    include_extra_files : bool
        Keep manim's scratch directories next to the output (manim only).
        Default ``False``.
    output_format : str, optional
        ``"mp4"`` (default) or ``"gif"`` (manim only). With the manim
        backend every method returns the saved file, which plays inline
        when it is the last value in a Jupyter cell.

    Examples
    --------
    >>> from panchi import Vector
    >>> from panchi.visualizations import Animator3D
    >>>
    >>> animator = Animator3D()
    >>> animator.plot_vectors([Vector([3, 2, 1]), Vector([1, 3, 2])])
    """

    _dims = 3
