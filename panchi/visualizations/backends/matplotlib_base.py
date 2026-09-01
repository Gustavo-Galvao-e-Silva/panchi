from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from panchi.primitives.vector import Vector
from panchi.visualizations.backends.base import _BaseBackend

DEFAULT_COLORS = ["#805B49", "#FFB592", "#3E7C7B", "#D4A15E", "#45423F"]

# Per-method default color roles, shared across dimensions. Overridable by
# passing ``colors=`` (animations) or ``span_color=`` (spans) to the method.
ADDITION_COLORS = ("#805B49", "#FFB592", "#3E7C7B")  # v1, v2, result
SCALING_COLORS = ("#805B49", "#FFB592")  # original, scaled
TRANSFORM_COLORS = ("#805B49", "#FFB592")  # e1, e2 (2D)
TRANSFORM_COLORS_3D = ("#805B49", "#FFB592", "#3E7C7B")  # e1, e2, e3 (3D)
SPAN_COLOR = "#805B49"

GRID_COLOR = "#CCCCCC"

AXIS_PADDING = 1.3
MIN_AXIS_RANGE = 0.5
PAUSE_SECONDS = 0.4
_EPSILON = 1e-12

_QUALITY_DPI = {
    "low": 72,
    "medium": 100,
    "high": 150,
}


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


def _calculate_axis_range(vectors: list[Vector]) -> tuple[float, float]:
    coords = [v[i] for v in vectors for i in range(v.dims)]
    max_coord = max((abs(c) for c in coords), default=0.0) * AXIS_PADDING
    max_coord = max(max_coord, MIN_AXIS_RANGE)
    return (-max_coord, max_coord)


def _smooth_step(t: float) -> float:
    return t * t * (3 - 2 * t)


def _in_notebook() -> bool:
    """True when matplotlib is using an inline/notebook backend (Jupyter, Colab)."""
    backend = plt.get_backend().lower()
    return "inline" in backend or "nbagg" in backend or "ipympl" in backend


class _InlineAnimation:
    """Play a matplotlib animation inline in a notebook.

    Holds the JS player HTML that matplotlib's own ``to_jshtml`` produced, so
    Jupyter renders the animation via ``_repr_html_`` — no IPython import and no
    global rcParams mutation. Outside a notebook this object is simply ignored.
    """

    def __init__(self, html: str) -> None:
        self._html = html

    def _repr_html_(self) -> str:
        return self._html


class _MatplotlibBackendBase(_BaseBackend):
    """Shared scaffolding for matplotlib backends, independent of dimension.

    Holds output finalization (figures and animations) and the quality/figure
    configuration common to the 2D and 3D backends; the dimension-specific
    drawing lives in the subclasses.
    """

    def __init__(
        self,
        save_path: Path | None,
        quality: str,
        figsize: tuple[int, int],
    ) -> None:
        super().__init__(save_path=save_path, quality=quality)
        self.figsize = figsize
        self._dpi = _QUALITY_DPI.get(quality, 100)

    def _finalize_figure(self, fig: plt.Figure, name: str) -> None:
        plt.tight_layout()
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path / f"{name}.png", dpi=self._dpi)
            plt.close(fig)
        else:
            plt.show()

    def _run_animation(
        self,
        fig: plt.Figure,
        frame_fn,
        frames: int,
        interval: int,
        name: str,
        blit: bool = True,
    ) -> _InlineAnimation | None:
        """Build a ``FuncAnimation`` from ``frame_fn`` and finalize it.

        Short pauses hold the first and last frames, so each loop opens on the
        static setup and settles on the result before restarting. ``blit``
        defaults to ``True`` (2D); the 3D backend passes ``blit=False`` because
        mplot3d artists do not support blitting.
        """
        pause = max(1, round(PAUSE_SECONDS * 1000 / interval))
        last = frames - 1

        def with_pauses(frame: int) -> tuple:
            return frame_fn(min(max(frame - pause, 0), last))

        anim = animation.FuncAnimation(
            fig,
            with_pauses,
            frames=frames + 2 * pause,
            interval=interval,
            blit=blit,
            repeat=True,
        )
        return self._finalize_animation(anim, name, interval)

    def _finalize_animation(
        self,
        anim: animation.FuncAnimation,
        name: str,
        interval: int,
    ) -> _InlineAnimation | None:
        plt.tight_layout()
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            anim.save(
                str(self.save_path / f"{name}.gif"),
                writer="pillow",
                fps=1000 // interval,
            )
            plt.close(anim._fig)
            return None
        if _in_notebook():
            html = anim.to_jshtml()
            plt.close(anim._fig)
            return _InlineAnimation(html)
        plt.show()
        return None
