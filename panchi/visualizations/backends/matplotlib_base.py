from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from panchi.primitives.vector import Vector
from panchi.visualizations.backends.base import _BaseBackend

DEFAULT_COLORS = ["#E63946", "#F77F00", "#06FFA5", "#118AB2", "#073B4C"]

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


def _calculate_axis_range(vectors: list[Vector]) -> tuple[float, float]:
    coords = [v[i] for v in vectors for i in range(v.dims)]
    max_coord = max((abs(c) for c in coords), default=0.0) * AXIS_PADDING
    max_coord = max(max_coord, MIN_AXIS_RANGE)
    return (-max_coord, max_coord)


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
    ) -> None:
        """Build a blitting ``FuncAnimation`` from ``frame_fn`` and finalize it.

        Short pauses hold the first and last frames, so each loop opens on the
        static setup and settles on the result before restarting.
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
            blit=True,
            repeat=True,
        )
        self._finalize_animation(anim, name, interval)

    def _finalize_animation(
        self,
        anim: animation.FuncAnimation,
        name: str,
        interval: int,
    ) -> None:
        plt.tight_layout()
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            anim.save(
                str(self.save_path / f"{name}.gif"),
                writer="pillow",
                fps=1000 // interval,
            )
            plt.close(anim._fig)
        else:
            plt.show()
