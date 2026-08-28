from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

try:
    import manim

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    raise

from panchi.visualizations.backends.base import _BaseBackend

DEFAULT_COLORS = [manim.RED, manim.ORANGE, manim.GREEN, manim.BLUE, manim.PURPLE]

# Per-method default color roles, shared across dimensions. Overridable via
# ``colors=`` / ``span_color=``.
ADDITION_COLORS = (manim.RED, manim.ORANGE, manim.GREEN)  # v1, v2, result
SCALING_COLORS = (manim.RED, manim.BLUE)  # original, scaled
TRANSFORM_COLORS = (manim.RED, manim.BLUE)  # e1, e2 (2D)
TRANSFORM_COLORS_3D = (manim.RED, manim.BLUE, manim.GREEN)  # e1, e2, e3 (3D)
SPAN_COLOR = manim.PURPLE

AXIS_COLOR = manim.GREY_B
GUIDE_COLOR = manim.GREY
PARALLELOGRAM_COLOR = manim.BLUE_E
MATRIX_HUD_COLOR = manim.YELLOW

AXIS_PADDING = 1.4
_EPSILON = 1e-12

_QUALITY_MAP = {
    "low": "low_quality",
    "medium": "medium_quality",
    "high": "high_quality",
    "production": "production_quality",
}

# Scratch/cache subdirectories manim writes alongside the final ``<name>.mp4``.
# Removed after rendering unless ``include_extra_files`` is set.
_INTERMEDIATE_DIRS = ("partial_movie_files", "Tex", "texts", "images")


def _resolve_colors(colors: list[str] | None, defaults: tuple) -> list:
    """Fill in per-role colors positionally, keeping defaults for omitted roles."""
    colors = colors or []
    return [colors[i] if i < len(colors) else defaults[i] for i in range(len(defaults))]


def _axis_range(coords, padding: float = AXIS_PADDING) -> int:
    """Symmetric axis half-extent covering ``coords`` with padding."""
    return int(max(abs(c) for c in coords) * padding) + 1


class _BuilderSceneMixin:
    """Delegates a scene's ``construct`` to a supplied builder callable.

    Combined with a per-dimension ``Scene``/``ThreeDScene`` subclass so each
    backend method can describe its animation as a plain function of the scene
    instead of declaring a bespoke scene subclass inline.
    """

    def __init__(self, builder: Callable) -> None:
        super().__init__()
        self._builder = builder

    def construct(self) -> None:
        self._builder(self)


class _ManimBackendBase(_BaseBackend):
    """Dimension-agnostic scaffolding shared by the manim backends.

    Owns output configuration, scratch-dir cleanup, and the render loop; the
    dimension-specific drawing lives in the subclasses, which set
    ``_function_scene_cls`` to the builder-scene to render.
    """

    _function_scene_cls: type

    def __init__(
        self, save_path: Path | None, quality: str, include_extra_files: bool
    ) -> None:
        super().__init__(save_path=save_path, quality=quality)
        self.include_extra_files = include_extra_files

    def _configure(self, name: str) -> None:
        manim.config.quality = _QUALITY_MAP.get(self.quality, "medium_quality")
        manim.config.disable_caching = True
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            manim.config.media_dir = str(self.save_path)
            manim.config.video_dir = "{media_dir}"
            manim.config.output_file = name

    def _cleanup(self) -> None:
        """Remove manim's intermediary scratch dirs, keeping the final ``.mp4``.

        The rendered video lands directly in ``save_path``; manim also leaves
        ``Tex/``, ``images/`` and ``partial_movie_files/`` behind. Those are
        deleted unless the user opted into ``include_extra_files``.
        """
        if self.include_extra_files or not self.save_path:
            return
        for sub in _INTERMEDIATE_DIRS:
            shutil.rmtree(self.save_path / sub, ignore_errors=True)

    def _render(self, name: str, builder: Callable) -> None:
        """Configure output, render the builder as a scene, then clean up."""
        self._configure(name)
        self._function_scene_cls(builder).render()
        self._cleanup()
