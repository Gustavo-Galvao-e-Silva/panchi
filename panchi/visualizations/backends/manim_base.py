from __future__ import annotations

import base64
import shutil
from collections.abc import Callable
from pathlib import Path

try:
    import manim

    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    raise

from panchi.visualizations.backends.base import (  # noqa: F401
    ADDITION_COLORS,
    DEFAULT_COLORS,
    SCALING_COLORS,
    SPAN_COLOR,
    TRANSFORM_COLORS,
    TRANSFORM_COLORS_3D,
    _BaseBackend,
    _resolve_colors,
    _resolve_n_colors,
)

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


def _axis_range(coords, padding: float = AXIS_PADDING) -> int:
    """Symmetric axis half-extent covering ``coords`` with padding."""
    return int(max(abs(c) for c in coords) * padding) + 1


class _ManimOutput:
    """A rendered manim file that plays inline in Jupyter and Colab.

    Returned by every manim backend method. Jupyter renders it through
    ``_repr_html_``, embedding the file so the notebook keeps working after
    the file moves; elsewhere it is a handle on the saved file, ``path``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def __repr__(self) -> str:
        return f"ManimOutput('{self.path}')"

    def _repr_html_(self) -> str:
        data = base64.b64encode(self.path.read_bytes()).decode("ascii")
        if self.path.suffix == ".gif":
            return f'<img src="data:image/gif;base64,{data}">'
        return (
            f"<video autoplay loop muted controls playsinline "
            f'src="data:video/mp4;base64,{data}"></video>'
        )


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
        self,
        save_path: Path | None,
        quality: str,
        include_extra_files: bool,
        output_format: str = "mp4",
    ) -> None:
        super().__init__(save_path=save_path, quality=quality)
        self.include_extra_files = include_extra_files
        self.output_format = output_format

    def _configure(self, name: str) -> None:
        manim.config.quality = _QUALITY_MAP.get(self.quality, "medium_quality")
        manim.config.disable_caching = True
        manim.config.format = self.output_format
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            manim.config.media_dir = str(self.save_path)
            manim.config.video_dir = "{media_dir}"
            manim.config.output_file = name

    def _cleanup(self) -> None:
        """Remove manim's intermediary scratch dirs, keeping the final file.

        The rendered ``.mp4`` or ``.gif`` lands directly in ``save_path``; manim also leaves
        ``Tex/``, ``images/`` and ``partial_movie_files/`` behind. Those are
        deleted unless the user opted into ``include_extra_files``.
        """
        if self.include_extra_files or not self.save_path:
            return
        for sub in _INTERMEDIATE_DIRS:
            shutil.rmtree(self.save_path / sub, ignore_errors=True)

    def _render(self, name: str, builder: Callable) -> _ManimOutput:
        """Render the builder as a scene and return the saved file."""
        self._configure(name)
        scene = self._function_scene_cls(builder)
        scene.render()
        self._cleanup()
        writer = scene.renderer.file_writer
        if self.output_format == "gif":
            return _ManimOutput(Path(writer.gif_file_path))
        return _ManimOutput(Path(writer.movie_file_path))
