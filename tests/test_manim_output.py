import base64

import pytest

from panchi import Matrix, Vector
from panchi.visualizations import Animator2D, Animator3D

# ==================== OUTPUT FORMAT VALIDATION ====================


class TestOutputFormatValidation:
    """output_format is checked before any backend is imported."""

    @pytest.mark.parametrize("cls", [Animator2D, Animator3D])
    def test_rejected_with_matplotlib(self, cls):
        with pytest.raises(ValueError, match="only applies to the manim backend"):
            cls(backend="matplotlib", output_format="gif")

    @pytest.mark.parametrize("cls", [Animator2D, Animator3D])
    def test_unknown_format_rejected(self, cls):
        with pytest.raises(ValueError, match="Choose 'mp4' or 'gif'"):
            cls(backend="manim", output_format="webm")

    def test_unknown_backend_still_reported(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Animator2D(backend="plotly", output_format="gif")


# ==================== INLINE NOTEBOOK OUTPUT ====================

_TINY_GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


class TestManimOutput:
    """_ManimOutput embeds the rendered file for Jupyter."""

    @pytest.fixture(autouse=True)
    def _manim(self):
        pytest.importorskip("manim")

    def _output(self, path):
        from panchi.visualizations.backends.manim_base import _ManimOutput

        return _ManimOutput(path)

    def test_gif_embeds_as_image(self, tmp_path):
        path = tmp_path / "clip.gif"
        path.write_bytes(_TINY_GIF)
        html = self._output(path)._repr_html_()
        assert html.startswith('<img src="data:image/gif;base64,')
        assert base64.b64encode(_TINY_GIF).decode("ascii") in html

    def test_mp4_embeds_as_looping_video(self, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"not really a video")
        html = self._output(path)._repr_html_()
        assert html.startswith("<video autoplay loop muted controls playsinline")
        assert 'src="data:video/mp4;base64,' in html

    def test_repr_and_path(self, tmp_path):
        path = tmp_path / "clip.gif"
        output = self._output(path)
        assert output.path == path
        assert repr(output) == f"ManimOutput('{path}')"


# ==================== REAL RENDERS ====================


@pytest.mark.manim
class TestManimRender:
    """Low-quality renders in each format return the saved file."""

    @pytest.fixture(autouse=True)
    def _manim(self):
        pytest.importorskip("manim")

    @pytest.mark.parametrize("fmt", ["gif", "mp4"])
    def test_2d_animation_saves_requested_format(self, tmp_path, fmt):
        animator = Animator2D(
            backend="manim", save_path=tmp_path, quality="low", output_format=fmt
        )
        output = animator.animate_scaling(Vector([1, 2]), 2, name="scale")
        assert output.path == tmp_path / f"scale.{fmt}"
        assert output.path.exists()

    def test_3d_plot_returns_output(self, tmp_path):
        animator = Animator3D(
            backend="manim", save_path=tmp_path, quality="low", output_format="gif"
        )
        output = animator.plot_vectors([Vector([1, 0, 1])], name="vec")
        assert output.path.suffix == ".gif"
        assert output.path.exists()

    def test_formats_do_not_leak_between_animators(self, tmp_path):
        Animator2D(
            backend="manim", save_path=tmp_path, quality="low", output_format="gif"
        ).animate_scaling(Vector([1, 2]), 2, name="first")
        second = Animator2D(
            backend="manim", save_path=tmp_path, quality="low"
        ).animate_transform(Matrix([[0, -1], [1, 0]]), name="second")
        assert second.path.suffix == ".mp4"

    def test_transform_without_matrix_or_labels(self, tmp_path):
        output = Animator2D(
            backend="manim", save_path=tmp_path, quality="low"
        ).animate_transform(
            Matrix([[1, -2], [1, 3]]),
            vectors=[Vector([1, 0]), Vector([1, 1])],
            name="bare",
            show_matrix=False,
            show_labels=False,
        )
        assert output.path.exists()

    def test_3d_transform_without_matrix(self, tmp_path):
        output = Animator3D(
            backend="manim", save_path=tmp_path, quality="low"
        ).animate_transform(
            Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), show_matrix=False
        )
        assert output.path.exists()

    def test_only_final_files_remain(self, tmp_path):
        Animator2D(
            backend="manim", save_path=tmp_path, quality="low", output_format="gif"
        ).animate_scaling(Vector([1, 2]), 2, name="clean")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.gif"]
