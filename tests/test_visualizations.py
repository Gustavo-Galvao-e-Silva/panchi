import matplotlib

matplotlib.use("Agg")

import pytest

from panchi import Matrix, Vector, VectorSpace
from panchi.visualizations import Animator2D

# ==================== ANIMATOR2D INITIALIZATION ====================


class TestAnimator2DInit:
    """Test cases for Animator2D initialization and backend selection."""

    def test_default_backend_is_matplotlib(self):
        animator = Animator2D()
        print(f"\n✓ Default backend → {animator.backend}")
        assert animator.backend == "matplotlib"

    def test_explicit_matplotlib_backend(self):
        animator = Animator2D(backend="matplotlib")
        print(f"\n✓ Explicit matplotlib → {animator.backend}")
        assert animator.backend == "matplotlib"

    def test_invalid_backend_raises_value_error(self):
        print("\n✓ Invalid backend → raises ValueError")
        with pytest.raises(ValueError, match="Unknown backend"):
            Animator2D(backend="invalid")

    def test_save_path_creates_directory(self, tmp_path):
        save_dir = tmp_path / "output"
        animator = Animator2D(save_path=save_dir)
        v = Vector([1, 2])
        animator.plot_vectors([v])
        print(f"\n✓ save_path creates directory → {save_dir.exists()}")
        assert save_dir.exists()

    def test_quality_options(self):
        for quality in ("low", "medium", "high"):
            animator = Animator2D(quality=quality)
            print(f"\n✓ Quality '{quality}' accepted")
            assert animator is not None


# ==================== PLOT VECTORS ====================


class TestPlotVectors:
    """Test cases for Animator2D.plot_vectors."""

    def test_single_vector(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v = Vector([2, 3])
        animator.plot_vectors([v], labels=["v"])
        print(f"\n✓ Single vector saved to {tmp_path}")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_multiple_vectors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([1, 2])
        v2 = Vector([2, 1])
        v3 = Vector([-1, 1])
        animator.plot_vectors([v1, v2, v3], labels=["v1", "v2", "v3"])
        print("\n✓ Multiple vectors saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([1, 2])
        v2 = Vector([2, 1])
        animator.plot_vectors(
            [v1, v2], colors=["#FF0000", "#0000FF"], labels=["red", "blue"]
        )
        print("\n✓ Custom colors accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_no_labels(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors([Vector([1, 2]), Vector([2, 1])])
        print("\n✓ No labels accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_no_grid(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors([Vector([1, 2])], grid=False)
        print("\n✓ Grid disabled accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors([Vector([1, 2])], name="my_plot")
        print("\n✓ Custom name → my_plot.png")
        assert (tmp_path / "my_plot.png").exists()

    def test_rejects_3d_vector(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_vectors([Vector([1, 2, 3])])

    def test_rejects_mixed_dimensions(self):
        animator = Animator2D()
        print("\n✓ Mixed 2D/3D → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_vectors([Vector([1, 2]), Vector([1, 2, 3])])

    def test_rejects_bare_vector(self):
        animator = Animator2D()
        print("\n✓ Bare Vector (not a list) → raises TypeError")
        with pytest.raises(TypeError, match="list"):
            animator.plot_vectors(Vector([1, 2]))


# ==================== ANIMATE ADDITION ====================


class TestAnimateAddition:
    """Test cases for Animator2D.animate_addition."""

    def test_basic_addition(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([2, 1])
        v2 = Vector([1, 2])
        animator.animate_addition(v1, v2, frames=5, interval=100)
        print("\n✓ Addition animation saved")
        assert (tmp_path / "animate_addition.gif").exists()

    def test_negative_vectors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([3, 2])
        v2 = Vector([-1, -2])
        animator.animate_addition(v1, v2, frames=5, interval=100)
        print("\n✓ Negative vector addition saved")
        assert (tmp_path / "animate_addition.gif").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_addition(
            Vector([1, 0]), Vector([0, 1]), frames=5, interval=100, name="add_test"
        )
        print("\n✓ Custom name → add_test.gif")
        assert (tmp_path / "add_test.gif").exists()

    def test_rejects_3d_vectors(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.animate_addition(Vector([1, 2, 3]), Vector([4, 5, 6]))


# ==================== ANIMATE SCALING ====================


class TestAnimateScaling:
    """Test cases for Animator2D.animate_scaling."""

    def test_scale_up(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_scaling(
            Vector([2, 1]), scale_factor=2.0, frames=5, interval=100
        )
        print("\n✓ Scale up animation saved")
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_scale_down(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_scaling(
            Vector([3, 2]), scale_factor=0.5, frames=5, interval=100
        )
        print("\n✓ Scale down animation saved")
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_negative_scale(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_scaling(
            Vector([2, 3]), scale_factor=-1.5, frames=5, interval=100
        )
        print("\n✓ Negative scale animation saved")
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_rejects_3d_vector(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.animate_scaling(Vector([1, 2, 3]), scale_factor=2.0)


# ==================== ANIMATE TRANSFORM ====================


class TestAnimateTransform:
    """Test cases for Animator2D.animate_transform."""

    def test_rotation_matrix(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        rotation = Matrix([[0, -1], [1, 0]])
        animator.animate_transform(rotation, frames=5, interval=100)
        print("\n✓ Rotation transform animation saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_shear_matrix(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        shear = Matrix([[1, 1], [0, 1]])
        animator.animate_transform(shear, frames=5, interval=100)
        print("\n✓ Shear transform animation saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_identity_matrix(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        identity = Matrix([[1, 0], [0, 1]])
        animator.animate_transform(identity, frames=5, interval=100)
        print("\n✓ Identity transform animation saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[2, 0], [0, 2]]), frames=5, interval=100, name="scale_transform"
        )
        print("\n✓ Custom name → scale_transform.gif")
        assert (tmp_path / "scale_transform.gif").exists()

    def test_rejects_non_2x2(self):
        animator = Animator2D()
        print("\n✓ 3x3 matrix → raises ValueError")
        with pytest.raises(ValueError, match="2x2"):
            animator.animate_transform(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_rejects_non_square(self):
        animator = Animator2D()
        print("\n✓ 2x3 matrix → raises ValueError")
        with pytest.raises(ValueError, match="2x2"):
            animator.animate_transform(Matrix([[1, 2, 3], [4, 5, 6]]))


# ==================== PLOT SPAN ====================


class TestPlotSpan:
    """Test cases for Animator2D.plot_span."""

    def test_span_from_single_vector(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span([Vector([1, 2])])
        print("\n✓ 1D span from single vector saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_span_from_two_vectors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span([Vector([1, 0]), Vector([0, 1])])
        print("\n✓ 2D span from two vectors saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_span_from_vector_space(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        space = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        animator.plot_span(space)
        print("\n✓ Span from VectorSpace saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_1d_subspace(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span([Vector([1, 2]), Vector([2, 4])], name="1d_span")
        print("\n✓ 1D subspace (linearly dependent vectors) saved")
        assert (tmp_path / "1d_span.png").exists()

    def test_custom_labels_and_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span(
            [Vector([1, 0]), Vector([0, 1])],
            colors=["#FF0000", "#0000FF"],
            labels=["e1", "e2"],
        )
        print("\n✓ Custom labels and colors accepted")
        assert (tmp_path / "plot_span.png").exists()

    def test_rejects_3d_vectors(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_span([Vector([1, 2, 3])])

    def test_rejects_invalid_type(self):
        animator = Animator2D()
        print("\n✓ Invalid type → raises TypeError")
        with pytest.raises(TypeError, match="list of vectors or a VectorSpace"):
            animator.plot_span("not a vector")

    def test_rejects_empty_list(self):
        animator = Animator2D()
        print("\n✓ Empty list → raises ValueError")
        with pytest.raises(ValueError, match="at least one"):
            animator.plot_span([])

    def test_custom_span_color(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span([Vector([1, 0]), Vector([0, 1])], span_color="#805B49")
        print("\n✓ Custom span_color accepted")
        assert (tmp_path / "plot_span.png").exists()


# ==================== CUSTOM ANIMATION COLORS ====================


class TestAnimationColors:
    """The animate_* methods accept an optional per-role ``colors`` list."""

    def test_addition_custom_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_addition(
            Vector([2, 1]),
            Vector([1, 2]),
            frames=5,
            interval=100,
            colors=["#805B49", "#FFB592", "#45423F"],
        )
        print("\n✓ animate_addition custom colors saved")
        assert (tmp_path / "animate_addition.gif").exists()

    def test_scaling_custom_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_scaling(
            Vector([2, 1]),
            scale_factor=2.0,
            frames=5,
            interval=100,
            colors=["#805B49", "#FFB592"],
        )
        print("\n✓ animate_scaling custom colors saved")
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_transform_custom_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[1, 1], [0, 1]]),
            frames=5,
            interval=100,
            colors=["#805B49", "#FFB592"],
        )
        print("\n✓ animate_transform custom colors saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_partial_colors_keep_defaults(self):
        from panchi.visualizations.backends.matplotlib_2d import ADDITION_COLORS
        from panchi.visualizations.backends.matplotlib_base import _resolve_colors

        # A partial list fills only the roles supplied; the rest fall back.
        resolved = _resolve_colors(["#805B49"], ADDITION_COLORS)
        print(f"\n✓ Partial colors → {resolved}")
        assert resolved[0] == "#805B49"
        assert resolved[1] == ADDITION_COLORS[1]
        assert resolved[2] == ADDITION_COLORS[2]

    def test_none_colors_reproduce_defaults(self):
        from panchi.visualizations.backends.matplotlib_2d import TRANSFORM_COLORS
        from panchi.visualizations.backends.matplotlib_base import _resolve_colors

        resolved = _resolve_colors(None, TRANSFORM_COLORS)
        print(f"\n✓ None colors → default palette {resolved}")
        assert resolved == list(TRANSFORM_COLORS)


# ==================== INLINE NOTEBOOK PLAYBACK ====================


class TestInlineNotebookPlayback2D:
    """animate_* returns an inline-playable object under a notebook backend."""

    @staticmethod
    def _force_notebook(monkeypatch):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(
            plt, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
        )

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def _assert_inline(self, result):
        html = result._repr_html_()
        assert isinstance(html, str)
        assert html and "<" in html

    def test_addition_returns_inline_html(self, monkeypatch):
        self._force_notebook(monkeypatch)
        animator = Animator2D()
        result = animator.animate_addition(
            Vector([1, 0]), Vector([0, 1]), frames=5, interval=100
        )
        self._assert_inline(result)

    def test_scaling_returns_inline_html(self, monkeypatch):
        self._force_notebook(monkeypatch)
        animator = Animator2D()
        result = animator.animate_scaling(
            Vector([1, 1]), scale_factor=2.0, frames=5, interval=100
        )
        self._assert_inline(result)

    def test_transform_returns_inline_html(self, monkeypatch):
        self._force_notebook(monkeypatch)
        animator = Animator2D()
        result = animator.animate_transform(
            Matrix([[0, -1], [1, 0]]), frames=5, interval=100
        )
        self._assert_inline(result)

    def test_non_notebook_returns_none(self):
        animator = Animator2D()
        result = animator.animate_addition(
            Vector([1, 0]), Vector([0, 1]), frames=5, interval=100
        )
        assert result is None

    def test_saving_still_returns_none(self, monkeypatch, tmp_path):
        self._force_notebook(monkeypatch)
        animator = Animator2D(save_path=tmp_path)
        result = animator.animate_addition(
            Vector([1, 0]), Vector([0, 1]), frames=5, interval=100
        )
        assert result is None
        assert (tmp_path / "animate_addition.gif").exists()
