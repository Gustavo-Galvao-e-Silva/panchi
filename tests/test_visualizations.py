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
        animator.plot_vectors(v)
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
        animator.plot_vectors(v, labels=["v"])
        print(f"\n✓ Single vector saved to {tmp_path}")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_multiple_vectors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([1, 2])
        v2 = Vector([2, 1])
        v3 = Vector([-1, 1])
        animator.plot_vectors(v1, v2, v3, labels=["v1", "v2", "v3"])
        print("\n✓ Multiple vectors saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        v1 = Vector([1, 2])
        v2 = Vector([2, 1])
        animator.plot_vectors(
            v1, v2, colors=["#FF0000", "#0000FF"], labels=["red", "blue"]
        )
        print("\n✓ Custom colors accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_no_labels(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 2]), Vector([2, 1]))
        print("\n✓ No labels accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_no_grid(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 2]), grid=False)
        print("\n✓ Grid disabled accepted")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 2]), name="my_plot")
        print("\n✓ Custom name → my_plot.png")
        assert (tmp_path / "my_plot.png").exists()

    def test_rejects_3d_vector(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_vectors(Vector([1, 2, 3]))

    def test_rejects_mixed_dimensions(self):
        animator = Animator2D()
        print("\n✓ Mixed 2D/3D → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_vectors(Vector([1, 2]), Vector([1, 2, 3]))


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
        animator.plot_span(Vector([1, 2]))
        print("\n✓ 1D span from single vector saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_span_from_two_vectors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span(Vector([1, 0]), Vector([0, 1]))
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
        animator.plot_span(Vector([1, 2]), Vector([2, 4]), name="1d_span")
        print("\n✓ 1D subspace (linearly dependent vectors) saved")
        assert (tmp_path / "1d_span.png").exists()

    def test_custom_labels_and_colors(self, tmp_path):
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span(
            Vector([1, 0]),
            Vector([0, 1]),
            colors=["#FF0000", "#0000FF"],
            labels=["e1", "e2"],
        )
        print("\n✓ Custom labels and colors accepted")
        assert (tmp_path / "plot_span.png").exists()

    def test_rejects_3d_vectors(self):
        animator = Animator2D()
        print("\n✓ 3D vector → raises ValueError")
        with pytest.raises(ValueError, match="2D"):
            animator.plot_span(Vector([1, 2, 3]))

    def test_rejects_invalid_type(self):
        animator = Animator2D()
        print("\n✓ Invalid type → raises TypeError")
        with pytest.raises(TypeError, match="Expected Vector or VectorSpace"):
            animator.plot_span("not a vector")

    def test_rejects_empty_call(self):
        animator = Animator2D()
        print("\n✓ No arguments → raises ValueError")
        with pytest.raises(ValueError, match="At least one"):
            animator.plot_span()
