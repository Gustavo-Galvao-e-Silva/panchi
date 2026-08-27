import matplotlib

matplotlib.use("Agg")

import pytest

from panchi import Vector
from panchi.visualizations import Animator3D

# ==================== ANIMATOR3D INITIALIZATION ====================


class TestAnimator3DInit:
    """Test cases for Animator3D initialization and backend selection."""

    def test_default_backend_is_matplotlib(self):
        animator = Animator3D()
        print(f"\n✓ Default backend → {animator.backend}")
        assert animator.backend == "matplotlib"

    def test_manim_backend_constructs(self):
        pytest.importorskip("manim")
        animator = Animator3D(backend="manim")
        print(f"\n✓ manim backend → {animator.backend}")
        assert animator.backend == "manim"

    def test_invalid_backend_raises_value_error(self):
        print("\n✓ Invalid backend → raises ValueError")
        with pytest.raises(ValueError, match="Unknown backend"):
            Animator3D(backend="invalid")

    def test_save_path_creates_directory(self, tmp_path):
        save_dir = tmp_path / "output"
        animator = Animator3D(save_path=save_dir)
        animator.plot_vectors(Vector([1, 2, 3]))
        print(f"\n✓ save_path creates directory → {save_dir.exists()}")
        assert save_dir.exists()


# ==================== PLOT VECTORS ====================


class TestPlotVectors3D:
    """Test cases for Animator3D.plot_vectors."""

    def test_single_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(Vector([2, 3, 1]), labels=["v"])
        print(f"\n✓ Single 3D vector saved to {tmp_path}")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_multiple_vectors(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 2, 3]), Vector([-2, 1, 2]), Vector([0, -3, 1]))
        print("\n✓ Multiple 3D vectors saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_colors_and_labels(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(
            Vector([1, 0, 2]),
            Vector([0, 2, 1]),
            colors=["#805B49", "#FFB592"],
            labels=["a", "b"],
        )
        print("\n✓ Custom colors/labels saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_grid_off(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 1, 1]), grid=False)
        print("\n✓ grid=False saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(Vector([1, 2, 3]), name="my_plot")
        print("\n✓ Custom name saved")
        assert (tmp_path / "my_plot.png").exists()


# ==================== DIMENSION VALIDATION ====================


class TestAnimations3D:
    """Test cases for Animator3D 3D animations (matplotlib backend)."""

    def test_animate_scaling(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_scaling(Vector([1, 2, 1]), 2.0, frames=6, interval=50)
        print("\n✓ 3D scaling gif saved")
        assert (tmp_path / "animate_scaling.png").exists() is False
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_animate_scaling_negative(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_scaling(Vector([1, 0, 2]), -1.5, frames=6, interval=50)
        print("\n✓ 3D negative scaling gif saved")
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_animate_addition(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_addition(
            Vector([2, 1, 0]), Vector([1, 2, 2]), frames=6, interval=50
        )
        print("\n✓ 3D addition gif saved")
        assert (tmp_path / "animate_addition.gif").exists()

    def test_animate_custom_colors_and_name(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_addition(
            Vector([2, 1, 0]),
            Vector([1, 2, 2]),
            frames=6,
            interval=50,
            colors=["#805B49", "#FFB592"],
            name="my_add",
        )
        print("\n✓ 3D addition custom colors/name saved")
        assert (tmp_path / "my_add.gif").exists()

    def test_animate_addition_rejects_2d(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 2D vectors → animate_addition raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.animate_addition(Vector([1, 2]), Vector([3, 4]), frames=6)

    def test_animate_scaling_rejects_2d(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 2D vector → animate_scaling raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.animate_scaling(Vector([1, 2]), 2.0, frames=6)


class TestValidate3D:
    """Test cases for Animator3D._validate_3d."""

    def test_rejects_2d_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 2D vector → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors(Vector([1, 2]))

    def test_rejects_4d_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 4D vector → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors(Vector([1, 2, 3, 4]))

    def test_rejects_mixed_dimensions(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ Mixed 3D + 2D → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors(Vector([1, 2, 3]), Vector([1, 2]))
