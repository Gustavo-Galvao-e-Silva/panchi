import matplotlib

matplotlib.use("Agg")

import pytest

from panchi import Matrix, Vector, VectorSpace
from panchi.algorithms import ref, rref
from panchi.symbolic import sqrt
from panchi.visualizations import Animator3D


def _capture_frame_function(monkeypatch, animator):
    """Replace playback with a capture of the backend's per-frame function."""
    import matplotlib.pyplot as plt

    captured = {}

    def fake_run_animation(fig, frame_fn, frames, interval, name, blit=True):
        captured["frame_fn"] = frame_fn
        captured["frames"] = frames
        captured["fig"] = fig
        plt.close(fig)
        return None

    monkeypatch.setattr(animator._backend, "_run_animation", fake_run_animation)
    return captured


def _record_arrow_draws(monkeypatch):
    """Record every ``_draw_arrow_3d`` call so the artists can be inspected."""
    from panchi.visualizations.backends import matplotlib_3d

    draws = []
    original = matplotlib_3d._draw_arrow_3d

    def record(ax, start, disp, color, linewidth=2.5):
        draws.append({"start": start, "disp": disp, "linewidth": linewidth})
        return original(ax, start, disp, color, linewidth)

    monkeypatch.setattr(matplotlib_3d, "_draw_arrow_3d", record)
    return draws


def _tip(draw):
    return tuple(
        start + displacement
        for start, displacement in zip(draw["start"], draw["disp"], strict=True)
    )


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
        animator.plot_vectors([Vector([1, 2, 3])])
        print(f"\n✓ save_path creates directory → {save_dir.exists()}")
        assert save_dir.exists()


# ==================== PLOT VECTORS ====================


class TestPlotVectors3D:
    """Test cases for Animator3D.plot_vectors."""

    def test_single_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors([Vector([2, 3, 1])], labels=["v"])
        print(f"\n✓ Single 3D vector saved to {tmp_path}")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_multiple_vectors(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(
            [Vector([1, 2, 3]), Vector([-2, 1, 2]), Vector([0, -3, 1])]
        )
        print("\n✓ Multiple 3D vectors saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_colors_and_labels(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors(
            [Vector([1, 0, 2]), Vector([0, 2, 1])],
            colors=["#805B49", "#FFB592"],
            labels=["a", "b"],
        )
        print("\n✓ Custom colors/labels saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_grid_off(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors([Vector([1, 1, 1])], grid=False)
        print("\n✓ grid=False saved")
        assert (tmp_path / "plot_vectors.png").exists()

    def test_custom_name(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors([Vector([1, 2, 3])], name="my_plot")
        print("\n✓ Custom name saved")
        assert (tmp_path / "my_plot.png").exists()

    def test_rejects_non_vector_in_list(self):
        animator = Animator3D()
        print("\n✓ Non-Vector in list → raises TypeError")
        with pytest.raises(TypeError, match="list of Vector objects"):
            animator.plot_vectors([Vector([1, 2, 3]), [1, 2, 3]])

    def test_rejects_non_matrix(self):
        animator = Animator3D()
        print("\n✓ Non-Matrix → raises TypeError")
        with pytest.raises(TypeError, match="Matrix"):
            animator.animate_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


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

    def test_animate_transform(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[1, 0.5, 0], [0, 1, 0.5], [0.3, 0, 1]]), frames=6, interval=50
        )
        print("\n✓ 3D transform gif saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_animate_transform_custom_colors_and_name(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
            frames=6,
            interval=50,
            colors=["#805B49", "#FFB592", "#45423F"],
            name="my_xf",
        )
        print("\n✓ 3D transform custom colors/name saved")
        assert (tmp_path / "my_xf.gif").exists()

    def test_animate_transform_rejects_non_3x3(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ Non-3x3 matrix → animate_transform raises ValueError")
        with pytest.raises(ValueError, match="Only 3x3 matrices"):
            animator.animate_transform(Matrix([[1, 0], [0, 1]]), frames=6)

    def test_animate_transform_custom_vectors(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[2, 0, 0], [0, 1, 0], [0, 1, 1]]),
            vectors=[Vector([1, 1, 0]), Vector([0, 1, 2])],
            frames=6,
            interval=50,
        )
        print("\n✓ 3D transform with custom vectors saved")
        assert (tmp_path / "animate_transform.gif").exists()

    def test_animate_transform_hide_matrix(self, tmp_path, monkeypatch):
        from mpl_toolkits.mplot3d import Axes3D

        texts = []
        monkeypatch.setattr(
            Axes3D, "text2D", lambda self, *args, **kwargs: texts.append(args)
        )
        animator = Animator3D(save_path=tmp_path)
        animator.animate_transform(
            Matrix([[2, 0, 0], [0, 1, 0], [0, 1, 1]]), frames=6, show_matrix=False
        )
        print("\n✓ 3D show_matrix=False → no matrix text")
        assert texts == []
        assert (tmp_path / "animate_transform.gif").exists()

    def test_animate_transform_rejects_non_3d_vectors(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.animate_transform(
                Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                vectors=[Vector([1, 2])],
                frames=6,
            )


class TestFinalFrame3D:
    """The last frame lands exactly on the animation's end state (issue #153)."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_addition_last_frame_ends_on_sum(self, monkeypatch):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)
        draws = _record_arrow_draws(monkeypatch)
        animator.animate_addition(
            Vector([3, 1, 0]), Vector([1, 3, 0]), frames=10, interval=100
        )
        draws.clear()

        captured["frame_fn"](captured["frames"] - 1)

        tip_draw, result_draw = draws
        assert tip_draw["start"] == pytest.approx((3, 1, 0))
        assert _tip(tip_draw) == pytest.approx((4, 4, 0))
        assert result_draw["start"] == pytest.approx((0, 0, 0))
        assert _tip(result_draw) == pytest.approx((4, 4, 0))

        label = captured["fig"].axes[0].texts[-1]
        assert label.get_text() == "v₁+v₂"

    def test_addition_first_half_ends_on_sum(self, monkeypatch):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)
        draws = _record_arrow_draws(monkeypatch)
        animator.animate_addition(
            Vector([3, 1, 0]), Vector([1, 3, 0]), frames=10, interval=100
        )
        draws.clear()

        captured["frame_fn"](captured["frames"] // 2 - 1)

        (tail_draw,) = draws
        assert tail_draw["start"] == pytest.approx((3, 1, 0))
        assert _tip(tail_draw) == pytest.approx((4, 4, 0))

    def test_scaling_last_frame_ends_on_scaled_vector(self, monkeypatch):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)
        draws = _record_arrow_draws(monkeypatch)
        animator.animate_scaling(Vector([2, 1, 1]), 3, frames=10, interval=100)
        draws.clear()

        captured["frame_fn"](captured["frames"] - 1)

        (draw,) = draws
        assert _tip(draw) == pytest.approx((6, 3, 3))

        label = captured["fig"].axes[0].texts[-1]
        assert label.get_text() == "3v"

    def test_transform_last_frame_ends_on_matrix_image(self, monkeypatch):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)
        draws = _record_arrow_draws(monkeypatch)
        animator.animate_transform(
            Matrix([[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
            frames=10,
            interval=100,
        )
        draws.clear()

        captured["frame_fn"](captured["frames"] - 1)

        e1_draw, e2_draw, e3_draw = draws
        assert _tip(e1_draw) == pytest.approx((2, 0, 0))
        assert _tip(e2_draw) == pytest.approx((1, 1, 0))
        assert _tip(e3_draw) == pytest.approx((0, 0, 1))

    @pytest.mark.parametrize("frames", [1, 2])
    def test_tiny_frame_counts_do_not_raise(self, monkeypatch, frames):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)

        animator.animate_scaling(Vector([2, 1, 1]), 3, frames=frames, interval=100)
        for frame in range(frames):
            captured["frame_fn"](frame)

        animator.animate_transform(
            Matrix([[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
            frames=frames,
            interval=100,
        )
        for frame in range(frames):
            captured["frame_fn"](frame)

    @pytest.mark.parametrize("frames", [1, 2])
    def test_addition_tiny_frame_counts_end_on_sum(self, monkeypatch, frames):
        animator = Animator3D()
        captured = _capture_frame_function(monkeypatch, animator)
        draws = _record_arrow_draws(monkeypatch)

        animator.animate_addition(
            Vector([3, 1, 0]), Vector([1, 3, 0]), frames=frames, interval=100
        )
        draws.clear()
        for frame in range(frames):
            captured["frame_fn"](frame)

        tip_draw, result_draw = draws[-2:]
        assert tip_draw["start"] == pytest.approx((3, 1, 0))
        assert _tip(tip_draw) == pytest.approx((4, 4, 0))
        assert result_draw["start"] == pytest.approx((0, 0, 0))
        assert _tip(result_draw) == pytest.approx((4, 4, 0))


class TestPlotSpan3D:
    """Test cases for Animator3D.plot_span (matplotlib backend)."""

    def test_span_line(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span([Vector([1, 1, 1])])
        print("\n✓ 3D span (line) saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_span_plane(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span([Vector([1, 0, 0]), Vector([0, 1, 1])], name="plane")
        print("\n✓ 3D span (plane) saved")
        assert (tmp_path / "plane.png").exists()

    def test_span_full_space(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span(
            [Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])], name="r3"
        )
        print("\n✓ 3D span (R³) saved")
        assert (tmp_path / "r3.png").exists()

    def test_span_origin(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span(VectorSpace([Vector([0, 0, 0])]), name="origin")
        print("\n✓ 3D span (origin) saved")
        assert (tmp_path / "origin.png").exists()

    def test_span_vectorspace_input(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        space = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
        animator.plot_span(space, labels=["u", "w"], span_color="#805B49")
        print("\n✓ 3D span (VectorSpace input) saved")
        assert (tmp_path / "plot_span.png").exists()

    def test_span_rejects_2d(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 2D vector → plot_span raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_span([Vector([1, 2])])

    def test_span_type_error(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ non-Vector → plot_span raises TypeError")
        with pytest.raises(TypeError, match="list of vectors, a VectorSpace"):
            animator.plot_span("not a vector")

    def test_multiple_spans(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span(
            [
                [Vector([1, 0, 0]), Vector([0, 1, 0])],
                VectorSpace([Vector([0, 0, 1])]),
            ],
            labels=["plane", "axis"],
            name="two_spans",
        )
        print("\n✓ Two 3D spans drawn with a legend")
        assert (tmp_path / "two_spans.png").exists()


class TestValidate3D:
    """Test cases for Animator3D._validate_3d."""

    def test_rejects_2d_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 2D vector → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors([Vector([1, 2])])

    def test_rejects_4d_vector(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ 4D vector → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors([Vector([1, 2, 3, 4])])

    def test_rejects_mixed_dimensions(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        print("\n✓ Mixed 3D + 2D → raises ValueError")
        with pytest.raises(ValueError, match="Only 3D vectors"):
            animator.plot_vectors([Vector([1, 2, 3]), Vector([1, 2])])


# ==================== INLINE NOTEBOOK PLAYBACK ====================


class TestInlineNotebookPlayback3D:
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
        animator = Animator3D()
        result = animator.animate_addition(
            Vector([1, 0, 0]), Vector([0, 1, 0]), frames=5, interval=100
        )
        self._assert_inline(result)

    def test_scaling_returns_inline_html(self, monkeypatch):
        self._force_notebook(monkeypatch)
        animator = Animator3D()
        result = animator.animate_scaling(
            Vector([1, 1, 1]), scale_factor=2.0, frames=5, interval=100
        )
        self._assert_inline(result)

    def test_transform_returns_inline_html(self, monkeypatch):
        self._force_notebook(monkeypatch)
        animator = Animator3D()
        result = animator.animate_transform(
            Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), frames=5, interval=100
        )
        self._assert_inline(result)

    def test_non_notebook_returns_none(self):
        animator = Animator3D()
        result = animator.animate_addition(
            Vector([1, 0, 0]), Vector([0, 1, 0]), frames=5, interval=100
        )
        assert result is None


class TestExactIrrationalInputs3D:
    """QuadraticNumber components are plotted in R³."""

    def test_plot_vectors(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_vectors([Vector([1 + sqrt(2), 1, 0]), Vector([0, sqrt(3), 1])])
        assert (tmp_path / "plot_vectors.png").exists()

    def test_animate_addition(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_addition(
            Vector([sqrt(2), 1, 0]), Vector([0, 1, sqrt(2)]), frames=5, interval=100
        )
        assert (tmp_path / "animate_addition.gif").exists()

    def test_animate_scaling_by_irrational(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.animate_scaling(
            Vector([sqrt(2), 1, 1]), scale_factor=sqrt(2), frames=5, interval=100
        )
        assert (tmp_path / "animate_scaling.gif").exists()

    def test_plot_span(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span([Vector([sqrt(2), 1, 0]), Vector([0, 1, sqrt(3)])])
        assert (tmp_path / "plot_span.png").exists()

    def test_rejects_complex_components(self):
        animator = Animator3D()
        with pytest.raises(TypeError, match="real components"):
            animator.plot_vectors([Vector([sqrt(-1), 1, 0])])


# ==================== ANIMATE REDUCTION ====================


class TestAnimateReduction3D:
    """Test cases for Animator3D.animate_reduction."""

    def test_rref_of_square_matrix(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        matrix = Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
        animator.animate_reduction(rref(matrix), frames=4, interval=50)
        assert (tmp_path / "animate_reduction.gif").exists()

    def test_augmented_matrix_ref(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        matrix = Matrix([[2, 1, 1, 5], [4, 3, 3, 9], [8, 7, 9, 1]])
        animator.animate_reduction(ref(matrix), frames=4, interval=50)
        assert (tmp_path / "animate_reduction.gif").exists()

    def test_already_reduced_matrix_has_nothing_to_do(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        identity = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        animator.animate_reduction(rref(identity), frames=4, interval=50)
        assert (tmp_path / "animate_reduction.gif").exists()

    def test_rejects_two_row_matrix_and_points_to_2d(self):
        animator = Animator3D()
        with pytest.raises(ValueError, match="Animator2D"):
            animator.animate_reduction(rref(Matrix([[1, 2], [3, 4]])))

    def test_rejects_plain_matrix(self):
        animator = Animator3D()
        with pytest.raises(TypeError, match="Reduction"):
            animator.animate_reduction(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
