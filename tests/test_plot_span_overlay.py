"""Tests for the vectors, gaps and view options of plot_span."""

from fractions import Fraction
from math import isclose

import matplotlib.pyplot as plt
import pytest
from mpl_toolkits.mplot3d import proj3d

from panchi import Vector, VectorSpace
from panchi.visualizations import Animator2D, Animator3D
from panchi.visualizations.backends.geometry import edge_view, plane_in_cube
from panchi.visualizations.backends.matplotlib_base import _MatplotlibBackendBase

U = Vector([1, 0, 0])
V = Vector([0, 1, 0])
Y = Vector([3, 4, 5])
P = Vector([3, 4, 0])


@pytest.fixture
def figure(monkeypatch):
    """Capture the finished figure instead of showing or saving it."""
    captured = {}

    def capture(self, fig, name):
        captured["fig"] = fig

    monkeypatch.setattr(_MatplotlibBackendBase, "_finalize_figure", capture)
    yield captured
    plt.close("all")


def dashed_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "--"]


class TestOverlay3D:
    def test_saves_a_file(self, tmp_path):
        animator = Animator3D(save_path=tmp_path)
        animator.plot_span([U, V], vectors=[Y, P], gaps=[(Y, P)])
        assert (tmp_path / "plot_span.png").exists()

    def test_gap_joins_its_two_points(self, figure):
        Animator3D().plot_span([U, V], gaps=[(Y, P)])
        lines = dashed_lines(figure["fig"].axes[0])
        assert len(lines) == 1
        xs, ys, zs = lines[0].get_data_3d()
        assert list(zs) == [5, 0]
        assert list(xs) == [3, 3]
        assert list(ys) == [4, 4]

    def test_axes_make_room_for_vectors(self, figure):
        Animator3D().plot_span([U, V], vectors=[Vector([0, 0, 50])])
        assert figure["fig"].axes[0].get_zlim()[1] >= 50

    def test_axes_make_room_for_gaps(self, figure):
        Animator3D().plot_span([U, V], gaps=[(Vector([0, 0, 50]), P)])
        assert figure["fig"].axes[0].get_zlim()[1] >= 50

    def test_labels_are_drawn(self, figure):
        Animator3D().plot_span([U, V], vectors=[Y, P], vector_labels=["y", None])
        texts = [text.get_text() for text in figure["fig"].axes[0].texts]
        assert "y" in texts

    def test_accepts_a_vector_space(self, figure):
        Animator3D().plot_span(VectorSpace([U, V]), vectors=[Y])
        assert figure["fig"].axes

    def test_accepts_several_spans(self, figure):
        Animator3D().plot_span([[U], [V]], vectors=[Y], gaps=[(Y, P)])
        assert len(dashed_lines(figure["fig"].axes[0])) == 1

    def test_exact_vectors(self, figure):
        half = Vector([Fraction(1, 2), Fraction(1, 3), Fraction(1, 5)])
        Animator3D().plot_span([U, V], vectors=[half], gaps=[(half, U)])
        assert figure["fig"].axes

    def test_colors_cycle_when_too_few_are_given(self, figure):
        Animator3D().plot_span([U, V], vectors=[Y, P], vector_colors=["red"])
        assert figure["fig"].axes

    def test_without_overlay_nothing_extra_is_drawn(self, figure):
        Animator3D().plot_span([U, V])
        assert dashed_lines(figure["fig"].axes[0]) == []


class TestOverlay2D:
    def test_saves_a_file(self, tmp_path):
        x_axis, y, p = Vector([1, 0]), Vector([3, 4]), Vector([3, 0])
        animator = Animator2D(save_path=tmp_path)
        animator.plot_span([x_axis], vectors=[y, p], gaps=[(y, p)])
        assert (tmp_path / "plot_span.png").exists()

    def test_gap_is_drawn(self, figure):
        x_axis, y, p = Vector([1, 0]), Vector([3, 4]), Vector([3, 0])
        Animator2D().plot_span([x_axis], gaps=[(y, p)])
        assert len(dashed_lines(figure["fig"].axes[0])) == 1

    def test_labels_are_drawn(self, figure):
        Animator2D().plot_span(
            [Vector([1, 0])], vectors=[Vector([3, 4])], vector_labels=["y"]
        )
        texts = [text.get_text() for text in figure["fig"].axes[0].texts]
        assert "y" in texts

    def test_rejects_view(self):
        with pytest.raises(ValueError, match="Animator2D can't use it"):
            Animator2D().plot_span([Vector([1, 0])], view=(30, 45))


class TestOverlayValidation:
    def test_vectors_must_be_a_list(self):
        with pytest.raises(TypeError, match="vectors must be a list"):
            Animator3D().plot_span([U, V], vectors=Y)

    def test_vectors_must_hold_vectors(self):
        with pytest.raises(TypeError, match="only Vector objects"):
            Animator3D().plot_span([U, V], vectors=[[1, 2, 3]])

    def test_vectors_must_match_the_dimension(self):
        with pytest.raises(ValueError, match="Only 3D vectors"):
            Animator3D().plot_span([U, V], vectors=[Vector([1, 2])])

    def test_one_label_per_vector(self):
        with pytest.raises(ValueError, match="got 1 labels for 2 vectors"):
            Animator3D().plot_span([U, V], vectors=[Y, P], vector_labels=["y"])

    def test_gaps_must_be_a_list(self):
        with pytest.raises(TypeError, match="gaps must be a list"):
            Animator3D().plot_span([U, V], gaps=(Y, P))

    @pytest.mark.parametrize("gap", [(Y,), (Y, P, U), [Y, P], (Y, "p")])
    def test_each_gap_is_a_pair_of_vectors(self, gap):
        with pytest.raises(TypeError, match="pair of vectors"):
            Animator3D().plot_span([U, V], gaps=[gap])

    def test_gaps_must_match_the_dimension(self):
        with pytest.raises(ValueError, match="Only 3D vectors"):
            Animator3D().plot_span([U, V], gaps=[(Y, Vector([1, 2]))])


class TestView:
    def test_elevation_and_azimuth(self, figure):
        Animator3D().plot_span([U, V], view=(20, 45))
        ax = figure["fig"].axes[0]
        assert (ax.elev, ax.azim) == (20, 45)

    def test_roll(self, figure):
        Animator3D().plot_span([U, V], view=(20, 45, 10))
        assert figure["fig"].axes[0].roll == 10

    def test_default_view_is_unchanged(self, figure):
        Animator3D().plot_span([U, V])
        ax = figure["fig"].axes[0]
        assert (ax.elev, ax.azim) == (30, -60)

    def test_rejects_a_list(self):
        with pytest.raises(TypeError, match="tuple of angles"):
            Animator3D().plot_span([U, V], view=[20, 45])

    def test_rejects_an_unknown_name(self):
        with pytest.raises(TypeError, match="tuple of angles"):
            Animator3D().plot_span([U, V], view="top")

    def test_rejects_the_wrong_number_of_angles(self):
        with pytest.raises(ValueError, match="Got 1 values"):
            Animator3D().plot_span([U, V], view=(20,))

    @pytest.mark.parametrize("angle", ["20", True, None])
    def test_rejects_angles_that_are_not_numbers(self, angle):
        with pytest.raises(TypeError, match="must be numbers"):
            Animator3D().plot_span([U, V], view=(angle, 45))


class TestEdgeView:
    def test_needs_a_plane(self):
        with pytest.raises(ValueError, match="single span of dimension 2"):
            Animator3D().plot_span([U], view="edge")

    def test_full_space_is_not_a_plane(self):
        with pytest.raises(ValueError, match="single span of dimension 2"):
            Animator3D().plot_span([U, V, Vector([0, 0, 1])], view="edge")

    def test_needs_a_single_span(self):
        with pytest.raises(ValueError, match="single span of dimension 2"):
            Animator3D().plot_span([[U, V], [Y]], view="edge")

    def test_gap_runs_straight_up_the_screen(self, figure):
        tilted = [Vector([1, 1, 1]), Vector([1, 2, 4])]
        y = Vector([30.0, 10.0, 15.0])
        p = Vector([29.0, 19.0, 10.0])
        normal = (2.0, -3.0, 1.0)
        Animator3D().plot_span(tilted, vectors=[y, p], view="edge")
        ax = figure["fig"].axes[0]
        horizontal, vertical = screen_direction(ax, normal)
        assert abs(horizontal) < 1e-9
        assert vertical > 0

    def test_camera_stays_on_the_side_of_the_vectors(self, figure):
        below = Vector([1, 1, -5])
        Animator3D().plot_span([U, V], vectors=[below], view="edge")
        _, vertical = screen_direction(figure["fig"].axes[0], (0, 0, -1))
        assert vertical > 0

    def test_works_with_nothing_on_top(self, figure):
        Animator3D().plot_span([U, V], view="edge")
        _, vertical = screen_direction(figure["fig"].axes[0], (0, 0, 1))
        assert vertical > 0


def screen_direction(ax, direction):
    """Where ``direction`` points on screen, relative to the origin."""
    matrix = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(0, 0, 0, matrix)
    x1, y1, _ = proj3d.proj_transform(*direction, matrix)
    return x1 - x0, y1 - y0


class TestPlaneInCube:
    def test_flat_plane_is_a_square(self):
        polygon = plane_in_cube((1, 0, 0), (0, 1, 0), 2)
        assert sorted(polygon) == [
            (-2.0, -2.0, 0.0),
            (-2.0, 2.0, 0.0),
            (2.0, -2.0, 0.0),
            (2.0, 2.0, 0.0),
        ]

    def test_points_lie_in_the_plane_and_the_cube(self):
        u, v = (1.0, 1.0, 1.0), (0.5, 1.1, 2.4)
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        polygon = plane_in_cube(u, v, 3)
        assert len(polygon) >= 3
        for point in polygon:
            assert isclose(
                sum(a * b for a, b in zip(point, normal, strict=True)),
                0,
                abs_tol=1e-9,
            )
            assert all(abs(c) <= 3 + 1e-9 for c in point)

    def test_shape_ignores_how_parallel_the_spanning_vectors_are(self):
        wide = plane_in_cube((1, 0, 0), (0, 1, 0), 2)
        narrow = plane_in_cube((1, 0, 0), (1, 1e-6, 0), 2)
        assert sorted(wide) == sorted(narrow)

    def test_plane_through_cube_corners(self):
        polygon = plane_in_cube((1, 1, 0), (0, 0, 1), 1)
        assert len(polygon) >= 4


class TestEdgeViewAngles:
    def test_returns_elevation_azimuth_roll(self):
        assert len(edge_view((0, 0, 1), (1, 0, 0))) == 3

    def test_flat_plane_needs_no_roll(self):
        elevation, _, roll = edge_view((0, 0, 1), (1, 0, 0))
        assert isclose(elevation, 15)
        assert isclose(roll, 0, abs_tol=1e-9)

    def test_height_sets_the_elevation(self):
        elevation, _, _ = edge_view((0, 0, 1), (1, 0, 0), height=40)
        assert isclose(elevation, 40)
