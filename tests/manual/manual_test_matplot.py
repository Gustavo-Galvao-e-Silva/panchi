"""
Interactive test suite for the matplotlib visualization backend.

Run manually and close each plot window to proceed to the next test.
"""

from panchi import Matrix, Vector, VectorSpace
from panchi.visualizations import Animator2D


def test_basic_functionality():
    """Test basic vector operations."""
    print("\n" + "=" * 70)
    print("TEST: Basic Functionality")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    print("\n[1] Single vector")
    v = Vector([2, 3])
    animator.plot_vectors([v], labels=["v"])

    print("\n[2] Two vectors")
    v1 = Vector([1, 2])
    v2 = Vector([2, 1])
    animator.plot_vectors([v1, v2], labels=["v1", "v2"])

    print("\n[3] Five vectors")
    vectors = [Vector([i, 5 - i]) for i in range(5)]
    animator.plot_vectors(vectors, labels=[f"v{i}" for i in range(5)])

    print("\n[4] Custom colors")
    v1 = Vector([2, 1])
    v2 = Vector([1, 2])
    v3 = Vector([-1, 1])
    animator.plot_vectors(
        [v1, v2, v3],
        colors=["#FF0000", "#00FF00", "#0000FF"],
        labels=["red", "green", "blue"],
    )

    print("\n[5] No grid")
    animator.plot_vectors([v1, v2], grid=False, labels=["v1", "v2"])


def test_animations():
    """Test all animation types."""
    print("\n" + "=" * 70)
    print("TEST: Animations")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    print("\n[1] Addition: positive vectors")
    animator.animate_addition(Vector([2, 1]), Vector([1, 2]))

    print("\n[2] Addition: one negative vector")
    animator.animate_addition(Vector([3, 2]), Vector([-1, 2]))

    print("\n[3] Scaling: factor > 1")
    animator.animate_scaling(Vector([2, 1]), scale_factor=2.0)

    print("\n[4] Scaling: factor < 1")
    animator.animate_scaling(Vector([3, 2]), scale_factor=0.5)

    print("\n[5] Scaling: negative factor")
    animator.animate_scaling(Vector([2, 3]), scale_factor=-1.5)


def test_transforms():
    """Test linear transformation animations."""
    print("\n" + "=" * 70)
    print("TEST: Linear Transformations")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    print("\n[1] 90-degree rotation")
    animator.animate_transform(Matrix([[0, -1], [1, 0]]))

    print("\n[2] Horizontal shear")
    animator.animate_transform(Matrix([[1, 1], [0, 1]]))

    print("\n[3] Scaling transform (2x in both directions)")
    animator.animate_transform(Matrix([[2, 0], [0, 2]]))

    print("\n[4] Reflection across x-axis")
    animator.animate_transform(Matrix([[1, 0], [0, -1]]))

    print("\n[5] Singular matrix (projection onto x-axis)")
    animator.animate_transform(Matrix([[1, 0], [0, 0]]))


def test_spans():
    """Test span visualization."""
    print("\n" + "=" * 70)
    print("TEST: Span Visualization")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    print("\n[1] 1D span (single vector)")
    animator.plot_span([Vector([1, 2])])

    print("\n[2] 1D span (linearly dependent vectors)")
    animator.plot_span([Vector([1, 2]), Vector([2, 4])])

    print("\n[3] 2D span (two independent vectors)")
    animator.plot_span([Vector([1, 0]), Vector([0, 1])])

    print("\n[4] 2D span from VectorSpace")
    space = VectorSpace([Vector([1, 1]), Vector([1, -1])])
    animator.plot_span(space, labels=["v1", "v2"])


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST: Error Handling")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    print("\n[1] Non-2D vector (should raise ValueError)")
    try:
        animator.plot_vectors([Vector([1, 2, 3])])
        print("  FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"  OK: {e}")

    print("\n[2] Non-2x2 matrix (should raise ValueError)")
    try:
        animator.animate_transform(Matrix([[1, 2, 3], [4, 5, 6]]))
        print("  FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"  OK: {e}")

    print("\n[3] Invalid backend (should raise ValueError)")
    try:
        Animator2D(backend="invalid")
        print("  FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"  OK: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("PANCHI MATPLOTLIB VISUALIZATION TEST SUITE")
    print("=" * 70)
    print("\nClose each plot window to proceed to the next test.\n")

    input("Press Enter to start tests...")

    try:
        test_basic_functionality()
        input("\nPress Enter to continue to animations...")

        test_animations()
        input("\nPress Enter to continue to transforms...")

        test_transforms()
        input("\nPress Enter to continue to spans...")

        test_spans()
        input("\nPress Enter to continue to error handling...")

        test_error_handling()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
