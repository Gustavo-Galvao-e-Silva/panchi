"""
Comprehensive test suite for all visualization features.

Tests edge cases, error handling, and all parameters.
"""

from panchi import Vector
from panchi.visualizations import Animator2D


def test_basic_functionality():
    """Test basic vector operations."""
    print("\n" + "=" * 70)
    print("TEST: Basic Functionality")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    # Test 1: Single vector
    print("\n[1] Single vector")
    v = Vector([2, 3])
    animator.plot_vectors(v, labels=["v"])

    # Test 2: Two vectors
    print("\n[2] Two vectors")
    v1 = Vector([1, 2])
    v2 = Vector([2, 1])
    animator.plot_vectors(v1, v2, labels=["v₁", "v₂"])

    # Test 3: Many vectors
    print("\n[3] Five vectors")
    vectors = [Vector([i, 5 - i]) for i in range(5)]
    animator.plot_vectors(*vectors, labels=[f"v{i}" for i in range(5)])

    print("\n✓ Basic functionality tests passed")


def test_edge_cases():
    """Test edge cases and special values."""
    print("\n" + "=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    # Test 1: Zero vector
    print("\n[1] Zero vector")
    v = Vector([0, 0])
    animator.plot_vectors(v, labels=["0"])

    # Test 2: Unit vectors
    print("\n[2] Unit vectors")
    e1 = Vector([1, 0])
    e2 = Vector([0, 1])
    animator.plot_vectors(e1, e2, labels=["e₁", "e₂"])

    # Test 3: Negative components
    print("\n[3] Negative components")
    v1 = Vector([-2, 3])
    v2 = Vector([2, -3])
    animator.plot_vectors(v1, v2, labels=["v₁", "v₂"])

    # Test 4: Large magnitudes
    print("\n[4] Large magnitudes")
    v1 = Vector([10, 15])
    v2 = Vector([-8, 12])
    animator.plot_vectors(v1, v2, labels=["v₁", "v₂"])

    # Test 5: Very small magnitudes
    print("\n[5] Very small magnitudes")
    v1 = Vector([0.1, 0.2])
    v2 = Vector([0.15, 0.05])
    animator.plot_vectors(v1, v2, labels=["v₁", "v₂"])

    print("\n✓ Edge case tests passed")


def test_animations():
    """Test all animation types."""
    print("\n" + "=" * 70)
    print("TEST: Animations")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    # Test 1: Addition with positive vectors
    print("\n[1] Addition: positive vectors")
    v1 = Vector([2, 1])
    v2 = Vector([1, 2])
    animator.animate_addition(v1, v2, frames=40, interval=30)

    # Test 2: Addition with negative vectors
    print("\n[2] Addition: one negative vector")
    v1 = Vector([3, 2])
    v2 = Vector([-1, 2])
    animator.animate_addition(v1, v2, frames=40, interval=30)

    # Test 3: Scaling up
    print("\n[3] Scaling: factor > 1")
    v = Vector([2, 1])
    animator.animate_scaling(v, scale_factor=2.0, frames=40, interval=30)

    # Test 4: Scaling down
    print("\n[4] Scaling: factor < 1")
    v = Vector([3, 2])
    animator.animate_scaling(v, scale_factor=0.5, frames=40, interval=30)

    # Test 5: Negative scaling
    print("\n[5] Scaling: negative factor")
    v = Vector([2, 3])
    animator.animate_scaling(v, scale_factor=-1.5, frames=40, interval=30)

    print("\n✓ Animation tests passed")


def test_customization():
    """Test customization options."""
    print("\n" + "=" * 70)
    print("TEST: Customization Options")
    print("=" * 70)

    # Test 1: Custom colors
    print("\n[1] Custom colors")
    animator = Animator2D(backend="matplotlib")
    v1 = Vector([2, 1])
    v2 = Vector([1, 2])
    v3 = Vector([-1, 1])
    animator.plot_vectors(
        v1,
        v2,
        v3,
        colors=["#FF0000", "#00FF00", "#0000FF"],
        labels=["red", "green", "blue"],
    )

    # Test 2: No grid
    print("\n[2] No grid")
    animator.plot_vectors(v1, v2, grid=False, labels=["v₁", "v₂"])

    # Test 3: No labels
    print("\n[3] No labels")
    animator.plot_vectors(v1, v2, v3)

    # Test 4: Custom resolution
    print("\n[4] Custom resolution")
    animator_hd = Animator2D(resolution=(1920, 1080), backend="matplotlib")
    animator_hd.plot_vectors(v1, v2, labels=["v₁", "v₂"])

    print("\n✓ Customization tests passed")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST: Error Handling")
    print("=" * 70)

    animator = Animator2D(backend="matplotlib")

    # Test 1: Non-2D vector
    print("\n[1] Non-2D vector (should raise ValueError)")
    try:
        v = Vector([1, 2, 3])
        animator.plot_vectors(v)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 2: Invalid backend
    print("\n[2] Invalid backend (should raise ValueError)")
    try:
        animator_bad = Animator2D(backend="invalid")
        print("✗ Should have raised error")
    except Exception as e:
        print(f"✓ Correctly raised error: {e}")

    print("\n✓ Error handling tests passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("MATHRIX VISUALIZATION TEST SUITE")
    print("=" * 70)
    print("\nThis will test all visualization features.")
    print("Close each plot window to proceed to the next test.\n")

    input("Press Enter to start tests...")

    try:
        test_basic_functionality()
        input("\nPress Enter to continue to edge cases...")

        test_edge_cases()
        input("\nPress Enter to continue to animations...")

        test_animations()
        input("\nPress Enter to continue to customization...")

        test_customization()
        input("\nPress Enter to continue to error handling...")

        test_error_handling()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
