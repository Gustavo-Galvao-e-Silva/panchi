"""
Interactive test suite for the Manim visualization backend.

Renders videos to ./manim_test_output/. Requires manim installation.

REQUIREMENTS:
  - System dependencies (cairo, pkg-config, ffmpeg)
  - Manim installed: pip install panchi[manim]
"""

from pathlib import Path

from panchi import Matrix, Vector, VectorSpace
from panchi.visualizations import Animator2D


def check_manim():
    """Check if Manim is available."""
    try:
        import manim

        print(f"Manim is installed (version {manim.__version__})")
        return True
    except ImportError:
        print("\n" + "=" * 70)
        print("ERROR: Manim is not installed")
        print("=" * 70)
        print("\nTo install:")
        print("  pip install panchi[manim]")
        print("\nmacOS also needs:")
        print("  brew install cairo pkg-config ffmpeg")
        return False


def run_tests():
    """Run all manim visualization tests."""
    output_dir = "./manim_test_output"
    animator = Animator2D(backend="manim", save_path=output_dir)

    print("\n[1/7] Multiple vector plotting")
    v1 = Vector([3, 2])
    v2 = Vector([1, 3])
    v3 = Vector([-2, 1])
    try:
        animator.plot_vectors(
            v1, v2, v3, labels=[r"v_1", r"v_2", r"v_3"], name="vectors"
        )
        print("  OK: vectors rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[2/7] Vector addition")
    try:
        animator.animate_addition(v1, v2, name="addition")
        print("  OK: addition rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[3/7] Vector scaling (positive)")
    try:
        animator.animate_scaling(v1, scale_factor=2.0, name="scaling_positive")
        print("  OK: scaling rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[4/7] Vector scaling (negative)")
    try:
        animator.animate_scaling(v2, scale_factor=-1.5, name="scaling_negative")
        print("  OK: negative scaling rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[5/7] Linear transformation (90-degree rotation)")
    try:
        animator.animate_transform(Matrix([[0, -1], [1, 0]]), name="rotation")
        print("  OK: rotation rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[6/7] Linear transformation (shear)")
    try:
        animator.animate_transform(Matrix([[1, 1], [0, 1]]), name="shear")
        print("  OK: shear rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n[7/7] Span visualization")
    try:
        space = VectorSpace([Vector([1, 0]), Vector([0, 1])])
        animator.plot_span(space, labels=[r"e_1", r"e_2"], name="span_r2")
        print("  OK: span rendered")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    return True


def main():
    """Run Manim backend tests."""
    print("=" * 70)
    print("PANCHI MANIM VISUALIZATION TEST SUITE")
    print("=" * 70)

    if not check_manim():
        return

    print("\nThis will render 7 videos to ./manim_test_output/")
    print("Estimated time: ~5-10 minutes\n")

    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        print("Tests cancelled.")
        return

    try:
        success = run_tests()

        if success:
            output = Path("./manim_test_output")
            videos = list(output.rglob("*.mp4"))
            print("\n" + "=" * 70)
            print(f"ALL TESTS PASSED! {len(videos)} videos rendered.")
            print("=" * 70)
            print(f"\nOutput: {output.resolve()}")
        else:
            print("\n" + "=" * 70)
            print("SOME TESTS FAILED")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
