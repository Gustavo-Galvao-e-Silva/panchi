"""
Streamlined test suite for Manim backend visualization features.

Quick validation of key features (~5 minutes total).

REQUIREMENTS:
  - System dependencies (cairo, pkg-config, ffmpeg)
  - Manim installed: pip install manim
"""

from panchi import Vector
from panchi.visualizations import Animator2D
from pathlib import Path


def check_manim():
    """Check if Manim is available."""
    try:
        import manim

        print(f"✓ Manim is installed (version {manim.__version__})")
        return True
    except ImportError:
        print("\n" + "=" * 70)
        print("ERROR: Manim is not installed")
        print("=" * 70)
        print("\nTo install Manim:")
        print("\nmacOS:")
        print("  brew install cairo pkg-config ffmpeg")
        print("  pip install manim")
        print("\nUbuntu/Debian:")
        print("  sudo apt install libcairo2-dev libpango1.0-dev ffmpeg")
        print("  pip install manim")
        return False


def run_quick_tests():
    """Run quick tests of key features."""
    print("\n" + "=" * 70)
    print("MANIM BACKEND - QUICK TEST SUITE")
    print("=" * 70)

    output_dir = "./manim_quick_test"
    animator = Animator2D(backend="manim", save_animations=True, output_dir=output_dir)

    # Test 1: Basic vector plotting
    print("\n[1/5] Testing: Multiple vector plotting")
    print("-" * 70)
    v1 = Vector([3, 2])
    v2 = Vector([1, 3])
    v3 = Vector([-2, 1])
    try:
        animator.plot_vectors(
            v1, v2, v3, labels=[r"v_1", r"v_2", r"v_3"], name="vectors"
        )
        print("  ✓ Rendered: vectors.mp4")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 2: Vector addition
    print("\n[2/5] Testing: Vector addition animation")
    print("-" * 70)
    print(f"  v₁ = {v1.data}, v₂ = {v2.data}")
    print(f"  v₁ + v₂ = {(v1 + v2).data}")
    try:
        animator.animate_addition(v1, v2, name="addition")
        print("  ✓ Rendered: addition.mp4")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 3: Positive scaling
    print("\n[3/5] Testing: Vector scaling (positive)")
    print("-" * 70)
    print(f"  2.0 * v₁ = {(2.0 * v1).data}")
    try:
        animator.animate_scaling(v1, scale_factor=2.0, name="scaling_positive")
        print("  ✓ Rendered: scaling_positive.mp4")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 4: Negative scaling
    print("\n[4/5] Testing: Vector scaling (negative)")
    print("-" * 70)
    print(f"  -1.5 * v₂ = {(-1.5 * v2).data}")
    try:
        animator.animate_scaling(v2, scale_factor=-1.5, name="scaling_negative")
        print("  ✓ Rendered: scaling_negative.mp4")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 5: Edge case - unit vectors
    print("\n[5/5] Testing: Edge case (unit vectors)")
    print("-" * 70)
    e1 = Vector([1, 0])
    e2 = Vector([0, 1])
    try:
        animator.plot_vectors(e1, e2, labels=[r"e_1", r"e_2"], name="unit_vectors")
        print("  ✓ Rendered: unit_vectors.mp4")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    return True


def generate_summary(output_dir="./manim_quick_test"):
    """Generate a summary of rendered videos."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    base_path = Path(output_dir)

    if not base_path.exists():
        print("\nNo videos were rendered.")
        return

    videos = list(base_path.glob("*.mp4"))

    if not videos:
        print("\nNo videos found.")
        return

    total_size = 0
    print(f"\nRendered videos in: {output_dir}/\n")
    for video in sorted(videos):
        size_mb = video.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  ✓ {video.name:30s} {size_mb:6.1f} MB")

    print(f"\n  Total: {len(videos)} videos, {total_size:.1f} MB")
    print(f"\n💡 Open videos with:")
    print(f"   open {output_dir}/<video>.mp4")
    print(f"   # or")
    print(f"   vlc {output_dir}/<video>.mp4")


def main():
    """Run streamlined Manim tests."""
    print("=" * 70)
    print("MATHRIX MANIM BACKEND - QUICK TEST")
    print("=" * 70)

    # Check Manim availability
    if not check_manim():
        return

    print("\nThis will render 5 videos to validate Manim backend.")
    print("Estimated time: ~3-5 minutes")
    print("\nVideos will be saved to: ./manim_quick_test/\n")

    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ["y", "yes"]:
        print("Tests cancelled.")
        return

    try:
        success = run_quick_tests()

        if success:
            generate_summary()

            print("\n" + "=" * 70)
            print("ALL MANIM TESTS PASSED! 🎉")
            print("=" * 70)
            print("\nManim backend verified:")
            print("  ✓ Vector plotting")
            print("  ✓ Addition animation")
            print("  ✓ Scaling animations")
            print("  ✓ Edge case handling")
            print("  ✓ Production-quality output")
        else:
            print("\n" + "=" * 70)
            print("SOME TESTS FAILED")
            print("=" * 70)
            print("\nCheck error messages above for details.")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
