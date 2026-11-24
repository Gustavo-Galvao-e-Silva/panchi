"""
Manual benchmark for panchi's one-sided Jacobi SVD.

Times ``svd`` on seeded, rank-deficient bitmaps of size 32×32, 64×64 and
128×128 and compares the result against NumPy. Not collected by pytest.

Run with::

    python tests/manual/bench_svd.py
"""

from __future__ import annotations

import random
import time

from panchi import Matrix, svd

TARGETS = {32: 0.04, 64: 0.35, 128: 3.5}


def make_bitmap(n: int, seed: int, rectangles: int = 4) -> list[list[float]]:
    rng = random.Random(seed)
    pixels = [[0.0] * n for _ in range(n)]
    for _ in range(rectangles):
        top = rng.randrange(n)
        bottom = rng.randrange(top, n)
        left = rng.randrange(n)
        right = rng.randrange(left, n)
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                pixels[i][j] = 1.0
    return pixels


def main() -> None:
    import numpy as np

    header = (
        f"{'size':>9} {'sweeps':>6} {'conv':>5} {'rank':>4} "
        f"{'time (s)':>9} {'target':>7} {'max σ err':>10} {'max recon err':>13}"
    )
    print(header)
    print("-" * len(header))
    for n, target in TARGETS.items():
        pixels = make_bitmap(n, seed=n)
        matrix = Matrix(pixels)

        start = time.perf_counter()
        result = svd(matrix)
        elapsed = time.perf_counter() - start

        reference = np.linalg.svd(np.array(pixels), compute_uv=False)
        sigma_error = float(np.max(np.abs(np.array(result.sigma) - reference)))

        rebuilt = result.u @ result.s @ result.vt
        recon_error = max(
            abs(rebuilt[i, j] - pixels[i][j]) for i in range(n) for j in range(n)
        )

        status = "ok" if elapsed <= target else "SLOW"
        print(
            f"{n:>4}×{n:<4} {result.sweeps:>6} {str(result.converged):>5} "
            f"{result.rank:>4} {elapsed:>9.3f} {target:>7.2f} "
            f"{sigma_error:>10.1e} {recon_error:>13.1e}  {status}"
        )


if __name__ == "__main__":
    main()
