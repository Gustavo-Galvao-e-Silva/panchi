# Decompositions

A matrix decomposition factors a matrix into simpler pieces that reveal its structure. panchi implements **LU** decomposition with row swaps, **QR** decomposition via Gram-Schmidt, numerical eigenvalue computation via the **QR algorithm** (exact eigenvalues have [their own page](eigenvalues.md)), and the **singular value decomposition (SVD)** via one-sided Jacobi.

## LU Decomposition

LU decomposition factors a square matrix A into a lower triangular matrix L and an upper triangular matrix U. Elimination can land on a zero pivot, so panchi swaps rows when it has to, which introduces a permutation matrix P:

$$P A = L U$$

```python
import panchi as pan
from panchi.algorithms import lu

A = pan.Matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 10]])

decomp = lu(A)

print(decomp.lower)        # L — lower triangular, ones on diagonal
print(decomp.upper)        # U — upper triangular
print(decomp.permutation)  # P — encodes the row swaps made
```

The decomposition satisfies the invariant `P @ A == L @ U`:

```python
assert decomp.permutation @ A == decomp.lower @ decomp.upper
```

## The LUDecomposition object

`lu()` returns an `LUDecomposition` result object:

```python
decomp.original     # the original matrix A
decomp.lower        # L
decomp.upper        # U
decomp.permutation  # P
decomp.steps        # the row operations applied during elimination
```

Printing it gives a readable summary:

```python
print(decomp)
# LU decomposition of 3×3 matrix — 3 steps
#
# P @ A = L @ U
#
# P:
# ...
# L:
# ...
# U:
# ...
```

## When rows are swapped

At each step of elimination, panchi keeps the current pivot unless it is zero. If it is zero, the row is swapped with the first row below that has a non-zero entry in the pivot column. This is the rule you would use by hand, and it is not partial pivoting, which always moves the largest entry into the pivot position to limit rounding error. A small non-zero pivot is kept, so for floating-point matrices L and U can pick up more rounding error than they would with partial pivoting.

The row swaps are recorded in P, so the factorisation relationship remains exact even after swapping. When a swap happens after some elimination steps, the multipliers already written into L for those two rows are swapped too, so each multiplier stays with the row it was computed for. This is the step that is easiest to miss when computing PA = LU by hand.

## What L and U encode

U is the result of Gaussian elimination applied to P @ A — it is the REF of the permuted matrix. L encodes the elimination steps: each entry below the diagonal in column j of L is the negative of the scalar that was used to zero out that row's entry in column j.

This means the steps recorded in `decomp.steps` and the entries of L are two ways of reading the same information.

## QR Decomposition

QR decomposition factors a matrix A into a matrix Q with orthonormal columns and an upper triangular matrix R:

$$A = Q R$$

panchi builds Q by applying the **Gram-Schmidt process** to the columns of A, then recovers R as $R = Q^\top A$.

```python
import panchi as pan
from panchi.algorithms import qr_decomposition

A = pan.Matrix([[1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]])

decomp = qr_decomposition(A)

print(decomp.q)  # Q — orthonormal columns
print(decomp.r)  # R — upper triangular
```

The decomposition satisfies the invariant `A == Q @ R` (up to floating-point rounding, since normalization introduces square roots).

### The QRDecomposition object

`qr_decomposition()` returns a `QRDecomposition` result object:

```python
decomp.original  # the original matrix A
decomp.q         # Q
decomp.r         # R
decomp.steps     # the per-column Gram-Schmidt steps used to build Q
```

Printing it walks through the orthonormalization column by column, then shows the `A = Q @ R` relationship:

```python
print(decomp)
# QR decomposition of 3×3 matrix — 3 steps
#
# Step 1: v0 = [1, 1, 0]
#   orthogonal: [1, 1, 0]
#   normalize -> q0: [0.707..., 0.707..., 0.0]
# ...
#
# A = Q @ R
# ...
```

!!! note
    QR decomposition here is the reduced ("thin") form and assumes the columns of A are linearly independent. Dependent columns produce a zero vector during Gram-Schmidt, which cannot be normalized and raises `ZeroDivisionError`.

## Numerical eigenvalues (QR algorithm)

!!! tip
    For the exact, textbook-style answer on 2×2 and 3×3 matrices, use [`eigen()`](eigenvalues.md). `eigen_qr()` is the numerical approach.

`eigen_qr()` computes the eigenvalues and eigenvectors of a square matrix using the **QR algorithm**: it repeatedly decomposes the matrix and reassembles it as `R @ Q`. For matrices with real eigenvalues this sequence converges to an upper triangular matrix whose diagonal entries are the eigenvalues.

```python
import panchi as pan
from panchi.algorithms import eigen_qr

A = pan.Matrix([[2, 1],
                [1, 2]])

result = eigen_qr(A)

print(result.eigenvalues)   # ≈ [3.0, 1.0]
print(result.eigenvectors)  # unit eigenvectors, paired by index
```

Each eigenvector satisfies `A @ v ≈ λ * v`:

```python
for value, vector in result.pairs:
    assert all(
        abs((A @ vector)[i] - (value * vector)[i]) < 1e-6
        for i in range(vector.dims)
    )
```

Eigenvectors are recovered as the null space of $A - \lambda I$: since a computed eigenvalue is an approximation, panchi solves `(A - λI) x = 0` with a small numerical [tolerance](solving-systems.md#numerical-tolerance) so the near-singular matrix is treated as singular.

### The QREigenResult object

```python
result.original      # the original matrix A
result.eigenvalues   # list of eigenvalues (diagonal of the converged iterate)
result.eigenvectors  # list of eigenvectors, paired with eigenvalues
result.pairs         # list of (eigenvalue, eigenvector) tuples
result.iterations    # number of QR iterations performed
result.converged     # whether the iteration settled within the limit
result.triangular    # the final (near) upper-triangular matrix
```

!!! warning "Real eigenvalues only"
    The unshifted QR algorithm converges for matrices with **real** eigenvalues. Matrices with complex eigenvalues (for example a rotation matrix) or eigenvalues of equal magnitude may not converge. In that case `result.converged` is `False` and `result.eigenvectors` is empty — `eigen_qr()` never raises for this, so you can still inspect the partial result.

    You can adjust `max_iterations` and `tolerance` if needed:

    ```python
    eigen_qr(A, max_iterations=5000, tolerance=1e-14)
    ```

## Singular Value Decomposition (SVD)

The singular value decomposition factors **any** matrix A, square or not, into three pieces:

$$A = U \Sigma V^\top$$

Read from right to left, it says every linear map does three simple things in order:

1. $V^\top$ **rotates** (or reflects) the input space so the important directions line up with the axes.
2. $\Sigma$ **stretches** each axis by a non-negative amount $\sigma_i$, the *singular values*, listed from largest to smallest.
3. $U$ **rotates** the result into place in the output space.

The columns $v_i$ of V are the input directions, the columns $u_i$ of U are where they land, and $A v_i = \sigma_i u_i$. A circle of unit vectors becomes an ellipse whose semi-axes are $\sigma_i u_i$.

panchi computes the **thin** SVD. For an m×n matrix with $k = \min(m, n)$, U is m×k with orthonormal columns, Σ is k×k diagonal, and V is n×k with orthonormal columns.

### How one-sided Jacobi works

panchi uses **one-sided Jacobi**, which works directly on the columns of A. The idea is to keep rotating pairs of columns until every pair is orthogonal.

Take two columns $a_p$ and $a_q$ and compute three dot products:

$$\alpha = \|a_p\|^2, \qquad \beta = \|a_q\|^2, \qquad \gamma = a_p \cdot a_q$$

If $\gamma = 0$ the columns are already orthogonal. Otherwise, a plane rotation by angle θ makes them orthogonal. Its tangent $t = \tan\theta$ is the smaller root of $t^2 + 2\zeta t - 1 = 0$:

$$\zeta = \frac{\beta - \alpha}{2\gamma}, \qquad t = \frac{\operatorname{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}}, \qquad c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = c\,t$$

and the pair is replaced by

$$a_p \leftarrow c\,a_p - s\,a_q, \qquad a_q \leftarrow s\,a_p + c\,a_q$$

Choosing the smaller root keeps $|\theta| \le \pi/4$, so each rotation is as gentle as possible. The same rotation is applied to the columns of a matrix V that starts as the identity, so the working columns W always satisfy $W = A V$.

A **sweep** visits every pair $(p, q)$ with $p < q$ once. Rotating one pair can spoil a pair that was orthogonal earlier, so panchi sweeps again and again. A pair is skipped once $|\gamma| / \sqrt{\alpha\beta}$, the cosine of the angle between the two columns, is below `tolerance`. The algorithm has converged when a full sweep makes no rotations. Then the columns of W are mutually orthogonal and

$$\sigma_j = \|w_j\|, \qquad u_j = \frac{w_j}{\sigma_j}, \qquad A V = W = U \Sigma$$

which is $A = U \Sigma V^\top$ because V is orthogonal.

!!! note "Why not compute AᵀA?"
    A textbook route to the SVD is to find the eigenvalues of $A^\top A$, which are $\sigma_i^2$. Forming $A^\top A$ squares the condition number, so small singular values drown in rounding error, and rank-deficient matrices give a singular $A^\top A$ that eigenvalue iterations handle badly. One-sided Jacobi never forms $A^\top A$. It only ever rotates the columns of A, which is why it stays accurate and works fine on rank-deficient input.

### Computing an SVD

```python
import panchi as pan
from panchi import svd

A = pan.Matrix([[3, 2],
                [2, 3],
                [2, -2]])

result = svd(A)

print(result.sigma)  # [5.0, 2.9999999999999996]
print(result.rank)   # 2
```

The factors multiply back to A, up to floating-point rounding:

```python
rebuilt = result.u @ result.s @ result.vt
assert all(
    abs(rebuilt[i, j] - A[i, j]) < 1e-12
    for i in range(A.rows)
    for j in range(A.cols)
)
```

Each pair $(\sigma_i, u_i, v_i)$ satisfies $A v_i = \sigma_i u_i$:

```python
for sigma, u, v in result.pairs:
    assert all(abs((A @ v)[i] - (sigma * u)[i]) < 1e-12 for i in range(u.dims))
```

Wide matrices (m < n) work too. panchi runs Jacobi on $A^\top$ and swaps U and V at the end, so the steps you see then describe the $A^\top$ run.

Singular vectors are only defined up to sign: flipping both $u_i$ and $v_i$ gives another valid SVD. panchi picks one so results are reproducible: the first non-zero entry of each column of V is positive, and the matching column of U is flipped with it.

### The SVDResult object

`svd()` returns an `SVDResult` result object:

```python
result.original    # the original matrix A (m×n)
result.u           # U — m×k, orthonormal columns
result.sigma       # list of singular values, largest first
result.s           # Σ — k×k diagonal matrix of sigma
result.v           # V — n×k, orthonormal columns
result.vt          # Vᵀ
result.rank        # numerical rank: how many σ are not negligible
result.pairs       # list of (σᵢ, uᵢ, vᵢ) tuples
result.sweeps      # number of Jacobi sweeps performed
result.converged   # whether a sweep finished with no rotations
result.steps       # the recorded sweeps (see below)
result.tolerance   # the tolerance used
```

The numerical rank counts the singular values above `tolerance · σ₁ · max(m, n)`, so rounding noise is not mistaken for a real direction.

Printing it shows each sweep, then the factors:

```python
print(result)
# SVD of 3×2 matrix — 2 sweeps, converged
#
# Sweep 1: 1 rotation, max |cos(angle between columns)| = 4.7e-01
# Sweep 2: 0 rotations, max |cos(angle between columns)| = 0.0e+00
#
# σ = [5, 3]
#
# A = U Σ Vᵀ
#
# U:
# ...
# Σ:
# ...
# Vᵀ:
# ...
```

The second sweep makes no rotations: that is how panchi knows it has converged, and it counts as a sweep.

### Recording levels

The `record` argument controls how much history is kept:

| `record` | `result.steps` |
|----------|----------------|
| `"none"` | `[]` — fastest, nothing kept |
| `"sweeps"` (default) | one `JacobiSweep` per sweep: rotation count and the largest column cosine seen |
| `"rotations"` | every sweep, plus each `JacobiRotation` inside it and a snapshot of the working columns after the sweep |

```python
detailed = svd(A, record="rotations")

sweep = detailed.steps[0]
print(sweep)
# Sweep 1: 1 rotation, max |cos(angle between columns)| = 4.7e-01

rotation = sweep.rotations[0]
print(rotation.p, rotation.q)              # 0 1
print(rotation.alpha, rotation.beta, rotation.gamma)  # 17.0 17.0 8.0
print(rotation.t, rotation.cos, rotation.sin)
# 1.0 0.7071067811865475 0.7071067811865475

print(sweep.matrix)  # the working columns W after sweep 1
# [[0.7071067811865475, 3.5355339059327373],
#  [-0.7071067811865475, 3.5355339059327373],
#  [2.82842712474619, 0.0]]
```

Here both columns of A have length² 17, so $\zeta = 0$ and the rotation is exactly 45°. After it, the two columns of W are orthogonal, with lengths 3 and 5. The descending sort puts 5 first.

!!! warning "Floats and convergence"
    The SVD always returns floats, even for integer or `Fraction` input, because singular values are square roots and usually irrational. If `max_sweeps` (default 60) runs out first, `result.converged` is `False` and the partial result is still returned. `svd()` never raises for this. Invalid input does raise: `TypeError` for a non-`Matrix`, and `ValueError` for NaN or infinite entries, an unknown `record`, `max_sweeps < 1` or `tolerance <= 0`.

### Low-rank approximation

The SVD writes A as a sum of rank-1 pieces, ordered from most to least important:

$$A = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \cdots + \sigma_k u_k v_k^\top$$

Keeping only the first $r$ terms gives $A_r$, the **best** rank-$r$ approximation of A (the Eckart–Young theorem). How good it is can be read off the singular values without building $A_r$:

$$\|A - A_r\|_F = \sqrt{\sigma_{r+1}^2 + \cdots + \sigma_k^2}, \qquad \text{energy}(r) = \frac{\sigma_1^2 + \cdots + \sigma_r^2}{\sigma_1^2 + \cdots + \sigma_k^2}$$

Storing $A_r$ takes $r$ columns of U, $r$ columns of V and $r$ singular values:

$$\text{storage}(r) = r\,(m + n + 1)$$

versus $mn$ numbers for A itself. The rank-$r$ form only saves space while $r(m + n + 1) < mn$:

| Image size | Pixels $mn$ | Numbers per rank | Largest rank that saves space |
|------------|-------------|------------------|-------------------------------|
| 8×8 | 64 | 17 | 3 |
| 32×32 | 1,024 | 65 | 15 |
| 64×64 | 4,096 | 129 | 31 |
| 128×128 | 16,384 | 257 | 63 |

`SVDResult` has a method for each of these quantities. Every `k` runs from 0 to `len(result.sigma)`:

```python
result.component(i)          # σᵢ uᵢ vᵢᵀ, the i-th rank-1 piece (0-indexed)
result.approximation(k)      # A_k, the sum of the first k components
result.error(k)              # ‖A − A_k‖_F
result.relative_error(k)     # error(k) / ‖A‖_F
result.energy(k)             # share of Σσᵢ² kept by the first k terms
result.storage(k)            # k · (m + n + 1)
result.compression_ratio(k)  # storage(k) / (m · n)
result.rank_for_energy(0.9)  # smallest k with energy(k) ≥ 0.9
```

### Example: compressing an 8×8 bitmap

A black-and-white image is a matrix of 0s and 1s. This one is a frame around a square:

```python
import panchi as pan
from panchi import svd

pixels = pan.Matrix([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

image = svd(pixels)

print(image.rank)                                # 3
print([round(value, 4) for value in image.sigma])
# [4.494, 1.6039, 1.1099, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Only four different rows appear, and one of them is all zeros, so the rank is 3. Five of the eight singular values are zero. Now compare what each rank costs and how much of the image it keeps:

```python
for k in range(4):
    print(
        f"k={k}  error={image.relative_error(k):.3f}  "
        f"energy={image.energy(k):.3f}  storage={image.storage(k)}/64"
    )
# k=0  error=1.000  energy=0.000  storage=0/64
# k=1  error=0.398  energy=0.841  storage=17/64
# k=2  error=0.227  energy=0.949  storage=34/64
# k=3  error=0.000  energy=1.000  storage=51/64
```

One rank-1 piece already holds 84% of the energy in about a quarter of the space. Rank 3 rebuilds the image exactly in 51 numbers instead of 64. Here is what the approximations look like, with each value clipped at 0 like a pixel and rounded to one decimal:

```python
def show(matrix):
    for i in range(matrix.rows):
        print(" ".join(f"{max(matrix[i, j], 0.0):.1f}" for j in range(matrix.cols)))

show(image.approximation(1))
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
# 0.0 1.2 0.5 1.0 1.0 0.5 1.2 0.0
# 0.0 0.5 0.2 0.4 0.4 0.2 0.5 0.0
# 0.0 1.0 0.4 0.8 0.8 0.4 1.0 0.0
# 0.0 1.0 0.4 0.8 0.8 0.4 1.0 0.0
# 0.0 0.5 0.2 0.4 0.4 0.2 0.5 0.0
# 0.0 1.2 0.5 1.0 1.0 0.5 1.2 0.0
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

show(image.approximation(2))
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
# 0.0 0.9 0.9 1.1 1.1 0.9 0.9 0.0
# 0.0 0.9 0.0 0.2 0.2 0.0 0.9 0.0
# 0.0 1.1 0.2 0.7 0.7 0.2 1.1 0.0
# 0.0 1.1 0.2 0.7 0.7 0.2 1.1 0.0
# 0.0 0.9 0.0 0.2 0.2 0.0 0.9 0.0
# 0.0 0.9 0.9 1.1 1.1 0.9 0.9 0.0
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

show(image.approximation(3))
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
# 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0
# 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
# 0.0 1.0 0.0 1.0 1.0 0.0 1.0 0.0
# 0.0 1.0 0.0 1.0 1.0 0.0 1.0 0.0
# 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
# 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0
# 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```

Rank 1 is a blurry blob, the outer product of one row profile and one column profile. Rank 2 already shows the frame and the dark gap. Rank 3 is exact. To see the same story as pictures, pass the result to [`plot_compression`](visualizations.md#svd-image-compression).

!!! tip "Keep images small"
    panchi is pure Python, and the cost of one sweep grows roughly as $n^3$ for an n×n image. On a laptop, images up to 64×64 finish in well under a second, while a detailed 128×128 image can take a couple of seconds. Shared notebooks such as Google Colab are slower, so stick to 64×64 or smaller there.
