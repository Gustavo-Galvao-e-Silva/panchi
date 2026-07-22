# Decompositions

A matrix decomposition factors a matrix into simpler pieces that reveal its structure. panchi implements **LU** decomposition with partial pivoting, **QR** decomposition via Gram-Schmidt, and eigenvalue/eigenvector computation via the **QR algorithm**.

## LU Decomposition

LU decomposition factors a square matrix A into a lower triangular matrix L and an upper triangular matrix U. Because row swaps are often necessary for numerical stability, panchi uses **partial pivoting**, which introduces a permutation matrix P:

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

## What partial pivoting means

At each step of elimination, partial pivoting swaps rows so that the entry with the largest absolute value in the current pivot column becomes the pivot. This avoids division by small numbers and keeps the decomposition numerically well-behaved.

The row swaps are recorded in P, so the factorisation relationship remains exact even after pivoting.

## What L and U encode

U is the result of Gaussian elimination applied to P @ A — it is the REF of the permuted matrix. L encodes the elimination steps: each entry below the diagonal in column j of L is the scalar that was used to zero out that row's entry in column j.

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

## Eigenvalues and eigenvectors

`eigen()` computes the eigenvalues and eigenvectors of a square matrix using the **QR algorithm**: it repeatedly decomposes the matrix and reassembles it as `R @ Q`. For matrices with real eigenvalues this sequence converges to an upper triangular matrix whose diagonal entries are the eigenvalues.

```python
import panchi as pan
from panchi.algorithms import eigen

A = pan.Matrix([[2, 1],
                [1, 2]])

result = eigen(A)

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

### The EigenResult object

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
    The unshifted QR algorithm converges for matrices with **real** eigenvalues. Matrices with complex eigenvalues (for example a rotation matrix) or eigenvalues of equal magnitude may not converge. In that case `result.converged` is `False` and `result.eigenvectors` is empty — `eigen()` never raises for this, so you can still inspect the partial result.

    You can adjust `max_iterations` and `tolerance` if needed:

    ```python
    eigen(A, max_iterations=5000, tolerance=1e-14)
    ```
