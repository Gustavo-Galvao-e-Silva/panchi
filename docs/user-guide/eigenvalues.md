# Eigenvalues and Eigenvectors

An eigenvector of a square matrix $A$ is a non-zero vector $v$ that $A$ only stretches: $Av = \lambda v$. The scale factor $\lambda$ is its eigenvalue.

`eigen()` finds them **exactly**, the way you would by hand on a 2×2 or 3×3 matrix:

1. Form the characteristic polynomial $\det(A - \lambda I)$.
2. Solve it. Its roots are the eigenvalues.
3. For each eigenvalue, row-reduce $A - \lambda I$. Its null space is the eigenspace.

```python
import panchi as pan

A = pan.Matrix([[2, 1],
                [1, 2]])

result = pan.eigen(A)
print(result)
```

```text
Eigenvalues of 2×2 matrix
det(A - λI) = λ² - 4λ + 3 = (λ - 1)(λ - 3)

λ = 1  (algebraic multiplicity 1, geometric multiplicity 1)
  v = [-1, 1]
λ = 3  (algebraic multiplicity 1, geometric multiplicity 1)
  v = [1, 1]

Diagonalizable: yes
```

Every answer is exact, so `A @ v == λ * v` holds with `==`, not just approximately:

```python
for value, vector in result.pairs:
    assert A @ vector == value * vector
```

## Irrational and complex eigenvalues

Eigenvalues that aren't rational come out as `QuadraticNumber`s, which are exact numbers of the form $a + b\sqrt{d}$. A negative $d$ gives a complex number, so $\sqrt{-1}$ is $i$.

```python
fibonacci = pan.Matrix([[1, 1],
                        [1, 0]])
for value in pan.eigen(fibonacci).eigenvalues:
    print(value)
# (1 - √5)/2
# (1 + √5)/2   — the golden ratio, exactly

rotation = pan.Matrix([[0, -1],
                       [1, 0]])
result = pan.eigen(rotation)
print(result)
```

```text
Eigenvalues of 2×2 matrix
det(A - λI) = λ² + 1 = (λ + i)(λ - i)

λ = -i  (algebraic multiplicity 1, geometric multiplicity 1)
  v = [-i, 1]
λ = i  (algebraic multiplicity 1, geometric multiplicity 1)
  v = [i, 1]

Diagonalizable: yes
```

A 90° rotation leaves no real direction fixed, so it has no real eigenvalues. Over the complex numbers it has two, and `result.has_complex_eigenvalues` is `True`.

## 3×3 matrices

For a 3×3 matrix the characteristic polynomial is a cubic. `eigen()` uses the **rational root theorem** to find a rational root, divides it out with synthetic division, and solves the leftover quadratic with the quadratic formula:

```python
A = pan.Matrix([[1, 0, 0],
                [0, 1, -2],
                [0, 1, 3]])
print(pan.eigen(A))
```

```text
Eigenvalues of 3×3 matrix
det(A - λI) = -λ³ + 5λ² - 9λ + 5 = -(λ - 1)(λ - (2 - i))(λ - (2 + i))
...
```

!!! warning "Cubics with no rational root"
    If the characteristic cubic has no rational root, its roots involve cube roots and cannot be written as $a + b\sqrt{d}$. `eigen()` then raises a `ValueError` explaining why. The characteristic polynomial is still available from `characteristic_polynomial(A)`, and `eigen_qr(A)` gives numerical approximations. Matrices larger than 3×3 also need [`eigen_qr()`](decompositions.md#numerical-eigenvalues-qr-algorithm).

## Multiplicities and diagonalization

An eigenvalue's **algebraic multiplicity** is how many times it is a root of the characteristic polynomial. Its **geometric multiplicity** is the dimension of its eigenspace. When the two match for every eigenvalue, the eigenvectors form a basis and $A = PDP^{-1}$:

```python
P, D = pan.eigen(pan.Matrix([[2, 1], [1, 2]])).diagonalization()
print(D)
# [[1, 0],
#  [0, 3]]
```

When an eigenvalue has too few eigenvectors, the matrix is **defective**:

```python
shear = pan.Matrix([[1, 1],
                    [0, 1]])
result = pan.eigen(shear)
result.algebraic_multiplicities  # [2]
result.geometric_multiplicities  # [1]
result.is_diagonalizable         # False
result.diagonalization()         # ValueError: not enough independent eigenvectors
```

## The EigenResult object

```python
result.original                   # the matrix A
result.characteristic_polynomial  # det(A - λI) as a Polynomial
result.eigenvalues                # distinct eigenvalues (real ascending, then complex)
result.algebraic_multiplicities   # paired with eigenvalues
result.geometric_multiplicities   # dimension of each eigenspace
result.eigenspaces                # list of VectorSpace, paired with eigenvalues
result.pairs                      # (eigenvalue, eigenvector) for every basis vector
result.is_diagonalizable
result.has_complex_eigenvalues
result.diagonalization()          # (P, D)
```

In a notebook, the result renders as LaTeX: the characteristic polynomial, its factored form, and each eigenspace as a span. See [Rich Display](rich-display.md).

## The building blocks

The pieces `eigen()` uses are available on their own in `panchi.symbolic`:

```python
from panchi.symbolic import Polynomial, QuadraticNumber, sqrt, exact_roots

p = pan.characteristic_polynomial(A)
p.rational_roots()          # rational root theorem
p.divide_by_root(1)         # synthetic division → (quotient, remainder)
exact_roots(p)              # [(root, multiplicity), ...]

phi = (1 + sqrt(5)) / 2
phi * phi == phi + 1        # True, exactly
```

See [Exact Arithmetic](exact-arithmetic.md#square-roots-and-complex-numbers) for more on `QuadraticNumber`.
