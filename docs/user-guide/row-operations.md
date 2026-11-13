# Row Operations

Every row reduction algorithm — Gaussian elimination, Gauss-Jordan, LU decomposition — is built from three elementary row operations. In panchi, these are first-class objects, not implementation details.

## The three operations

**RowSwap** — swap two rows:

$$R_a \leftrightarrow R_b$$

```python
from panchi.algorithms import RowSwap
import panchi as pan

m = pan.Matrix([[1, 2], [3, 4], [5, 6]])
op = RowSwap(0, 2)
print(op.apply(m))
# [[5, 6],
#  [3, 4],
#  [1, 2]]
```

**RowScale** — multiply a row by a non-zero scalar:

$$R_i \to c \cdot R_i$$

```python
from panchi.algorithms import RowScale

op = RowScale(1, 3)
print(op.apply(m))
# [[1, 2],
#  [9, 12],
#  [5, 6]]
```

**RowAdd** — add a scalar multiple of one row to another:

$$R_{\text{target}} \to R_{\text{target}} + c \cdot R_{\text{source}}$$

```python
from panchi.algorithms import RowAdd

m = pan.Matrix([[1, 2], [3, 4]])
op = RowAdd(target=1, source=0, scalar=-3)
print(op.apply(m))
# [[1, 2],
#  [0, -2]]
```

This is the core step of Gaussian elimination: choosing a scalar that zeroes out an entry below the pivot.

## Elementary matrices

Each row operation corresponds to an elementary matrix — the identity matrix with the operation applied to it. Multiplying a matrix on the left by an elementary matrix performs the row operation:

```python
op = RowSwap(0, 1)
E = op.elementary_matrix(2)
print(E)
# [[0, 1],
#  [1, 0]]

assert op.apply(m) == E @ m
```

This connection between row operations and matrix multiplication is a core idea in linear algebra, and panchi makes it explicit.

## Inverses

Every row operation has an inverse that undoes it:

```python
op = RowAdd(target=1, source=0, scalar=-3)
inv = op.inverse()
# RowAdd(target=1, source=0, scalar=3)

assert inv.apply(op.apply(m)) == m
```

Inverses are used internally when building the L matrix in LU decomposition.

## String representation

Operations print in standard mathematical notation:

```python
print(RowSwap(0, 2))            # R0 <-> R2
print(RowScale(1, 3))           # R1 -> 3 * R1
print(RowAdd(1, 0, -3))         # R1 -> R1 + (-3) * R0
```

This makes the step-by-step output from `ref()` and `rref()` readable without any translation.
