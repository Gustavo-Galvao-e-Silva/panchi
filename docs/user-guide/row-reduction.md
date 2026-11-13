# Row Reduction

Row reduction is the process of applying elementary row operations to bring a matrix into a standard simplified form. It is the foundation of most of linear algebra: solving systems, computing rank, finding inverses, and performing decompositions all rely on it.

panchi implements two forms — REF and RREF — and returns a result object that records every step.

## Row Echelon Form (REF)

A matrix is in Row Echelon Form when every row's leading entry (pivot) is to the right of the pivot in the row above, and all rows of zeros are at the bottom. This is the result of forward elimination — Gaussian elimination.

```python
import panchi as pan
from panchi.algorithms import ref

A = pan.Matrix([[2, 1, -1],
                [-3, -1, 2],
                [-2, 1, 2]])

reduction = ref(A)
print(reduction.result)
```

## Reduced Row Echelon Form (RREF)

RREF goes further: each pivot is scaled to 1, and all other entries in the pivot column are eliminated. This is Gauss-Jordan elimination, and the result is unique for any given matrix.

```python
from panchi.algorithms import rref

reduction = rref(A)
print(reduction.result)
```

## The Reduction object

Both `ref()` and `rref()` return a `Reduction` object. This object carries more than just the answer:

```python
reduction = rref(A)

print(reduction.result)   # the reduced matrix
print(reduction.rank)     # number of pivot columns
print(reduction.nullity)  # cols - rank (dimension of null space)
print(reduction.pivots)   # list of (row, col) pivot positions
print(reduction.steps)    # ordered list of RowOperation objects
```

Printing the `Reduction` object produces a full step-by-step walkthrough:

```python
print(reduction)
# RREF of 3×3 matrix — 9 steps, rank 3
#
# Step 1: R1 -> R1 + (1.5) * R0
# [[2, 1, -1],
#  [0.0, 0.5, 0.5],
#  [-2, 1, 2]]
# ...
```

This is the primary educational feature of row reduction in panchi. Rather than just returning the answer, it shows the sequence of operations that produced it — the same sequence a student would write out by hand.

## Rank and nullity

Rank is the number of pivot columns found during reduction. Nullity is the dimension of the null space, given by the rank-nullity theorem:

$$\text{rank} + \text{nullity} = \text{number of columns}$$

```python
A = pan.Matrix([[1, 2, 3], [2, 4, 6]])  # rank-deficient

reduction = rref(A)
print(reduction.rank)     # 1
print(reduction.nullity)  # 2
```

## Using reduction steps directly

The `steps` attribute is a list of `RowOperation` objects. You can replay them, inspect them, or apply them to other matrices:

```python
for step in reduction.steps:
    print(step)  # prints in standard row operation notation
```
