# Vector Spaces

A vector space (or subspace) is the set of all linear combinations of a collection of vectors — every vector you can form by scaling and adding the generators together. In panchi, `VectorSpace` takes a list of spanning vectors and computes the structure of that subspace: its basis and its dimension.

## Construction

```python
from panchi.primitives.vector_space import VectorSpace
from panchi.primitives import Vector

v1 = Vector([1, 0, 0])
v2 = Vector([0, 1, 0])
v3 = Vector([1, 1, 0])  # linearly dependent on v1 and v2

vs = VectorSpace([v1, v2, v3])
```

The spanning set is stored as provided. All vectors must be `Vector` instances with the same number of components. Passing a non-`Vector`, an empty list, or vectors of mixed dimensions raises an informative error:

```python
VectorSpace([])                             # ValueError — no subspace defined
VectorSpace([Vector([1, 2]), [3, 4]])       # TypeError — list is not a Vector
VectorSpace([Vector([1, 2]), Vector([1])])  # ValueError — dimension mismatch
```

## Basis

The `basis` property returns a maximal linearly independent subset of the spanning set — the vectors that actually define the subspace, with all redundant generators removed.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
vs.basis  # [Vector([1, 0]), Vector([0, 1])]
```

The basis is computed by forming a matrix whose columns are the spanning vectors, reducing it to row echelon form, and returning the original vectors at the pivot columns. The result is always drawn from the original spanning set in the original order.

If the entire spanning set is linearly independent, `basis` returns all of them unchanged:

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])
len(vs.basis)  # 3
```

## Dimension

`dims` is the dimension of the subspace — the number of vectors in the basis, or equivalently, the rank of the spanning set.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
vs.dims  # 2
```

The dimension is always less than or equal to the number of spanning vectors:

```python
len(vs)   # 3 — number of generators
vs.dims   # 2 — dimension of the span
```

## Indexing and iteration

The generating set is indexable and iterable, just like `Vector` and `Matrix`:

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])

vs[0]    # Vector([1, 0])
vs[-1]   # Vector([0, 1])
len(vs)  # 2

for v in vs:
    print(v)
# [1, 0]
# [0, 1]
```

You can replace a spanning vector by index. The replacement must be a `Vector` with the same number of components as the rest of the space:

```python
vs[0] = Vector([2, 0])   # valid — same dimension
vs[0] = Vector([1, 0, 0])  # ValueError — wrong number of components
vs[0] = [2, 0]             # TypeError — not a Vector
```

Note that replacing a vector changes the generating set. The `basis` recomputes on the next access to reflect the updated spans.

## Equality

Two `VectorSpace` objects are equal if their generating sets contain the same vectors in the same order:

```python
v1, v2 = Vector([1, 0]), Vector([0, 1])

VectorSpace([v1, v2]) == VectorSpace([v1, v2])  # True
VectorSpace([v1, v2]) == VectorSpace([v2, v1])  # False — order matters
```

## String representation

`str()` shows the ambient dimension, the computed dimension, and the basis:

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
print(vs)
# VectorSpace in R^3, dimension 2
# Basis:
#   [1, 0, 0]
#   [0, 1, 0]
```

`repr()` gives a compact summary:

```python
repr(vs)  # 'VectorSpace(ambient=3, dim=2, generators=3)'
```
