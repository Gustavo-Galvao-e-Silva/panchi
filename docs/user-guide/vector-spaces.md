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

Two `VectorSpace` objects are equal if their generating sets contain the same vectors, regardless of order:

```python
v1, v2 = Vector([1, 0]), Vector([0, 1])

VectorSpace([v1, v2]) == VectorSpace([v1, v2])  # True
VectorSpace([v1, v2]) == VectorSpace([v2, v1])  # True — order does not matter
```

## Ambient dimension

`ambient_dims` returns the dimension of the ambient space R^n — the number of components in each vector, independent of how many generators the space was given.

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
vs.ambient_dims  # 3  — the space lives inside R³
vs.dims          # 2  — the subspace itself is 2-dimensional
```

## Full rank

`is_full_rank` returns `True` when the subspace spans all of R^n, i.e. when `dims == ambient_dims`.

```python
VectorSpace([Vector([1, 0]), Vector([0, 1])]).is_full_rank         # True  — spans R²
VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])]).is_full_rank   # False — plane in R³
```

## Membership testing

`contains(v)` returns `True` if vector `v` lies in the subspace. It solves the system `Bx = v` where B is the matrix of basis vectors, and reports whether the system is consistent.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
vs.contains(Vector([3, 4]))   # True  — any R² vector is in the full plane

vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
vs2.contains(Vector([0, 0, 1]))  # False — z-axis is out of the xy-plane
```

Passing a non-`Vector` or a vector of the wrong dimension raises an error:

```python
vs.contains([1, 0])               # TypeError — argument must be a Vector
vs.contains(Vector([1, 0, 0]))    # ValueError — wrong number of components
```

## Subspace equality

The `==` operator checks whether two `VectorSpace` objects were built with the same generators (order-independent). For a mathematical comparison — do the two spaces span the same subspace? — use `same_subspace`:

```python
v1, v2 = Vector([1, 0]), Vector([0, 1])

vs1 = VectorSpace([v1, v2])
vs2 = VectorSpace([v1, v1 + v2])

vs1 == vs2              # False — different generators
vs1.same_subspace(vs2)  # True  — both span all of R²
```

Two subspaces are equal when they have the same dimension and every basis vector of one is contained in the other. `same_subspace` checks this using the existing `contains` method:

```python
vs3 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
vs4 = VectorSpace([Vector([1, 1, 0]), Vector([1, -1, 0])])
vs3.same_subspace(vs4)  # True — both span the xy-plane in R³
```

Comparing spaces of different ambient dimensions raises an error:

```python
vs_r2 = VectorSpace([Vector([1, 0])])
vs_r3 = VectorSpace([Vector([1, 0, 0])])
vs_r2.same_subspace(vs_r3)  # ValueError — different ambient dimensions
```

## Orthogonal complement

The orthogonal complement of a subspace W is the set of all vectors perpendicular to every vector in W. Use `orthogonal_complement` from `panchi.algorithms`:

```python
from panchi.algorithms import orthogonal_complement

vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
comp = orthogonal_complement(vs)
print(comp)
# VectorSpace in R^3, dimension 1
# Basis:
#   [0, 0, 1]
```

The complement satisfies the rank-nullity relationship: `comp.dims + vs.dims == vs.ambient_dims`.

For a line in R²:

```python
vs = VectorSpace([Vector([1, 1])])
comp = orthogonal_complement(vs)
comp.basis  # [Vector([-1, 1])]  — perpendicular to [1, 1]
```

If the space spans all of R^n, the complement is trivial (the zero vector):

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
comp = orthogonal_complement(vs)
comp.dims  # 0
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
