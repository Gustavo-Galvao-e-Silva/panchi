# Vector Spaces

A vector space (or subspace) is the set of all linear combinations of a collection of vectors — every vector you can form by scaling and adding the generators together.

In panchi, `VectorSpace` is a **slim container**: it holds the spanning set and validates it, nothing more. The derived structure — basis, rank, membership, complements — is computed on demand by free functions in `panchi.algorithms`:

```python
from panchi import Vector, VectorSpace
from panchi.algorithms import (
    basis,
    rank,
    is_full_rank,
    contains,
    same_subspace,
    orthogonal_complement,
    span,
    standard_basis,
    column_space,
    row_space,
    null_space,
)
```

This mirrors the rest of the library, where the heavy operations (`solve`, `inverse`, `determinant_lu`) are functions rather than methods. It keeps the primitive lightweight and the layering clean.

## Construction

```python
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

`basis(space)` returns a maximal linearly independent subset of the spanning set — the vectors that actually define the subspace, with all redundant generators removed.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
basis(vs)  # [Vector([1, 0]), Vector([0, 1])]
```

The basis is computed by forming a matrix whose columns are the spanning vectors, reducing it to row echelon form, and returning the original vectors at the pivot columns. The result is always drawn from the original spanning set in the original order.

If the entire spanning set is linearly independent, `basis` returns all of them unchanged:

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])
len(basis(vs))  # 3
```

## Rank (dimension of the span)

`rank(space)` is the dimension of the subspace — the number of vectors in a basis, or equivalently the rank of the spanning set. The teachable identity is `rank(vs) == len(basis(vs))`.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1]), Vector([1, 1])])
rank(vs)  # 2
```

The operation is named `rank`, not "dimension", on purpose. A space carries **two** distinct notions of "dimension", and reusing the word invites confusion:

- `rank(vs)` counts **independent directions** — the size of the span.
- `vs.ambient_dims` counts **components** — the *n* in Rⁿ that the space lives inside (the same quantity a single `Vector.dims` reports).

The rank is always less than or equal to the number of spanning vectors:

```python
len(vs)   # 3 — number of generators
rank(vs)  # 2 — independent directions
```

Rank has a geometric meaning: a rank-1 span is a line through the origin, while a rank-2 span fills the whole plane. `Animator2D.plot_span` shades the region a set of vectors covers:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/span_line.png" alt="A single vector and the line through the origin it spans">
  <figcaption>A single vector spans a line — <code>plot_span(Vector([2, 1]))</code>.</figcaption>
</figure>

<figure class="viz" markdown="span">
  <img src="../../assets/viz/span_plane.png" alt="Two independent vectors shading the entire plane they span">
  <figcaption>Two independent vectors span all of R² — <code>plot_span(VectorSpace([Vector([1, 1]), Vector([1, -1])]))</code>.</figcaption>
</figure>

## Ambient dimension

`ambient_dims` is a property on the space itself — it is intrinsic to the container (just the component count) and needs no computation, so it stays on `VectorSpace`. It returns the *n* in Rⁿ, independent of how many generators the space was given.

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
vs.ambient_dims  # 3  — the space lives inside R³
rank(vs)         # 2  — the subspace itself is 2-dimensional
```

## Full rank

`is_full_rank(space)` returns `True` when the subspace spans all of Rⁿ, i.e. when `rank(space) == space.ambient_dims`.

```python
is_full_rank(VectorSpace([Vector([1, 0]), Vector([0, 1])]))        # True  — spans R²
is_full_rank(VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])]))  # False — plane in R³
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
vs[0] = Vector([2, 0])     # valid — same dimension
vs[0] = Vector([1, 0, 0])  # ValueError — wrong number of components
vs[0] = [2, 0]             # TypeError — not a Vector
```

Because `basis` and `rank` are computed on each call (nothing is cached on the container), they always reflect the current generating set after such an edit.

## Equality

Two `VectorSpace` objects are equal if their generating sets contain the same vectors, regardless of order:

```python
v1, v2 = Vector([1, 0]), Vector([0, 1])

VectorSpace([v1, v2]) == VectorSpace([v1, v2])  # True
VectorSpace([v1, v2]) == VectorSpace([v2, v1])  # True — order does not matter
```

## Membership testing

`contains(space, v)` returns `True` if vector `v` lies in the subspace. It solves the system `Bx = v` where B is the matrix of basis vectors, and reports whether the system is consistent.

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
contains(vs, Vector([3, 4]))   # True  — any R² vector is in the full plane

vs2 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
contains(vs2, Vector([0, 0, 1]))  # False — z-axis is out of the xy-plane
```

Passing a non-`Vector` or a vector of the wrong dimension raises an error:

```python
contains(vs, [1, 0])               # TypeError — argument must be a Vector
contains(vs, Vector([1, 0, 0]))    # ValueError — wrong number of components
```

## Subspace equality

The `==` operator checks whether two `VectorSpace` objects were built with the same generators (order-independent). For a mathematical comparison — do the two spaces span the same subspace? — use `same_subspace`:

```python
v1, v2 = Vector([1, 0]), Vector([0, 1])

vs1 = VectorSpace([v1, v2])
vs2 = VectorSpace([v1, v1 + v2])

vs1 == vs2                  # False — different generators
same_subspace(vs1, vs2)    # True  — both span all of R²
```

Two subspaces are equal when they have the same rank and every basis vector of one is contained in the other — which is exactly what `same_subspace` checks:

```python
vs3 = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
vs4 = VectorSpace([Vector([1, 1, 0]), Vector([1, -1, 0])])
same_subspace(vs3, vs4)  # True — both span the xy-plane in R³
```

Comparing spaces of different ambient dimensions raises an error:

```python
vs_r2 = VectorSpace([Vector([1, 0])])
vs_r3 = VectorSpace([Vector([1, 0, 0])])
same_subspace(vs_r2, vs_r3)  # ValueError — different ambient dimensions
```

## Constructing subspaces

Rather than assembling spaces by hand, panchi provides the textbook constructors. `span` is a
readable alias for the constructor, and `standard_basis(n)` builds all of Rⁿ:

```python
span([Vector([1, 0]), Vector([0, 1])])   # VectorSpace(ambient=2, generators=2)
rank(standard_basis(3))                   # 3 — the standard basis spans R³
```

The three fundamental subspaces of a matrix are one call each. The `column_space` (the span of the
columns) and `row_space` both have dimension `rank(A)`; the `null_space` (the kernel) has dimension
`nullity(A)`:

```python
A = Matrix([[1, 2, 3], [4, 5, 6]])

rank(column_space(A))   # 2 == rank(A)
rank(row_space(A))      # 2 == rank(A)
rank(null_space(A))     # 1 == nullity(A)

# every null-space vector is annihilated by A
all(A @ v == Vector([0, 0]) for v in null_space(A))  # True
```

When `A` has full column rank the null space is trivial — represented by the zero vector (rank 0).

## Orthogonal complement

The orthogonal complement of a subspace W is the set of all vectors perpendicular to every vector in W:

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0])])
comp = orthogonal_complement(vs)
print(comp)
# VectorSpace in R^3, spanned by 1 vector:
#   [0, 0, 1]
```

The complement satisfies the rank–nullity relationship: `rank(comp) + rank(vs) == vs.ambient_dims`.

For a line in R²:

```python
vs = VectorSpace([Vector([1, 1])])
comp = orthogonal_complement(vs)
basis(comp)  # [Vector([-1, 1])]  — perpendicular to [1, 1]
```

If the space spans all of Rⁿ, the complement is trivial (the zero vector):

```python
vs = VectorSpace([Vector([1, 0]), Vector([0, 1])])
comp = orthogonal_complement(vs)
rank(comp)  # 0
```

## String representation

`str()` shows the ambient dimension and the spanning set the space was built from:

```python
vs = VectorSpace([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([1, 1, 0])])
print(vs)
# VectorSpace in R^3, spanned by 3 vectors:
#   [1, 0, 0]
#   [0, 1, 0]
#   [1, 1, 0]
```

`repr()` gives a compact summary:

```python
repr(vs)  # 'VectorSpace(ambient=3, generators=3)'
```

To see the reduced basis or the rank instead, call `basis(vs)` and `rank(vs)`.
