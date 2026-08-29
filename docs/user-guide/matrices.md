# Matrices

A matrix is a rectangular grid of numbers. In panchi, `Matrix` represents an m×n matrix and supports the arithmetic and structural operations you would expect from a mathematical matrix.

## Construction

```python
import panchi as pan

A = pan.Matrix([[1, 2, 3],
                [4, 5, 6]])

print(A.shape)  # (2, 3)
print(A.rows)   # 2
print(A.cols)   # 3
```

Every row must have the same number of columns. Inconsistent rows raise a `ValueError`.

Matrices accept fractions as strings:

```python
A = pan.Matrix([["1/2", "1/3"],
                ["1/4", "1/5"]])
```

See [Exact Arithmetic](exact-arithmetic.md) for more on working with rational numbers.

## Arithmetic

```python
A = pan.Matrix([[1, 2], [3, 4]])
B = pan.Matrix([[5, 6], [7, 8]])

print(A + B)   # element-wise addition
print(A - B)   # element-wise subtraction
print(2 * A)   # scalar multiplication
print(A * 2)   # same as above
print(-A)      # negation
```

## Matrix multiplication

Matrix multiplication uses the `@` operator, following Python convention:

```python
print(A @ B)   # [[19, 22], [43, 50]]
```

`@` is used for matrix–matrix and matrix–vector multiplication. `*` is reserved for scalar multiplication only.

Multiplying a matrix by a vector also uses `@` and returns a `Vector`:

```python
v = pan.Vector([1, 0])
print(A @ v)   # [1, 3]
```

Geometrically, multiplying by a matrix *is* a transformation of space. The matrix below stretches the plane by 2 along x and 3 along y — every point moves, and the basis vectors land on the matrix's columns:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/matrix_scale.gif" alt="A coordinate grid stretching by 2 horizontally and 3 vertically under a diagonal matrix">
  <figcaption><code>Matrix([[2, 0], [0, 3]])</code> acting on the plane. See <a href="../transformations/">Linear Transformations</a> for more.</figcaption>
</figure>

## Matrix powers

```python
print(A ** 2)  # A @ A
print(A ** 0)  # identity matrix of matching size
```

Powers are only defined for square matrices.

## Transpose

```python
print(A.T)          # shorthand property
print(A.transpose()) # equivalent method
```

## Properties

```python
A = pan.Matrix([[1, 2], [3, 4]])

print(A.trace)      # 5       — sum of diagonal (square only)
print(A.is_square)  # True
```

The determinant is a free function (like `solve`, `inverse`, and `determinant_lu`), not a
property — it is an algorithm, so it lives in `panchi.algorithms` rather than on the matrix:

```python
print(pan.determinant(A))  # -2  — cofactor expansion, exact (square only)
```

## Rank, invertibility, and symmetry

Rank and its companions are free functions too. `rank` counts the linearly independent rows
(equivalently columns) of a matrix; `nullity` is the dimension of its null space; together they
satisfy the rank–nullity theorem, `rank(A) + nullity(A) == A.cols`:

```python
A = pan.Matrix([[1, 2, 3], [4, 5, 6]])

print(pan.rank(A))       # 2
print(pan.nullity(A))    # 1   — and 2 + 1 == A.cols
```

`is_invertible` and `is_symmetric` answer the two most common structural questions. `is_invertible`
is decided by rank (square and full rank), which avoids the O(n!) cofactor determinant:

```python
print(pan.is_invertible(pan.Matrix([[1, 2], [3, 4]])))  # True
print(pan.is_invertible(pan.Matrix([[1, 2], [2, 4]])))  # False — singular
print(pan.is_symmetric(pan.Matrix([[1, 2], [2, 1]])))   # True
```

`rank` is deliberately one function for both matrices and vector spaces — a subspace's rank *is*
the rank of its generating matrix, so `rank(pan.column_space(A)) == rank(A)`.

## Identity matrices

Every matrix has a left and right identity — the square identity matrices of the appropriate size for multiplication on each side:

```python
A = pan.Matrix([[1, 2, 3], [4, 5, 6]])  # 2×3

print(A.left_identity)   # 2×2 identity
print(A.right_identity)  # 3×3 identity

# These satisfy:
assert A.left_identity @ A == A
assert A @ A.right_identity == A
```

## Row and column access

`row_vectors` and `col_vectors` mirror the mathematical operators Row(A) and Col(A), returning the row and column vectors of a matrix respectively.

```python
A = pan.Matrix([[1, 2, 3], [4, 5, 6]])

A.row_vectors      # [Vector([1, 2, 3]), Vector([4, 5, 6])]
A.col_vectors      # [Vector([1, 4]), Vector([2, 5]), Vector([3, 6])]

A.row_vectors[0]   # Vector([1, 2, 3])
A.col_vectors[1]   # Vector([2, 5])
```

Both properties return copies — modifying the result does not affect the original matrix.

The columns of a matrix are vectors in their own right. Here are the two columns of `A = [[1, 2], [3, 4]]` — `col 1 = (1, 3)` and `col 2 = (2, 4)` — drawn as arrows:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/matrix_columns.png" alt="The two column vectors of a 2x2 matrix drawn as arrows from the origin">
  <figcaption>The column vectors of <code>A = [[1, 2], [3, 4]]</code>, plotted with <code>Animator2D.plot_vectors</code>.</figcaption>
</figure>

## Factory functions

```python
pan.identity(3)            # 3×3 identity
pan.zero_matrix(2, 3)      # 2×3 matrix of zeros
pan.one_matrix(2, 3)       # 2×3 matrix of ones
pan.diagonal([1, 2, 3])    # 3×3 diagonal matrix
pan.random_matrix(3, 3)    # random entries
```

## Conversion and copying

```python
A.to_list()  # returns a 2D list copy of the data
A.copy()     # returns an independent Matrix copy
```
