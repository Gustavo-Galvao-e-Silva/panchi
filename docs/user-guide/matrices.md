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

## Arithmetic

```python
A = pan.Matrix([[1, 2], [3, 4]])
B = pan.Matrix([[5, 6], [7, 8]])

print(A + B)   # element-wise addition
print(A - B)   # element-wise subtraction
print(2 * A)   # scalar multiplication
print(-A)      # negation
```

## Matrix multiplication

Matrix multiplication uses the `@` operator, following Python convention:

```python
print(A @ B)   # [[19, 22], [43, 50]]
```

`*` is reserved for scalar multiplication. Using `*` between two matrices raises a `TypeError` — this is intentional, since it avoids a common source of confusion.

Multiplying a matrix by a vector also uses `@` and returns a `Vector`:

```python
v = pan.Vector([1, 0])
print(A @ v)   # [1, 3]
```

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

print(A.trace)        # 5       — sum of diagonal (square only)
print(A.determinant)  # -2      — cofactor expansion (square only)
print(A.is_square)    # True
```

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

`row()` and `col()` mirror the mathematical operators Row(A) and Col(A), returning the row and column vectors of a matrix respectively.

```python
A = pan.Matrix([[1, 2, 3], [4, 5, 6]])

A.row()      # [Vector([1, 2, 3]), Vector([4, 5, 6])]
A.col()      # [Vector([1, 4]), Vector([2, 5]), Vector([3, 6])]

A.row()[0]   # Vector([1, 2, 3])
A.col()[1]   # Vector([2, 5])
```

Both methods return copies — modifying the result does not affect the original matrix.

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
