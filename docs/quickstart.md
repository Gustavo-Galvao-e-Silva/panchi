# Quickstart

A feel for panchi in a few minutes. For deeper explanations, follow the links into the [User Guide](user-guide/vectors.md). For those that wish to contribute, take a look at [Contributing](contributing.md)

## Installation

```bash
pip install panchi
```

## Vectors

```python
import panchi as pan

v = pan.Vector([3, 4])
print(v.magnitude)    # 5.0
print(v.normalize())  # [0.6, 0.8]

u = pan.Vector([1, 2, 3])
w = pan.Vector([4, 5, 6])
print(pan.dot(u, w))    # 32
print(pan.cross(u, w))  # [-3, 6, -3]
```

Vectors support the arithmetic you would expect: `+`, `-`, scalar `*` and `/`, and unary `-` for negation.

## Matrices

```python
A = pan.Matrix([[1, 2], [3, 4]])
B = pan.Matrix([[5, 6], [7, 8]])

print(A @ B)          # matrix multiplication
print(A.T)            # transpose
print(A.trace)        # 5
print(A.determinant)  # -2
```

`@` is matrix multiplication, `*` is scalar multiplication — consistent with standard Python convention.

## Factory Functions

```python
I = pan.identity(3)
Z = pan.zero_matrix(2, 3)
D = pan.diagonal([1, 2, 3])
R = pan.rotation_matrix_2d(3.14159 / 2)
```

## Exact Arithmetic

panchi supports exact fractions — write them as strings and they stay exact through every operation:

```python
v = pan.Vector(["1/3", "2/3", "1/2"])
A = pan.exact_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
inv = pan.inverse(A).inverse
print(A @ inv)  # exact identity matrix, no floating-point drift
```

See the [Exact Arithmetic guide](user-guide/exact-arithmetic.md) for the full story.

## Algorithms

Algorithms return result objects that carry both the answer and the work behind it.

```python
from panchi.algorithms import rref

A = pan.Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])

reduction = rref(A)
print(reduction.result)  # the RREF matrix
print(reduction.rank)    # 3
print(reduction)         # full step-by-step walkthrough
```

The step-by-step output is panchi's most distinctive feature — every algorithm lets you see exactly what happened, not just the final answer.

## Visualizations

panchi can visualize vectors, transformations, and spans right out of the box.

```python
from panchi.visualizations import Animator2D

animator = Animator2D()

# Plot vectors
animator.plot_vectors(pan.Vector([3, 2]), pan.Vector([-1, 3]), labels=["v1", "v2"])

# Animate a linear transformation (grid morph)
animator.animate_transform(pan.Matrix([[0, -1], [1, 0]]))
```

For video output, use the manim backend (`pip install panchi[manim]`):

```python
animator = Animator2D(backend="manim", save_path="./videos")
animator.animate_transform(pan.Matrix([[1, 1], [0, 1]]))
```

See the [Visualizations guide](user-guide/visualizations.md) for the full feature set.

---

From here, explore the [User Guide](user-guide/vectors.md) for the concepts and math behind each part of the library.
