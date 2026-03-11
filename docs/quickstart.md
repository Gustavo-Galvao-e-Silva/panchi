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
print(A.determinant)  # -2.0
```

`@` is matrix multiplication, `*` is scalar multiplication — consistent with standard Python convention.

## Factory Functions

```python
I = pan.identity(3)
Z = pan.zero_matrix(2, 3)
D = pan.diagonal([1, 2, 3])
R = pan.rotation_matrix_2d(3.14159 / 2)
```

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

---

From here, explore the [User Guide](../user-guide/vectors.md) for the concepts and math behind each part of the library.
