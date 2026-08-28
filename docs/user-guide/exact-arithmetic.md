# Exact Arithmetic

When working with linear algebra by hand, fractions like $\frac{1}{3}$ stay exact. But computers default to floating-point, which introduces rounding errors — $0.333...$ instead of $\frac{1}{3}$, or $0.9999...$ instead of $1$.

panchi supports exact rational arithmetic using Python's `fractions.Fraction`. This means $\frac{1}{3}$ stays $\frac{1}{3}$ through every operation — RREF, solve, inverse, determinant — with zero rounding.

## Writing fractions

The simplest way to use fractions is to write them as strings:

```python
import panchi as pan

v = pan.Vector(["1/3", "2/3", "1/2"])
print(v)  # [1/3, 2/3, 1/2]
```

You can mix fractions with regular numbers:

```python
v = pan.Vector(["1/3", 2, 0])
print(v)  # [1/3, 2, 0]
```

Matrices work the same way:

```python
A = pan.Matrix([["1/2", "1/3"],
                ["1/4", "1/5"]])
print(A)
# [[1/2, 1/3],
#  [1/4, 1/5]]
```

Negative fractions use a minus sign in the numerator:

```python
v = pan.Vector(["-1/3", "5/7"])
```

## Factory functions

If you want all elements to be exact (even integers), use the factory functions:

```python
v = pan.exact_vector([1, 2, 3])
A = pan.exact_matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 10]])
```

These convert every element to `Fraction`, ensuring that all subsequent arithmetic stays exact. This is especially useful when division would normally produce floats:

```python
v = pan.exact_vector([1, 2, 3])
print(v / 3)  # [1/3, 2/3, 1]   — not [0.333..., 0.666..., 1.0]
```

## Using Fraction directly

You can also use Python's `Fraction` type directly:

```python
from fractions import Fraction

v = pan.Vector([Fraction(1, 3), Fraction(2, 3)])
```

panchi re-exports `Fraction` for convenience:

```python
from panchi import Fraction
```

## Exact algorithms

All panchi algorithms work with fractions out of the box. The results are exact — no floating-point drift.

### RREF

```python
from panchi.algorithms import rref

A = pan.exact_matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 10]])

reduction = rref(A)
print(reduction.result)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

### Inverse

```python
A = pan.exact_matrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 10]])

result = pan.inverse(A)
print(result.inverse)
# [[-2/3, -4/3, 1],
#  [-2/3, 11/3, -2],
#  [1, -2, 1]]

# Verify: A @ A⁻¹ = I (exactly, not approximately)
print(A @ result.inverse)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

### Solve

```python
A = pan.exact_matrix([[2, 1],
                      [5, 3]])
b = pan.exact_vector([1, 2])

result = pan.solve(A, b)
print(result.solution)  # [1, -1]
```

### Determinant

```python
A = pan.exact_matrix([[1, 2], [3, 4]])
print(pan.determinant(A))  # -2
```

## When to use exact arithmetic

**Use exact arithmetic when:**

- You're learning linear algebra and want to see clean results
- You need to verify that $A A^{-1} = I$ exactly
- You're working through textbook problems with rational answers
- Floating-point drift is obscuring the mathematical structure

**Stick with floats when:**

- You need `normalize()` or `magnitude` (these involve square roots, which are irrational)
- You're working with measured/experimental data
- Performance matters for large matrices

## Backward compatibility

Existing code is unaffected. `Vector([1, 2, 3])` with integers stays as integers. Fractions are opt-in — you choose to use them by passing string fractions or using `exact_vector()` / `exact_matrix()`.

```python
# These still work exactly as before
v = pan.Vector([1, 2, 3])
print(type(v[0]))  # <class 'int'>

w = pan.Vector([1.0, 2.0, 3.0])
print(type(w[0]))  # <class 'float'>
```
