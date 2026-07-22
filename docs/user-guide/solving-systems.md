# Solving Linear Systems

A linear system Ax = b asks: for a given matrix A and vector b, what vector x satisfies the equation? This is one of the most fundamental questions in linear algebra, and the answer can take three different forms.

## The three outcomes

A linear system always falls into exactly one of three cases:

- **Unique solution** — exactly one x satisfies Ax = b
- **Infinite solutions** — infinitely many x satisfy it (the system is underdetermined)
- **Inconsistent** — no x satisfies it (the system is contradictory)

panchi's `solve()` function identifies which case applies, returns the unique solution when one exists, and provides the general solution (particular solution + null space) for underdetermined systems.

## Basic usage

```python
import panchi as pan
from panchi.algorithms import solve

A = pan.Matrix([[2, 1],
                [5, 3]])
b = pan.Vector([1, 2])

result = solve(A, b)

print(result.status)    # 'unique'
print(result.solution)  # [1.0, -1.0]
```

Verify the solution:

```python
assert A @ result.solution == b
```

## The Solution object

`solve()` returns a `Solution` object:

```python
result.original    # the coefficient matrix A
result.target      # the right-hand side vector b
result.status      # 'unique', 'infinite', or 'inconsistent'
result.solution    # the solution Vector, or None if not unique
result.steps       # row operations applied to the augmented matrix [A | b]
result.particular  # particular solution Vector (infinite case only)
result.null_space  # VectorSpace basis for the null space (infinite case only)
```

## Inconsistent systems

```python
A = pan.Matrix([[1, 2],
                [2, 4]])
b = pan.Vector([1, 3])  # 2×row 1 gives [2, 4] but b[1] = 3 ≠ 2

result = solve(A, b)
print(result.status)    # 'inconsistent'
print(result.solution)  # None
```

## Underdetermined systems

```python
A = pan.Matrix([[1, 2, 3],
                [4, 5, 6]])
b = pan.Vector([7, 8])

result = solve(A, b)
print(result.status)    # 'infinite'
```

When the status is `'infinite'`, the general solution is expressed as a particular solution plus the null space of A:

```python
print(result.particular)  # a Vector satisfying A @ x == b
print(result.null_space)  # VectorSpace — basis for the null space of A

# Verify
assert A @ result.particular == b
for v in result.null_space:
    assert A @ v == pan.Vector([0, 0])
```

Printing the result shows the full parametric form:

```python
print(result)
# Solution to 2×3 system — infinite
#
# x = [-6.33..., 6.66..., 0] + t·[1.0, -2.0, 1]
```

For homogeneous systems (b = 0), the particular solution is the zero vector and is omitted from the display. Systems with multiple free variables use `s, t` (for two) or `t1, t2, ...` (for three or more) as parameters.

## How it works

`solve()` reduces the augmented matrix [A | b] to RREF. The status is determined by inspecting the result:

- If a pivot appears in the last (b) column, the system is inconsistent.
- If there are fewer pivots than variables, there are free variables and infinite solutions.
- Otherwise, back-substitution yields the unique solution.

## Numerical tolerance

By default `solve()` uses **exact** comparisons: a pivot counts as zero only if it is exactly `0`. This is the right behaviour for integer and `Fraction` inputs, where the arithmetic is exact.

Floating-point matrices are trickier. A matrix that is *mathematically* singular can end up with a tiny non-zero residual pivot (like `1e-15`) after elimination, so an exact solve would wrongly report it as full rank. The optional `tolerance` parameter treats any pivot at or below that magnitude as zero:

```python
from panchi.algorithms import solve

# Only approximately singular — the second row is a hair off the first.
A = pan.Matrix([[1.0, 1.0],
                [1.0, 1.0 + 1e-12]])

solve(A, pan.Vector([0.0, 0.0]))                 # status 'unique'  (exact)
solve(A, pan.Vector([0.0, 0.0]), tolerance=1e-6) # status 'infinite' → null space
```

The default of `tolerance=0.0` reproduces the exact behaviour exactly, so existing code is unaffected. The same `tolerance` parameter is available on `ref()` and `rref()`.

This is the mechanism [`eigen()`](decompositions.md#eigenvalues-and-eigenvectors) uses to recover eigenvectors as the null space of an approximately-singular `A - λI`.
