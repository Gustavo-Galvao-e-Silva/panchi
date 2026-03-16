# Solving Linear Systems

A linear system Ax = b asks: for a given matrix A and vector b, what vector x satisfies the equation? This is one of the most fundamental questions in linear algebra, and the answer can take three different forms.

## The three outcomes

A linear system always falls into exactly one of three cases:

- **Unique solution** — exactly one x satisfies Ax = b
- **Infinite solutions** — infinitely many x satisfy it (the system is underdetermined)
- **Inconsistent** — no x satisfies it (the system is contradictory)

panchi's `solve()` function identifies which case applies and, when a unique solution exists, returns it.

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
result.original   # the coefficient matrix A
result.target     # the right-hand side vector b
result.status     # 'unique', 'infinite', or 'inconsistent'
result.solution   # the solution Vector, or None if not unique
result.steps      # row operations applied to the augmented matrix [A | b]
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
print(result.solution)  # None
```

When the status is `'infinite'`, `solution` is `None`. A unique solution cannot be extracted without additional constraints. The `steps` attribute still records the full reduction of the augmented matrix, which shows the free variables.

## How it works

`solve()` reduces the augmented matrix [A | b] to RREF. The status is determined by inspecting the result:

- If a pivot appears in the last (b) column, the system is inconsistent.
- If there are fewer pivots than variables, there are free variables and infinite solutions.
- Otherwise, back-substitution yields the unique solution.
