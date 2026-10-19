# Determinants

`determinant(A)` returns the number. When you want to see *how* that number is reached, panchi
offers the two methods taught by hand, each returning a result object that shows its working, plus
the adjugate and Cramer's rule built on top of them. Everything stays exact for `int` and `Fraction`
entries.

```python
import panchi as pan

A = pan.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
print(pan.determinant(A))  # 18
```

## Cofactor expansion

`cofactor_expansion` expands along one row or column. Each entry $a_{ij}$ is multiplied by its
cofactor $(-1)^{i+j}\det(M_{ij})$, where $M_{ij}$ is the matrix left after deleting row $i$ and
column $j$:

```python
result = pan.cofactor_expansion(A)
print(result)
```

```
Cofactor expansion of 3×3 matrix along row 0
  (+1) · 2 · 11 = 22
  (-1) · 1 · 4 = -4
  (+1) · 0 · 1 = 0
det(A) = 18
```

Every term is kept, including the zero ones, so you can see why a row full of zeros is the
cheapest one to expand along. Any row or column gives the same answer:

```python
pan.cofactor_expansion(A, along="column", index=2).determinant  # 18
```

Each entry of `result.terms` is a `CofactorTerm` with `row`, `col`, `entry`, `minor`,
`minor_determinant`, `sign`, `cofactor` and `value`. To expand a minor further, call
`cofactor_expansion(term.minor)`.

## Row reduction

`determinant_by_reduction` reduces the matrix to row echelon form $U$ with `ref()`. That uses only
row swaps and row additions: an addition leaves the determinant unchanged and a swap flips its
sign, so

$$\det A = (-1)^{\text{swaps}} \det U = (-1)^{\text{swaps}}\, u_{11} u_{22} \cdots u_{nn}.$$

```python
result = pan.determinant_by_reduction(pan.Matrix([[0, 2], [3, 4]]))
print(result)
```

```
Determinant of 2×2 matrix by row reduction — 1 step, 1 row swap
U =
[[3, 4],
 [0, 2]]
det(A) = (-1)^1 · 3 · 2 = -6
```

This is O(n³) instead of the O(n!) of cofactor expansion. `result.reduction` is the full
`Reduction`, so the individual row operations are there too.

## Adjugate

The adjugate is the transpose of the matrix of cofactors. It satisfies
$A\,\operatorname{adj}(A) = \det(A)\, I$, which gives the inverse formula
$A^{-1} = \operatorname{adj}(A) / \det(A)$:

```python
B = pan.Matrix([[1, 2], [3, 4]])
print(pan.adjugate(B))
# [[4, -2],
#  [-3, 1]]
print(B @ pan.adjugate(B) == pan.determinant(B) * pan.identity(2))  # True
```

## Cramer's rule

For a square system with $\det A \neq 0$, each unknown is $x_i = \det(A_i)/\det(A)$, where $A_i$ is
$A$ with column $i$ replaced by $b$:

```python
result = pan.cramer(pan.Matrix([[2, 1], [1, 3]]), pan.Vector([3, 5]))
print(result)
```

```
Cramer's rule for 2×2 system, det(A) = 5
  x0 = det(A0) / det(A) = 4 / 5 = 0.8
  x1 = det(A1) / det(A) = 7 / 5 = 1.4
x = [0.8, 1.4]
```

Use `exact_matrix`/`exact_vector` to get `4/5` and `7/5` instead of decimals. When $\det A = 0$ the
system has no unique solution, so `cramer` raises a `ValueError` and points you to `solve()`.

All four results render as LaTeX in Jupyter, with each determinant drawn between bars
(see [Rich Display](rich-display.md)).
