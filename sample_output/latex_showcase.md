# panchi 2.0 -- rich display (`_repr_latex_`) showcase

Raw LaTeX each object emits for Jupyter/MathJax. Open `latex_showcase.html` to see it rendered.

## Vector
_A vector with an exact fraction component_

**`_repr_latex_()`**
```latex
$\begin{bmatrix} 1 \\ \frac{3}{2} \\ -2 \end{bmatrix}$
```
**plain `str()`**
```
[1, 3/2, -2]
```

## Matrix
_A 3x3 matrix (note the exact 1/2 entry)_

**`_repr_latex_()`**
```latex
$\begin{bmatrix} 2 & \frac{1}{2} & 0 \\ 1 & 3 & -1 \\ 0 & 4 & 1 \end{bmatrix}$
```
**plain `str()`**
```
[[2, 1/2, 0],
 [1, 3, -1],
 [0, 4, 1]]
```

## VectorSpace
_A subspace of R^3 from three generators_

**`_repr_latex_()`**
```latex
$\operatorname{span}\left\{ \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix},\; \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix},\; \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \right\}$
```
**plain `str()`**
```
VectorSpace in R^3, spanned by 3 vectors:
  [1, 0, -1]
  [0, 1, 2]
  [1, 1, 1]
```

## Reduction (rref)
_Row-reduction as an arrow chain -- the centerpiece_

**`_repr_latex_()`**
```latex
$$\begin{aligned}
& \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 0 \\ 1 & 1 & 1 \end{bmatrix} \\
&\xrightarrow{R_{1} \to R_{1} + (-2.0)\,R_{0}} \begin{bmatrix} 1 & 2 & 1 \\ 0.0 & 0.0 & -2.0 \\ 1 & 1 & 1 \end{bmatrix} \\
&\xrightarrow{R_{2} \to R_{2} + (-1.0)\,R_{0}} \begin{bmatrix} 1.0 & 2.0 & 1.0 \\ 0.0 & 0.0 & -2.0 \\ 0.0 & -1.0 & 0.0 \end{bmatrix} \\
&\xrightarrow{R_{1} \leftrightarrow R_{2}} \begin{bmatrix} 1.0 & 2.0 & 1.0 \\ 0.0 & -1.0 & 0.0 \\ 0.0 & 0.0 & -2.0 \end{bmatrix} \\
&\xrightarrow{R_{1} \to -1.0\,R_{1}} \begin{bmatrix} 1.0 & 2.0 & 1.0 \\ 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & -2.0 \end{bmatrix} \\
&\xrightarrow{R_{0} \to R_{0} + (-2.0)\,R_{1}} \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & -2.0 \end{bmatrix} \\
&\xrightarrow{R_{2} \to -0.5\,R_{2}} \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix} \\
&\xrightarrow{R_{0} \to R_{0} + (-1.0)\,R_{2}} \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}
\end{aligned}$$
```
**plain `str()`**
```
RREF of 3×3 matrix — 7 steps, rank 3

Step 1: R1 -> R1 + (-2.0) * R0
[[1, 2, 1],
 [0.0, 0.0, -2.0],
 [1, 1, 1]]

Step 2: R2 -> R2 + (-1.0) * R0
[[1.0, 2.0, 1.0],
 [0.0, 0.0, -2.0],
 [0.0, -1.0, 0.0]]

Step 3: R1 <-> R2
[[1.0, 2.0, 1.0],
 [0.0, -1.0, 0.0],
 [0.0, 0.0, -2.0]]

Step 4: R1 -> -1.0 * R1
[[1.0, 2.0, 1.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, -2.0]]

Step 5: R0 -> R0 + (-2.0) * R1
[[1.0, 0.0, 1.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, -2.0]]

Step 6: R2 -> -0.5 * R2
[[1.0, 0.0, 1.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]

Step 7: R0 -> R0 + (-1.0) * R2
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]

Result:
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]

Pivots: [(0, 0), (1, 1), (2, 2)]
Rank: 3  |  Nullity: 0
```

## Solution (solve)
_Solving A x = b_

**`_repr_latex_()`**
```latex
$x = \begin{bmatrix} 1.0 \\ 3.0 \end{bmatrix}$
```
**plain `str()`**
```
Solution to 2×2 system — unique

x = [1.0, 3.0]
```

## LUDecomposition
_A = LU (PA = LU when pivoted)_

**`_repr_latex_()`**
```latex
$$\begin{bmatrix} 4 & 3 \\ 6 & 3 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1.5 & 1 \end{bmatrix}\,\begin{bmatrix} 4 & 3 \\ 0.0 & -1.5 \end{bmatrix}$$
```
**plain `str()`**
```
LU decomposition of 2×2 matrix — 1 steps

P @ A = L @ U

P:
[[1, 0],
 [0, 1]]

A:
[[4, 3],
 [6, 3]]

L:
[[1, 0],
 [1.5, 1]]

U:
[[4, 3],
 [0.0, -1.5]]
```

## QRDecomposition
_A = QR_

**`_repr_latex_()`**
```latex
$$\begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.7071067811865475 & 0.4082482904638631 \\ 0.0 & 0.8164965809277261 \\ 0.7071067811865475 & -0.4082482904638631 \end{bmatrix}\,\begin{bmatrix} 1.414213562373095 & 0.7071067811865475 \\ 0.0 & 1.2247448713915892 \end{bmatrix}$$
```
**plain `str()`**
```
QR decomposition of 3×2 matrix — 2 steps

Step 1: v0 = [1, 0, 1]
  orthogonal: [1, 0, 1]
  normalize -> q0: [0.7071067811865475, 0.0, 0.7071067811865475]

Step 2: v1 = [1, 1, 0]
  remove projection onto q0: [0.5, 0.0, 0.5]
  orthogonal: [0.5, 1.0, -0.5]
  normalize -> q1: [0.4082482904638631, 0.8164965809277261, -0.4082482904638631]

A = Q @ R

A:
[[1, 1],
 [0, 1],
 [1, 0]]

Q:
[[0.7071067811865475, 0.4082482904638631],
 [0.0, 0.8164965809277261],
 [0.7071067811865475, -0.4082482904638631]]

R:
[[1.414213562373095, 0.7071067811865475],
 [0.0, 1.2247448713915892]]
```

## InverseResult
_A^{-1} via Gauss-Jordan_

**`_repr_latex_()`**
```latex
$$\begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}^{-1} = \begin{bmatrix} 0.6000000000000001 & -0.7000000000000001 \\ -0.2 & 0.4 \end{bmatrix}$$
```
**plain `str()`**
```
Inverse of 2×2 matrix — 4 steps

Inverse:
[[0.6000000000000001, -0.7000000000000001],
 [-0.2, 0.4]]
```

## EigenResult
_Eigenvalues and eigenvectors_

**`_repr_latex_()`**
```latex
$$\begin{aligned}
\lambda_{1} = 2, \quad v_{1} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} \\
\lambda_{2} = 3, \quad v_{2} = \begin{bmatrix} -0.0 \\ 1.0 \end{bmatrix}
\end{aligned}$$
```
**plain `str()`**
```
Eigendecomposition of 2×2 matrix — 0 iterations (converged: True)

λ = 2
  v = [1.0, 0.0]
λ = 3
  v = [-0.0, 1.0]
```

## gram_schmidt
_Orthonormal basis from Gram-Schmidt (list of Vectors)_

**`_repr_latex_()`**
```latex
$\begin{bmatrix} 0.7071067811865475 \\ 0.7071067811865475 \\ 0.0 \end{bmatrix}  ,\quad  \begin{bmatrix} 0.4082482904638631 \\ -0.4082482904638631 \\ 0.8164965809277261 \end{bmatrix}$
```
**plain `str()`**
```
[0.7071067811865475, 0.7071067811865475, 0.0]
[0.4082482904638631, -0.4082482904638631, 0.8164965809277261]
```
