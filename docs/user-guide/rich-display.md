# Rich display in Jupyter

In a Jupyter notebook or Colab, panchi's objects render as **typeset LaTeX** instead of plain text — and every result object shows its full step-by-step derivation the way you'd write it on paper.

This is pure progressive enhancement. In a terminal or a script, `print(...)` and `str(...)` are exactly as before; the LaTeX only appears when a notebook asks for it, and the core library takes on no extra dependency.

## Objects

Displaying a `Vector`, `Matrix`, or `VectorSpace` as the last line of a cell renders it as math:

```python
import panchi as pan

pan.Matrix([[1, 2], [3, 4]])
```

renders as

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

Exact arithmetic pays off here — fractions typeset as fractions, with no floating-point noise:

```python
pan.exact_matrix([["1/2", "1/3"], ["1/4", "1/5"]])
```

$$\begin{bmatrix} \frac{1}{2} & \frac{1}{3} \\ \frac{1}{4} & \frac{1}{5} \end{bmatrix}$$

A `VectorSpace` renders as the span of its generators.

## Operations show their work

Result objects render the *derivation*, not just the answer. A row reduction becomes the sequence of matrices it passes through, each arrow labelled with the row operation:

```python
from panchi.algorithms import rref

rref(pan.Matrix([[1, 2], [3, 4]]))
```

$$\begin{aligned}
& \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\
&\xrightarrow{R_{1} \to R_{1} + (-3.0)\,R_{0}} \begin{bmatrix} 1 & 2 \\ 0 & -2.0 \end{bmatrix} \\
&\xrightarrow{R_{1} \to -0.5\,R_{1}} \begin{bmatrix} 1 & 2 \\ 0 & 1.0 \end{bmatrix} \\
&\xrightarrow{R_{0} \to R_{0} + (-2.0)\,R_{1}} \begin{bmatrix} 1 & 0 \\ 0 & 1.0 \end{bmatrix}
\end{aligned}$$

The same holds across the library:

- `solve(A, b)` renders the solution vector, or the full general solution `x = x_p + t\,v` when there are infinitely many.
- `lu(A)` and `qr_decomposition(A)` render the factorization `A = LU` / `A = QR`.
- `inverse(A)` renders `A^{-1} = …`.
- `eigen(A)` renders each eigenvalue with its eigenvector.

## Visualizations

Plots and animations display **inline** in a notebook automatically — no `save_path` needed:

```python
from panchi.visualizations import Animator2D

Animator2D().plot_vectors([pan.Vector([2, 1]), pan.Vector([1, 3])])
```

Animations play as inline players (via matplotlib's own HTML animation), so you can scrub through a transformation without leaving the notebook. Set a `save_path` to write files to disk instead, exactly as in a script.
