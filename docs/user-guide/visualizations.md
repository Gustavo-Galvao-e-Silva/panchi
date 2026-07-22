# Visualizations

panchi includes a built-in visualization system for 2D vectors, linear transformations, and vector spaces. All visualizations are accessed through a single class: `Animator2D`.

## Setup

panchi ships with a **matplotlib** backend by default. For higher-quality, video-based output, install the optional **manim** backend:

```bash
pip install panchi[manim]
```

## Creating an animator

```python
from panchi import Vector, Matrix, VectorSpace
from panchi.visualizations import Animator2D

# Interactive display (matplotlib pops up a window)
animator = Animator2D()

# Save to disk instead
animator = Animator2D(save_path="./my_plots")

# Use manim for video output
animator = Animator2D(backend="manim", save_path="./videos")
```

The `backend` parameter selects which rendering engine to use. Manim is only imported when you request it, so matplotlib users never need it installed.

### Constructor options

| Parameter   | Default         | Description |
|-------------|-----------------|-------------|
| `backend`   | `"matplotlib"`  | `"matplotlib"` or `"manim"` |
| `save_path` | `None`          | Directory for output files. If `None`, matplotlib shows interactively. |
| `quality`   | `"medium"`      | Render quality: `"low"`, `"medium"`, `"high"`, `"production"` (manim only) |
| `figsize`   | `(8, 8)`        | Figure size in inches (matplotlib only) |

When `save_path` is set, static plots are saved as `.png` and animations as `.gif` (matplotlib) or `.mp4` (manim).

---

## Plotting vectors

```python
v1 = Vector([3, 2])
v2 = Vector([-1, 3])
v3 = Vector([2, -1])

animator.plot_vectors(v1, v2, v3, labels=["v1", "v2", "v3"])
```

Pass any number of 2D vectors as positional arguments. Optional parameters:

- `colors` — list of hex color strings (e.g. `["#FF0000", "#0000FF"]`)
- `labels` — list of label strings
- `grid` — show coordinate grid (default `True`)
- `name` — output filename when saving (default `"plot_vectors"`)

<figure class="viz" markdown="span">
  <img src="../../assets/viz/vectors.png" alt="Example output: three vectors as arrows from the origin">
  <figcaption>Example output.</figcaption>
</figure>

---

## Animating vector addition

```python
v1 = Vector([3, 1])
v2 = Vector([1, 3])

animator.animate_addition(v1, v2)
```

The animation shows v1 drawn first, then v2 growing from the origin and from v1's tip simultaneously, and finally the result vector appearing. The manim backend also draws the parallelogram with dashed lines.

Optional parameters: `frames`, `interval` (milliseconds between frames), `name`, and `colors` (up to three, for `[v1, v2, v1 + v2]`).

<figure class="viz" markdown="span">
  <img src="../../assets/viz/addition.gif" alt="Example output: two vectors added tip-to-tail">
  <figcaption>Example output.</figcaption>
</figure>

---

## Animating scalar multiplication

```python
v = Vector([2, 1])

animator.animate_scaling(v, scale_factor=2.5)
```

The vector smoothly stretches (or shrinks, or flips for negative factors) from its original length to the scaled length.

Optional parameters: `frames`, `interval`, `name`, and `colors` (up to two, for `[original, scaled]`).

<figure class="viz" markdown="span">
  <img src="../../assets/viz/scaling.gif" alt="Example output: a vector stretching to twice its length">
  <figcaption>Example output.</figcaption>
</figure>

---

## Animating linear transformations

This is the signature visualization — a full grid deformation showing how a 2x2 matrix transforms the plane, in the style of 3Blue1Brown's *Essence of Linear Algebra*.

```python
# 90-degree rotation
animator.animate_transform(Matrix([[0, -1], [1, 0]]))

# Horizontal shear
animator.animate_transform(Matrix([[1, 1], [0, 1]]))

# Projection onto the x-axis
animator.animate_transform(Matrix([[1, 0], [0, 0]]))
```

The animation shows:

1. The standard basis vectors e1 (red) and e2 (blue) on a coordinate grid
2. The entire grid smoothly morphing from the identity to the target transformation
3. Labels updating to show the final column vectors of the matrix

Only 2x2 matrices are supported — a `ValueError` is raised for other shapes.

Optional parameters: `frames`, `interval`, `name`, and `colors` (up to two, for the basis arrows `[e1, e2]`).

<figure class="viz" markdown="span">
  <img src="../../assets/viz/transform_shear.gif" alt="Example output: a coordinate grid shearing">
  <figcaption>Example output — a horizontal shear.</figcaption>
</figure>

---

## Visualizing spans

`plot_span` draws the subspace spanned by a set of vectors, with a shaded region and basis vector arrows.

```python
# From individual vectors
animator.plot_span(Vector([1, 2]))                        # 1D span (line)
animator.plot_span(Vector([1, 0]), Vector([0, 1]))        # 2D span (full plane)

# From a VectorSpace object
space = VectorSpace([Vector([1, 1]), Vector([1, -1])])
animator.plot_span(space, labels=["v1", "v2"])
```

The visualization adapts to the dimension of the subspace:

- **dim 1** — a line through the origin in the basis direction, highlighted with a colored band
- **dim 2** — the entire visible plane shaded to indicate it spans all of R²

If you pass linearly dependent vectors, panchi computes the actual basis automatically:

```python
# These two vectors are parallel — the span is still 1D
animator.plot_span(Vector([1, 2]), Vector([2, 4]))
```

<figure class="viz" markdown="span">
  <img src="../../assets/viz/span_plane.png" alt="Example output: two vectors shading the plane they span">
  <figcaption>Example output — two independent vectors spanning R².</figcaption>
</figure>

Optional parameters: `colors`, `labels`, `grid`, `name`, and `span_color` (the shade of the span region).

---

## Saving output

When `save_path` is set, every method saves its output to that directory:

```python
animator = Animator2D(save_path="./output")

animator.plot_vectors(Vector([1, 2]), name="my_vectors")
# → ./output/my_vectors.png

animator.animate_transform(Matrix([[0, -1], [1, 0]]), name="rotation")
# → ./output/rotation.gif  (matplotlib)
# → ./output/rotation.mp4  (manim)
```

The `name` parameter controls the filename (without extension). If omitted, it defaults to the method name (e.g. `"plot_vectors"`, `"animate_transform"`).

---

## Backend comparison

| Feature | matplotlib | manim |
|---------|-----------|-------|
| Install | Included with panchi | `pip install panchi[manim]` + system deps |
| Output format | `.png` / `.gif` | `.mp4` video |
| Interactive display | Yes (`plt.show()`) | No (always renders to file) |
| Grid morph quality | Good (LineCollection interpolation) | Excellent (native NumberPlane) |
| LaTeX labels | No | Yes |
| Parallelogram in addition | No | Yes |
| Render speed | Fast | Slower (video encoding) |

Both backends support all five visualization methods with the same API.
