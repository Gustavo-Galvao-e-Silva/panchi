# panchi

The top-level `panchi` module exports all primitives and factory functions directly. You rarely need to import from submodules.

```python
import panchi as pan
```

## Primitives

::: panchi.Vector

::: panchi.Matrix

## Factory functions

::: panchi.identity

::: panchi.zero_matrix

!!! warning "Deprecated in 3.1.0, removed in 4.0"
    Use `Matrix([[1] * cols for _ in range(rows)])`.

::: panchi.one_matrix

::: panchi.zero_vector

!!! warning "Deprecated in 3.1.0, removed in 4.0"
    Use `Vector([1] * dims)`.

::: panchi.one_vector

::: panchi.unit_vector

::: panchi.diagonal

::: panchi.random_vector

::: panchi.random_matrix

::: panchi.rotation_matrix_2d

::: panchi.rotation_matrix_3d

## Exact arithmetic factories

::: panchi.exact_vector

::: panchi.exact_matrix

## Vector operations

::: panchi.dot

::: panchi.cross
