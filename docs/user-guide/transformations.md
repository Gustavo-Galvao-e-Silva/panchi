# Linear Transformations

A matrix is not just a grid of numbers — it is a function. When you multiply a matrix by a vector, you are applying a linear transformation: stretching, rotating, reflecting, or projecting the vector into a new position.

panchi makes this concrete. Here is a 90° rotation applied to the whole plane — watch the basis vectors carry the grid with them:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/transform_rotation.gif" alt="A coordinate grid rotating 90 degrees, with the basis vectors following the rotation">
  <figcaption><code>animate_transform(Matrix([[0, -1], [1, 0]]))</code> — a quarter-turn. The columns of the matrix are exactly where the basis vectors land.</figcaption>
</figure>

## Applying a transformation

The `@` operator applies a matrix to a vector:

```python
import panchi as pan

A = pan.Matrix([[2, 0],
                [0, 3]])

v = pan.Vector([1, 1])
print(A @ v)  # [2, 3]
```

This scales the x-component by 2 and the y-component by 3. The matrix encodes the transformation; the vector is what gets transformed.

`transform(v)` is an explicit alias for the same operation, useful when you want to be clear about intent:

```python
print(A.transform(v))  # [2, 3]
```

## Rotation matrices

A rotation matrix rotates vectors without changing their magnitude. panchi provides factory functions for 2D and 3D rotations:

```python
from math import pi

R = pan.rotation_matrix_2d(pi / 2)  # 90° counterclockwise

point = pan.Vector([1, 0])
rotated = R @ point
print(rotated)  # [~0.0, 1.0]
```

The rotated vector has the same magnitude as the original:

```python
from math import isclose
assert isclose(rotated.magnitude, point.magnitude, abs_tol=1e-10)
```

For 3D rotations, provide an axis vector:

```python
axis = pan.Vector([0, 0, 1])  # rotate around the z-axis
R3 = pan.rotation_matrix_3d(pi / 2, axis)
```

The axis is normalized automatically.

## Composing transformations

Because matrix multiplication is associative, you can compose transformations by multiplying their matrices. The transformations are applied right to left:

```python
scale = pan.Matrix([[2, 0], [0, 2]])
rotate = pan.rotation_matrix_2d(pi / 2)

# First scale, then rotate
composed = rotate @ scale

v = pan.Vector([1, 0])
print(composed @ v)  # [0, 2]
```

## Verifying mathematical identities

panchi's transparent implementation makes it easy to verify textbook identities experimentally:

```python
A = pan.Matrix([[1, 2], [3, 4]])
B = pan.Matrix([[5, 6], [7, 8]])

# (AB)ᵀ = BᵀAᵀ
assert (A @ B).T == B.T @ A.T
```
