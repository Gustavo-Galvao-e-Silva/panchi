# Vectors

A vector is an ordered list of numbers. In panchi, `Vector` is the fundamental object representing a point or direction in n-dimensional space.

## Construction

```python
import panchi as pan

v = pan.Vector([1, 2, 3])
print(v)       # [1, 2, 3]
print(v.dims)  # 3
```

Vectors hold numbers (int, float, or Fraction). You can write fractions as strings:

```python
v = pan.Vector(["1/3", "2/3", "1/2"])
print(v)  # [1/3, 2/3, 1/2]
```

For full details on exact arithmetic, see [Exact Arithmetic](exact-arithmetic.md).

In two dimensions, a `Vector` is an arrow from the origin. panchi can draw them for you:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/vectors.png" alt="Three 2D vectors drawn as arrows from the origin on a coordinate plane">
  <figcaption>Three vectors plotted with <code>Animator2D.plot_vectors</code>. See <a href="../visualizations/">Visualizations</a>.</figcaption>
</figure>

## Arithmetic

Vectors support standard arithmetic with natural syntax:

```python
a = pan.Vector([1, 2, 3])
b = pan.Vector([4, 5, 6])

print(a + b)   # [5, 7, 9]
print(a - b)   # [-3, -3, -3]
print(2 * a)   # [2, 4, 6]
print(a * 2)   # [2, 4, 6]  — same as above
print(a / 2)   # [0.5, 1.0, 1.5]
print(-a)      # [-1, -2, -3]
```

All operations return a new `Vector` and leave the original unchanged.

Addition places the two arrows tip-to-tail; scalar multiplication stretches an arrow along its own direction:

<figure class="viz" markdown="span">
  <img src="../../assets/viz/addition.gif" alt="Animation of adding two vectors tip-to-tail to produce their sum">
  <figcaption><code>animate_addition(Vector([3, 1]), Vector([1, 3]))</code></figcaption>
</figure>

<figure class="viz" markdown="span">
  <img src="../../assets/viz/scaling.gif" alt="Animation of a vector stretching to twice its length under scalar multiplication">
  <figcaption><code>animate_scaling(Vector([2, 1]), scale_factor=2.0)</code></figcaption>
</figure>

## Magnitude and normalization

The magnitude of a vector is its Euclidean length — the square root of the sum of squared components.

```python
v = pan.Vector([3, 4])
print(v.magnitude)   # 5
print(v.normalize()) # [0.6, 0.8]
```

`normalize()` returns the unit vector pointing in the same direction. It raises `ZeroDivisionError` if the vector has zero magnitude.

`magnitude_squared` skips the square root entirely, so it stays exact for integer and `Fraction` vectors — use it for length comparisons or orthogonality checks. `magnitude` and `normalize()` are exact too when the length is rational (a `[3, 4]` vector has length `5`), falling back to `float` only for irrational lengths like `[1, 1]`. See [Exact arithmetic](exact-arithmetic.md#length-and-normalization) for details.

## Dot product

The dot product of two vectors measures how much they point in the same direction. It is defined as the sum of the products of corresponding components.

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i} u_i v_i$$

```python
u = pan.Vector([1, 2, 3])
v = pan.Vector([4, 5, 6])
print(pan.dot(u, v))  # 32
```

If the dot product is zero, the vectors are orthogonal.

## Cross product

The cross product is defined only for 3D vectors. It produces a new vector perpendicular to both inputs, with magnitude equal to the area of the parallelogram they span.

```python
u = pan.Vector([1, 0, 0])
v = pan.Vector([0, 1, 0])
print(pan.cross(u, v))  # [0, 0, 1]
```

## Factory functions

panchi provides several convenience functions for constructing common vectors:

```python
pan.zero_vector(3)      # [0, 0, 0]
pan.one_vector(3)       # [1, 1, 1]
pan.unit_vector(3, 1)   # [0, 1, 0]  — standard basis vector e₁
pan.random_vector(3)    # random entries
```

## Indexing and iteration

Vectors are indexable and iterable:

```python
v = pan.Vector([10, 20, 30])
print(v[0])            # 10
v[1] = 99              # mutation is supported
print(list(v))         # [10, 99, 30]
```

## Conversion

```python
v = pan.Vector([1, 2, 3])
v.to_list()    # [1, 2, 3]  — returns an independent copy
v.to_tuple()   # (1, 2, 3)
```
