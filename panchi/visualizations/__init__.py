"""
Visualization module for panchi.

Provides 2D and 3D visualizations of vectors, matrices, and linear
transformations via matplotlib (default) and manim backends. Use
``Animator2D`` for vectors in R² and ``Animator3D`` for vectors in R³;
both share the same five-method API.

Examples
--------
Basic usage with matplotlib (default):

>>> from panchi import Vector, Matrix
>>> from panchi.visualizations import Animator2D
>>>
>>> animator = Animator2D()
>>> v1 = Vector([3, 2])
>>> v2 = Vector([1, 3])
>>>
>>> animator.plot_vectors([v1, v2], labels=["v1", "v2"])
>>> animator.animate_addition(v1, v2)

With Manim backend (requires ``pip install panchi[manim]``):

>>> animator = Animator2D(backend="manim", save_path="./videos")
>>> animator.plot_vectors([v1, v2])

Animate a linear transformation:

>>> animator = Animator2D(save_path="./output")
>>> animator.animate_transform(Matrix([[0, -1], [1, 0]]))

The same methods work in three dimensions with ``Animator3D``:

>>> from panchi.visualizations import Animator3D
>>>
>>> animator = Animator3D()
>>> animator.plot_vectors([Vector([3, 2, 1]), Vector([1, 3, 2])])
>>> animator.animate_transform(Matrix([[1, -1, 0], [1, 1, 0], [0, 0, 1]]))

``plot_compression`` shows low-rank SVD approximations of a matrix image:

>>> from panchi import svd
>>> from panchi.visualizations import plot_compression
>>>
>>> image = Matrix([[1 if 2 <= c <= 5 else 0 for c in range(8)] for r in range(8)])
>>> plot_compression(svd(image), [1, 2])
"""

from panchi.visualizations.animator import Animator2D, Animator3D
from panchi.visualizations.compression import plot_compression

__all__ = ["Animator2D", "Animator3D", "plot_compression"]
