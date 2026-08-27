"""
Visualization module for panchi.

Provides 2D visualizations of vectors, matrices, and linear transformations
via matplotlib (default) and manim backends.

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
>>> animator.plot_vectors(v1, v2, labels=["v1", "v2"])
>>> animator.animate_addition(v1, v2)

With Manim backend (requires ``pip install panchi[manim]``):

>>> animator = Animator2D(backend="manim", save_path="./videos")
>>> animator.plot_vectors(v1, v2)

Animate a linear transformation:

>>> animator = Animator2D(save_path="./output")
>>> animator.animate_transform(Matrix([[0, -1], [1, 0]]))
"""

from panchi.visualizations.animator import Animator2D, Animator3D

__all__ = ["Animator2D", "Animator3D"]
