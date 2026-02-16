"""
Visualization module for Mathrix.

Provides tools for creating beautiful 2D and 3D visualizations of vectors,
matrices, and linear transformations.

Examples
--------
Basic usage with matplotlib (default):

>>> from mathrix.visualizations import Animator2D
>>> from mathrix.primitives.vector import Vector
>>>
>>> animator = Animator2D()
>>> v1 = Vector([3, 2])
>>> v2 = Vector([1, 3])
>>>
>>> animator.plot_vectors(v1, v2, labels=['v₁', 'v₂'])
>>> animator.animate_addition(v1, v2)

With Manim backend (requires Manim installation):

>>> animator = Animator2D(backend='manim', save_animations=True)
>>> animator.plot_vectors(v1, v2)

Saving animations:

>>> animator = Animator2D(
...     save_animations=True,
...     output_dir='./my_animations',
...     resolution=(1920, 1080)
... )
>>> animator.animate_addition(v1, v2, name='vector_sum')
"""

from mathrix.visualizations.animator_2d import Animator2D

__all__ = ["Animator2D"]
