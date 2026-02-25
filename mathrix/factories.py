from __future__ import annotations
import random
from math import pi, cos, sin

from mathrix.primitives.vector import Vector
from mathrix.primitives.matrix import Matrix


def identity(n: int) -> Matrix:
    """
    Create an n×n identity matrix.

    The identity matrix is a square matrix with ones on the diagonal
    and zeros elsewhere. It acts as the multiplicative identity in
    matrix operations (I × A = A × I = A).

    Parameters
    ----------
    n : int
        Size of the identity matrix (number of rows and columns).

    Returns
    -------
    Matrix
        An n×n identity matrix.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is not a positive integer.

    Examples
    --------
    >>> I = identity(3)
    >>> print(I)
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    """
    if not isinstance(n, int):
        raise TypeError(f"Size must be an integer. Got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"Size must be positive. Got {n}.")

    return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


def zero_matrix(rows: int, cols: int) -> Matrix:
    """
    Create a matrix filled with zeros.

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    cols : int
        Number of columns in the matrix.

    Returns
    -------
    Matrix
        A rows×cols matrix filled with zeros.

    Raises
    ------
    TypeError
        If rows or cols is not an integer.
    ValueError
        If rows or cols is not positive.

    Examples
    --------
    >>> Z = zero_matrix(2, 3)
    >>> print(Z)
    [[0, 0, 0],
     [0, 0, 0]]
    """
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError(
            f"Rows and columns must be integers. Got rows: {type(rows).__name__}, cols: {type(cols).__name__}."
        )
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"Rows and columns must be positive. Got rows: {rows}, cols: {cols}."
        )

    return Matrix([[0 for _ in range(cols)] for _ in range(rows)])


def one_matrix(rows: int, cols: int) -> Matrix:
    """
    Create a matrix filled with ones.

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    cols : int
        Number of columns in the matrix.

    Returns
    -------
    Matrix
        A rows×cols matrix filled with ones.

    Raises
    ------
    TypeError
        If rows or cols is not an integer.
    ValueError
        If rows or cols is not positive.

    Examples
    --------
    >>> O = one_matrix(2, 3)
    >>> print(O)
    [[1, 1, 1],
     [1, 1, 1]]
    """
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError(
            f"Rows and columns must be integers. Got rows: {type(rows).__name__}, cols: {type(cols).__name__}."
        )
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"Rows and columns must be positive. Got rows: {rows}, cols: {cols}."
        )

    return Matrix([[1 for _ in range(cols)] for _ in range(rows)])


def zero_vector(dims: int) -> Vector:
    """
    Create a vector filled with zeros.

    The zero vector is the additive identity in vector spaces
    (v + 0 = v for any vector v).

    Parameters
    ----------
    dims: int
        Number of components in the vector.

    Returns
    -------
    Vector
        A vector with all components equal to zero.

    Raises
    ------
    TypeError
        If dims is not an integer.
    ValueError
        If dims is not positive.

    Examples
    --------
    >>> z = zero_vector(3)
    >>> print(z)
    [0, 0, 0]
    """
    if not isinstance(dims, int):
        raise TypeError(f"Dimension must be an integer. Got {type(dims).__name__}.")
    if dims <= 0:
        raise ValueError(f"Dimension must be positive. Got {dims}.")

    return Vector([0 for _ in range(dims)])


def one_vector(dims: int) -> Vector:
    """
    Create a vector filled with ones.

    Parameters
    ----------
    dims: int
        Number of components in the vector.

    Returns
    -------
    Vector
        A vector with all components equal to one.

    Raises
    ------
    TypeError
        If dims is not an integer.
    ValueError
        If dims is not positive.

    Examples
    --------
    >>> o = one_vector(3)
    >>> print(o)
    [1, 1, 1]
    """
    if not isinstance(dims, int):
        raise TypeError(f"Dimension must be an integer. Got {type(dims).__name__}.")
    if dims <= 0:
        raise ValueError(f"Dimension must be positive. Got {dims}.")

    return Vector([1 for _ in range(dims)])


def unit_vector(dims: int, index: int) -> Vector:
    """
    Create a unit vector (standard basis vector).

    A unit vector has a single component equal to 1 and all others equal to 0.
    These form the standard basis for vector spaces.

    Parameters
    ----------
    dims: int
        Number of components in the vector.
    index : int
        Position of the 1 component (0-indexed).

    Returns
    -------
    Vector
        A unit vector with 1 at the specified index.

    Raises
    ------
    TypeError
        If dims or index is not an integer.
    ValueError
        If dims is not positive or index is out of range.

    Examples
    --------
    >>> e2 = unit_vector(3, 1)
    >>> print(e2)
    [0, 1, 0]
    """
    if not isinstance(dims, int) or not isinstance(index, int):
        raise TypeError(
            f"Dimension and index must be integers. Got dimension: {type(dims).__name__}, index: {type(index).__name__}."
        )
    if dims <= 0:
        raise ValueError(f"Dimension must be positive. Got {dims}.")
    if index < 0 or index >= dims:
        raise ValueError(f"Index must be between 0 and {dims - 1}. Got {index}.")

    return Vector([1 if i == index else 0 for i in range(dims)])


def diagonal(values: list[int | float] | Vector) -> Matrix:
    """
    Create a diagonal matrix from a list of values.

    A diagonal matrix has the specified values on its main diagonal
    and zeros everywhere else.

    Parameters
    ----------
    values : list[int | float] | Vector
        The diagonal values.

    Returns
    -------
    Matrix
        A square matrix with values on the diagonal.

    Examples
    --------
    >>> D = diagonal([1, 2, 3])
    >>> print(D)
    [[1, 0, 0],
     [0, 2, 0],
     [0, 0, 3]]
    """
    n = len(values)
    return Matrix([[values[i] if i == j else 0 for j in range(n)] for i in range(n)])


def random_vector(dims: int, low: int | float = 0.0, high: int | float = 1.0) -> Vector:
    """
    Create a vector with random components.

    Components are uniformly distributed in the range [low, high).

    Parameters
    ----------
    dims : int
        Number of components in the vector.
    low : int | float, optional
        Lower bound of the random range (inclusive). Default is 0.0.
    high : int | float, optional
        Upper bound of the random range (exclusive). Default is 1.0.

    Returns
    -------
    Vector
        A vector with random components.

    Raises
    ------
    ValueError
        If low is not smaller than high.

    Examples
    --------
    >>> v = random_vector(3, low=-1.0, high=1.0)
    >>> v.dims
    3
    """
    if low >= high:
        raise ValueError(
            f"Low must be smaller than high. Got low: {low}, high: {high}."
        )

    return Vector([random.uniform(low, high) for _ in range(dims)])


def random_matrix(
    rows: int, cols: int, low: int | float = 0.0, high: int | float = 1.0
) -> Matrix:
    """
    Create a matrix with random elements.

    Elements are uniformly distributed in the range [low, high).

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    cols : int
        Number of columns in the matrix.
    low : int | float, optional
        Lower bound of the random range (inclusive). Default is 0.0.
    high : int | float, optional
        Upper bound of the random range (exclusive). Default is 1.0.

    Returns
    -------
    Matrix
        A matrix with random elements.

    Raises
    ------
    ValueError
        If low is not smaller than high.

    Examples
    --------
    >>> m = random_matrix(2, 3, low=0.0, high=10.0)
    >>> m.shape
    (2, 3)
    """
    if low >= high:
        raise ValueError(
            f"Low must be smaller than high. Got low: {low}, high: {high}."
        )

    return Matrix(
        [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]
    )


def rotation_matrix_2d(angle: int | float, radians: bool = True) -> Matrix:
    """
    Create a 2D rotation matrix.

    Produces a matrix that rotates vectors in 2D space by the specified angle
    counterclockwise around the origin.

    Parameters
    ----------
    angle : int | float
        The rotation angle.
    radians : bool, optional
        If True, angle is in radians. If False, angle is in degrees.
        Default is True.

    Returns
    -------
    Matrix
        A 2×2 rotation matrix.

    Examples
    --------
    >>> import math
    >>> R = rotation_matrix_2d(math.pi / 2)
    >>> v = Vector([1, 0])
    >>> rotated = R @ v
    >>> print(rotated)
    [0.0, 1.0]
    """
    angle_radians = angle if radians else (angle * pi / 180)
    cos_angle = cos(angle_radians)
    sin_angle = sin(angle_radians)

    return Matrix([[cos_angle, -sin_angle], [sin_angle, cos_angle]])


def rotation_matrix_3d(
    angle: int | float, axis: Vector, radians: bool = True
) -> Matrix:
    """
    Create a 3D rotation matrix using Rodrigues' rotation formula.

    Produces a matrix that rotates vectors in 3D space by the specified angle
    around the given axis vector.

    Parameters
    ----------
    angle : int | float
        The rotation angle.
    axis : Vector
        The axis of rotation (will be normalized automatically).
    radians : bool, optional
        If True, angle is in radians. If False, angle is in degrees.
        Default is True.

    Returns
    -------
    Matrix
        A 3×3 rotation matrix.

    Raises
    ------
    ValueError
        If the axis is the zero vector.

    Examples
    --------
    >>> import math
    >>> axis = Vector([0, 0, 1])
    >>> R = rotation_matrix_3d(math.pi / 2, axis)
    >>> v = Vector([1, 0, 0])
    >>> rotated = R @ v
    >>> print(rotated)
    [0.0, 1.0, 0.0]
    """
    if axis.magnitude == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")

    angle_radians = angle if radians else (angle * pi / 180)
    cos_angle = cos(angle_radians)
    sin_angle = sin(angle_radians)
    ver = 1 - cos_angle

    norm_vector = axis.normalize()
    unit_x, unit_y, unit_z = norm_vector.to_tuple()

    row_1 = [
        cos_angle + unit_x**2 * ver,
        unit_x * unit_y * ver - unit_z * sin_angle,
        unit_x * unit_z * ver + unit_y * sin_angle,
    ]

    row_2 = [
        unit_y * unit_x * ver + unit_z * sin_angle,
        cos_angle + unit_y**2 * ver,
        unit_y * unit_z * ver - unit_x * sin_angle,
    ]

    row_3 = [
        unit_z * unit_x * ver - unit_y * sin_angle,
        unit_z * unit_y * ver + unit_x * sin_angle,
        cos_angle + unit_z**2 * ver,
    ]

    return Matrix([row_1, row_2, row_3])
