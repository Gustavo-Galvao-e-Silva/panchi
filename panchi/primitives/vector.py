from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from math import isqrt

from panchi._latex import vector_to_latex
from panchi.types import SCALAR_TYPES, Scalar, parse_scalar


def _exact_or_float_sqrt(value: Scalar) -> Scalar:
    """Return the square root as an exact int/Fraction when rational, else float."""
    if isinstance(value, float):
        return value**0.5
    if isinstance(value, Fraction):
        num_root = isqrt(value.numerator)
        den_root = isqrt(value.denominator)
        if num_root**2 == value.numerator and den_root**2 == value.denominator:
            return Fraction(num_root, den_root)
        return float(value) ** 0.5
    root = isqrt(value)
    if root**2 == value:
        return root
    return value**0.5


class Vector:
    """
    A mathematical vector for linear algebra operations.

    A vector is an ordered collection of numbers representing a point or
    direction in n-dimensional space. Vectors are fundamental objects in
    linear algebra used to represent quantities with magnitude and direction.

    Parameters
    ----------
    data : list[int | float | Fraction | str]
        A list of numbers representing the vector components. Strings
        like '1/3' are automatically parsed as Fraction values.

    Attributes
    ----------
    data : list[int | float | Fraction]
        The underlying list of vector components.
    shape : tuple[int, int]
        The dimensions of the vector as (dims, 1).

    Raises
    ------
    TypeError
        If data is not a list or contains non-numeric elements.

    Examples
    --------
    >>> v = Vector([1, 2, 3])
    >>> print(v.dims)
    3
    >>> print(v)
    [1, 2, 3]
    """

    def __init__(self, data: list[int | float | Fraction | str]) -> None:
        if not isinstance(data, list):
            raise TypeError("Vectors can only be created with lists of numbers")

        parsed = []
        for i, x in enumerate(data):
            try:
                parsed.append(parse_scalar(x))
            except TypeError:
                raise TypeError(
                    f"All vector elements must be numbers (int, float, Fraction, "
                    f"or string like '1/3'). Found {type(x).__name__} at index {i}."
                ) from None

        self.data = parsed
        self.shape = (len(parsed), 1)

    def __getitem__(self, key: int) -> Scalar:
        """
        Access a vector component by index.

        Parameters
        ----------
        key : int
            The index (0-based) of the component to access.

        Returns
        -------
        int | float | Fraction
            The value at the specified index.

        Raises
        ------
        TypeError
            If key is not an integer.

        Examples
        --------
        >>> v = Vector([10, 20, 30])
        >>> v[0]
        10
        >>> v[2]
        30
        """
        if not isinstance(key, int):
            raise TypeError("Indexes can only be integer values")

        return self.data[key]

    def __setitem__(self, key: int, new_value: Scalar) -> None:
        """
        Set a vector component at a specific index.

        Parameters
        ----------
        key : int
            The index (0-based) of the component to modify.
        new_value : int | float | Fraction
            The new value to assign.

        Raises
        ------
        TypeError
            If key is not an integer or new_value is not a number.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> v[1] = 5
        >>> print(v)
        [1, 5, 3]
        """
        if not isinstance(key, int):
            raise TypeError("Indexes can only be integer values")

        try:
            new_value = parse_scalar(new_value)
        except TypeError:
            raise TypeError("Vectors can only hold numbers") from None

        self.data[key] = new_value

    def __len__(self) -> int:
        """
        Get the number of components in the vector.

        Returns
        -------
        int
            The dimension of the vector.

        Examples
        --------
        >>> v = Vector([1, 2, 3, 4])
        >>> len(v)
        4
        """
        return self.dims

    def __iter__(self) -> Iterator:
        """
        Iterate over the vector components.

        Returns
        -------
        Iterator
            Iterator over vector components.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> for component in v:
        ...     print(component)
        1
        2
        3
        """
        return iter(self.data)

    def __add__(self, other: Vector) -> Vector:
        """
        Add two vectors component-wise.

        Vector addition requires both vectors to have the same dimension.

        Parameters
        ----------
        other : Vector
            The vector to add.

        Returns
        -------
        Vector
            The sum of the two vectors.

        Raises
        ------
        TypeError
            If other is not a Vector.
        ValueError
            If vectors have different dimensions.

        Examples
        --------
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = Vector([4, 5, 6])
        >>> v3 = v1 + v2
        >>> print(v3)
        [5, 7, 9]
        """
        if not isinstance(other, Vector):
            raise TypeError(
                f"Cannot add Vector and {type(other).__name__}. "
                f"Both operands must be vectors."
            )

        if self.dims != other.dims:
            raise ValueError(
                f"Cannot add vectors with different dimensions. "
                f"First vector is {self.dims}D, second vector is {other.dims}D."
            )

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] + other[i])

        return Vector(result)

    def __sub__(self, other: Vector) -> Vector:
        """
        Subtract one vector from another component-wise.

        Vector subtraction requires both vectors to have the same dimension.

        Parameters
        ----------
        other : Vector
            The vector to subtract.

        Returns
        -------
        Vector
            The difference of the two vectors.

        Raises
        ------
        TypeError
            If other is not a Vector.
        ValueError
            If vectors have different dimensions.

        Examples
        --------
        >>> v1 = Vector([5, 7, 9])
        >>> v2 = Vector([1, 2, 3])
        >>> v3 = v1 - v2
        >>> print(v3)
        [4, 5, 6]
        """
        if not isinstance(other, Vector):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from Vector. "
                f"Both operands must be vectors."
            )

        if self.dims != other.dims:
            raise ValueError(
                f"Cannot subtract vectors with different dimensions. "
                f"First vector is {self.dims}D, second vector is {other.dims}D."
            )

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] - other[i])

        return Vector(result)

    def __rmul__(self, other: Scalar) -> Vector:
        """
        Multiply vector by a scalar (from the left).

        Allows scalar multiplication in the form: scalar * vector

        Parameters
        ----------
        other : int | float | Fraction
            The scalar to multiply by.

        Returns
        -------
        Vector
            The result of scalar multiplication.

        Raises
        ------
        TypeError
            If other is not a number.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> result = 3 * v
        >>> print(result)
        [3, 6, 9]
        """
        if not isinstance(other, SCALAR_TYPES):
            return NotImplemented

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] * other)

        return Vector(result)

    def __mul__(self, other: Scalar) -> Vector:
        """
        Multiply vector by a scalar (from the right).

        Allows scalar multiplication in the form: vector * scalar

        Parameters
        ----------
        other : int | float | Fraction
            The scalar to multiply by.

        Returns
        -------
        Vector
            The result of scalar multiplication.

        Raises
        ------
        TypeError
            If other is not a number.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> result = v * 3
        >>> print(result)
        [3, 6, 9]
        """
        return self.__rmul__(other)

    def __truediv__(self, other: Scalar) -> Vector:
        """
        Divide vector by a scalar.

        Divides each component of the vector by the scalar value.

        Parameters
        ----------
        other : int | float | Fraction
            The scalar to divide by.

        Returns
        -------
        Vector
            The result of scalar division.

        Raises
        ------
        TypeError
            If other is not a number.

        Examples
        --------
        >>> v = Vector([6, 9, 12])
        >>> result = v / 3
        >>> print(result)
        [2.0, 3.0, 4.0]
        """
        if not isinstance(other, SCALAR_TYPES):
            return NotImplemented

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] / other)

        return Vector(result)

    def __neg__(self) -> Vector:
        """
        Negate the vector (multiply by -1).

        Returns
        -------
        Vector
            The negated vector.

        Examples
        --------
        >>> v = Vector([1, -2, 3])
        >>> neg_v = -v
        >>> print(neg_v)
        [-1, 2, -3]
        """
        return -1 * self

    def __eq__(self, other: object) -> bool:
        """
        Check if two vectors are equal.

        Vectors are equal if they have the same dimension and all
        corresponding components are equal.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if vectors are equal, False otherwise.

        Examples
        --------
        >>> Vector([1, 2, 3]) == Vector([1, 2, 3])
        True
        >>> Vector([1, 2, 3]) == Vector([1, 2, 4])
        False
        """
        if not isinstance(other, Vector):
            return NotImplemented

        if self.dims != other.dims:
            return False

        for i in range(self.dims):
            if self.data[i] != other.data[i]:
                return False

        return True

    __hash__ = None

    def __str__(self) -> str:
        """
        Return a string representation of the vector.

        Returns
        -------
        str
            String representation showing the vector components.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> print(v)
        [1, 2, 3]
        """
        return "[" + ", ".join(str(x) for x in self.data) + "]"

    def __repr__(self) -> str:
        """
        Return a constructor-style string for data inspection.

        Returns
        -------
        str
            A string showing the class name and data needed to recreate
            this vector, such as 'Vector([1, 2, 3])'.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> repr(v)
        'Vector([1, 2, 3])'
        """
        return f"Vector({self.data})"

    def _repr_latex_(self) -> str:
        """Render as a LaTeX column vector for Jupyter/Colab display."""
        return f"${vector_to_latex(self)}$"

    @property
    def dims(self) -> int:
        """
        Get the dimension of the vector.

        Returns
        -------
        int
            The number of components in the vector.

        Examples
        --------
        >>> v = Vector([1, 2, 3, 4, 5])
        >>> v.dims
        5
        """
        return self.shape[0]

    @property
    def magnitude_squared(self) -> Scalar:
        """
        Calculate the squared magnitude (squared Euclidean norm).

        The sum of the squared components. Unlike ``magnitude`` this never
        takes a square root, so it stays exact for integer and ``Fraction``
        vectors — useful for length comparisons and orthogonality checks
        that do not need the actual length.

        Returns
        -------
        int | float | Fraction
            The squared magnitude of the vector.

        Examples
        --------
        >>> Vector([3, 4]).magnitude_squared
        25
        """
        return sum(val**2 for val in self.data)

    @property
    def magnitude(self) -> Scalar:
        """
        Calculate the magnitude (length) of the vector.

        The Euclidean norm — the square root of ``magnitude_squared``. The
        result stays exact (an ``int`` or ``Fraction``) when that square root
        is rational, and is a ``float`` only when the length is irrational.

        Returns
        -------
        int | float | Fraction
            The magnitude of the vector.

        Examples
        --------
        >>> Vector([3, 4]).magnitude
        5
        >>> Vector([1, 1]).magnitude
        1.4142135623730951
        """
        return _exact_or_float_sqrt(self.magnitude_squared)

    def normalize(self) -> Vector:
        """
        Compute the unit vector in the same direction.

        A unit vector has magnitude 1 and points in the same direction
        as the original vector. Exactness is preserved when the length is
        rational: an exact vector whose magnitude is an integer or
        ``Fraction`` normalizes to exact components.

        Returns
        -------
        Vector
            The normalized vector (magnitude = 1).

        Raises
        ------
        ZeroDivisionError
            If the vector has zero magnitude.

        Examples
        --------
        >>> v = Vector([3, 4])
        >>> normalized = v.normalize()
        >>> print(normalized)
        [0.6, 0.8]
        >>> exact_vector([3, 4]).normalize()
        Vector([Fraction(3, 5), Fraction(4, 5)])
        """
        return self / self.magnitude

    def copy(self) -> Vector:
        """
        Create a deep copy of the vector.

        Returns
        -------
        Vector
            A new Vector object with the same components.

        Examples
        --------
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = v1.copy()
        >>> v2[0] = 99
        >>> print(v1[0])
        1
        """
        return Vector(self.data.copy())

    def to_list(self) -> list[Scalar]:
        """
        Convert the vector to a list.

        Returns
        -------
        list[int | float | Fraction]
            A copy of the vector components as a list.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> v.to_list()
        [1, 2, 3]
        """
        return self.data.copy()

    def to_tuple(self) -> tuple[Scalar, ...]:
        """
        Convert the vector to a tuple.

        Returns
        -------
        tuple[int | float | Fraction, ...]
            The vector components as a tuple.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> v.to_tuple()
        (1, 2, 3)
        """
        return tuple(self)
