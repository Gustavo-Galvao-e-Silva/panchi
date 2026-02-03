from typing import Iterator


class Vector:
    def __init__(self, data: list[int | float]) -> None:
        if not (
            isinstance(data, list) and
            all(isinstance(x, (int, float)) for x in data)
        ):
            raise TypeError("Vectors can only be created with lists of numbers")

        self.data = data
        self.shape = (len(data), 1)

    def __getitem__(self, key: int) -> int | float:
        if not isinstance(key, int):
            raise TypeError("Indexes can only be integer values")

        return self.data[key]

    def __setitem__(self, key: int, new_value: int | float) -> None:
        if not isinstance(key, int):
            raise TypeError("Indexes can only be integer values")

        if not isinstance(new_value, (int, float)):
            raise TypeError("Vectors can only hold numbers")

        self.data[key] = new_value

    def __len__(self) -> int:
        return self.dims

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __add__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            raise TypeError("Cannot add up non-matrix objects to matrix")

        if self.dims != other.dims:
            raise TypeError("Cannot add up vectors with different dimensions")

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] + other[i])

        return Vector(result)

    def __sub__(self, other: Vector) -> Vector: 
        if not isinstance(other, Vector):
            raise TypeError("Cannot add up non-matrix objects to matrix")

        if self.dims != other.dims:
            raise TypeError("Cannot add up vectors with different dimensions")

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] - other[i])

        return Vector(result)

    def __rmul__(self, other: int | float) -> Vector: 
        if not isinstance(other, (int, float)):
            return NotImplemented

        result = []
        row_num = self.dims
        for i in range(row_num):
            result.append(self[i] * other)

        return Vector(result)

    def __neg__(self) -> Vector:
        return -1 * self

    def __str__(self) -> str:
         return f"{self.data}"
    
    @property
    def dims(self) -> int:
        return self.shape[0]

    @property
    def norm(self) -> float:
        return (sum(val ** 2 for val in self.data)) ** 0.5

    def copy(self) -> Vector:
        return Vector(self.data.copy())

    def to_list(self) -> list[int | float]:
        return self.data.copy()
