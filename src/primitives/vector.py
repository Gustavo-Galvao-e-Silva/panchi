class Vector:
    def __init__(self, data: list[float | int]) -> None:
        if not (
            isinstance(data, list) and
            all(isinstance(x, (int, float)) for x in data)
        ):
            raise TypeError("Vectors can only be created with lists of numbers")

        self.data = data
        self.shape = (len(data), 1)

    def __getitem__(self, key: int) -> int | float:
        if not (type(key) == int):
            raise TypeError("Indexes can only be integer values")

        return self.data[key]

    def __add__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            raise TypeError("Cannot add up non-matrix objects to matrix")

        if self.dim != other.dim:
            raise TypeError("Cannot add up vectors with different dimensions")

        result = []
        row_num = self.dim
        for i in range(row_num):
            result.append(self[i] + other[i])

        return Vector(result)

    def __sub__(self, other: Vector) -> Vector: 
        if not isinstance(other, Vector):
            raise TypeError("Cannot add up non-matrix objects to matrix")

        if self.dim != other.dim:
            raise TypeError("Cannot add up vectors with different dimensions")

        result = []
        row_num = self.dim
        for i in range(row_num):
            result.append(self[i] - other[i])

        return Vector(result)

    def __rmul__(self, other: int | float) -> Vector: 
        if not isinstance(other, (int, float)):
            return NotImplemented

        result = []
        row_num = self.dim
        for i in range(row_num):
            result.append(self[i] * other)

        return Vector(result)

    def __str__(self) -> str:
         return f"{self.data}"
    
    @property
    def dim(self) -> int:
        return self.shape[0]
