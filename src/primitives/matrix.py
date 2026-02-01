from vector import Vector


class Matrix:
    def __init__(self, data: list[list[int | float]]) -> None:
        if (
            not isinstance(data, list)
            or not all(isinstance(row, list) for row in data)
            or not all(
                isinstance(x, (int, float))
                for row in data
                for x in row
            )
        ):
            raise TypeError("Invalid matrix: all data must be lists of numbers")

        self.data = data
        self.shape = (len(self.data), len(self.data[0])) if data else (0, 0)

        if self.data:
            for row in self.data:
                if len(row) != self.shape[1]:
                    raise ValueError("Invalid matrix: all rows must have the same number of columns")

    def _apply_transformation(self, other: Vector | Matrix) -> Vector | Matrix:
        if isinstance(other, Vector):
            other_data = [[x] for x in other.data]
            return_vector = True
        else:
            other_data = other.data
            return_vector = False

        result = []
        for i in range(self.shape[0]):
            new_row = []
            for j in range(len(other_data[0])):
                val = 0
                for k in range(self.shape[1]):
                    val += self[i][k] * other_data[k][j]
                new_row.append(val)
            result.append(new_row)

        if return_vector:
            flattened = [item for row in result for item in row]
            return Vector(flattened)
        else:
            return Matrix(result)

    def __getitem__(self, key: int) -> list[int | float]:
        if not (type(key) == int):
            raise TypeError("Indexes can only be integer values")
        return self.data[key]

    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("Cannot add up matrices with different dimensions")

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self[i][j] + other[i][j])
            result.append(new_row)

        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("Cannot subtract matrices with different dimensions")

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self[i][j] - other[i][j])
            result.append(new_row)

        return Matrix(result)

    def __mul__(self, other: Vector | Matrix) -> Vector | Matrix: 
        if isinstance(other, (Matrix, Vector)):
            if self.shape[1] != (other.shape[0] if isinstance(other, Matrix) else other.dim):
                raise ValueError("Incompatible dimensions")

            return self._apply_transformation(other)
        else:
            return NotImplemented
    
    def __rmul__(self, other: int | float) -> Matrix:
        if not isinstance(other, (int, float)):
            return NotImplemented

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self[i][j] * other)
            result.append(new_row)

        return Matrix(result)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented

        if self.shape != other.shape:
            return False

        row_num, col_num = self.shape
        for i in range(row_num):
            for j in range(col_num):
                if self[i][j] != other[i][j]:
                    return False

        return True

    def __str__(self) -> str:
         rows = ",\n ".join(str(row) for row in self.data)
         return f"[{rows}]"
    
    def transpose(self) -> Matrix:
        result = []
        row_num, col_num = self.shape
        for j in range(col_num):
            new_row = []
            for i in range(row_num):
                new_row.append(self.data[i][j])
            result.append(new_row)

        return Matrix(result)

    @property
    def T(self) -> Matrix:
        return self.transpose()

    def apply(self, vec: Vector) -> Vector:
        return self * vec
