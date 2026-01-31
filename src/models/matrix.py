class Matrix:
    # TODO: add robust validation for dimension checking
    def __init__(self, data: list[list[int]] | list[list[float]]) -> None:
        if (
            not isinstance(data, list)
            or not all(isinstance(row, list) for row in data)
            or not all(
                isinstance(x, (int, float))
                for row in data
                for x in row
            )
        ):
            raise ValueError("Matrices can only accept numerical values")

        self.data = data
        self.shape = (len(self.data), len(self.data[0]))

    def __getitem__(self, key: int) -> list[int] | list[float]:
        if not (type(key) == int):
            raise ValueError("Indexes can only be integer values")
        return self.data[key]

    def __add__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            raise ValueError("Cannot add up non-matrix objects to matrix")

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

    def __sub__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            raise ValueError("Cannot add up non-matrix objects to matrix")

        if self.shape != other.shape:
            raise ValueError("Cannot add up matrices with different dimensions")

        result = []
        row_num, col_num = self.shape
        for i in range(row_num):
            new_row = []
            for j in range(col_num):
                new_row.append(self[i][j] - other[i][j])
            result.append(new_row)

        return Matrix(result)

    def __mul__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented

        self_row_num, self_col_num = self.shape
        other_row_num , other_col_num = other.shape 
        if self_col_num != other_row_num :
            raise ValueError("Cannot multiply matrices with incompatible dimensions")

        result = []
        for i in range(self_row_num):
            new_row = []
            for j in range(other_col_num):
                val = 0
                for k in range(other_row_num):
                    val += self.data[i][k] * other.data[k][j]
                new_row.append(val)
            result.append(new_row)

        return Matrix(result)
    
    def __rmul__(self, other: int | float) -> "Matrix":
        if not isinstance(other, (int, float)):
            return NotImplemented

        result = []
        for i in range(self.shape[0]):
            new_row = []
            for j in range(self.shape[1]): 
                new_row.append(self.data[i][j] * other)
            result.append(new_row)

        return Matrix(result)

    def __str__(self) -> str:
         rows = ",\n ".join(str(row) for row in self.data)
         return f"[{rows}]"

    @property
    def shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def T(self) -> "Matrix":
        pass
