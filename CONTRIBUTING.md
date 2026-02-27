# Contributing to panchi

Thank you for your interest in contributing to panchi! This guide will help you understand our development philosophy and contribution process.

---

## Table of Contents

1. [Philosophy and Values](#philosophy-and-values)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Standards](#code-standards)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Contribution Workflow](#contribution-workflow)
8. [What We're Looking For](#what-were-looking-for)
9. [Code Review Process](#code-review-process)
10. [Community Guidelines](#community-guidelines)

---

## Philosophy and Values

panchi is built on the following core principles. Every contribution should align with them:

### 1. Clarity Over Optimization
Code should be understandable by someone learning linear algebra. A clear implementation beats a clever one.

**Good:**
```python
def dot(vector_1: Vector, vector_2: Vector) -> float:
    if vector_1.dims != vector_2.dims:
        raise ValueError("Vector dimensions must match for dot product.")
    
    n = vector_1.dims
    return sum(vector_1[i] * vector_2[i] for i in range(n))
```

**Avoid:**
```python
def dot(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))
```

The second version is more concise but less transparent about what's happening.

### 2. Educational Value First
Implementations should focus on teaching linear algebra, instead of simplifying or obscuring calculations.

### 3. Mathematical Correctness
Implementations must match standard mathematical definitions and textbook algorithms.

### 4. Informative Errors
Error messages should explain what went wrong and why, helping users learn.

### 5. Minimal External Dependencies
The core library (i.e. vector and matrix implementations and operations) exclusevily uses Python's standard library to remain transparent and accessible. 

Additional features such as visualizations should strive to minimize dependencies, especially those that rely on packages that are not solely setup with pip (e.g. Manim, which relies on C libraries, though this case is provided as an optional package for prettier visualizations, and greatly improves learning).

### 6. Break Any of These Principles for Education
Although these are guides, if they are blocking the advancement of this library's purpose: break them. It's that simple!

---

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- pytest (for testing)
- Basic understanding of linear algebra
- Familiarity with type hints

### First Contribution Ideas
- Fix typos in documentation
- Add test cases for edge conditions
- Improve error messages
- Write examples in the docs
- Implement small utility functions

---

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/panchi.git
cd panchi
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install in Development Mode
```bash
pip install -e ".[dev]"
```

### 4. Verify Installation
```bash
python -c "from panchi.primitives import Vector; print(Vector([1, 2, 3]))"
pytest tests/
```

### 5. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

---

## Code Standards

### File Organization
```
panchi/
├── primitives/        # Core classes (Vector, Matrix)
├── operations.py      # Standalone functions
└── visualizations/     # Plotting utilities
```

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `Vector`, `Matrix`)
- **Functions:** `snake_case` (e.g., `dot`, `cross`, `rotation_matrix_2d`)
- **Constants:** `UPPER_SNAKE_CASE` (if needed)
- **Private methods:** `_leading_underscore` (e.g., `_apply_transformation`)

### Type Hints
Always include type hints. They're documentation and help catch errors.

```python
def zero_vector(dims: int) -> Vector:
    if not isinstance(dims, int):
        raise TypeError(f"Dimension must be an integer. Got {type(dims).__name__}.")
    if dims <= 0:
        raise ValueError(f"Dimension must be positive. Got {dims}.")
    
    return Vector([0 for _ in range(dims)])
```

### No Comments (Except Docstrings)
Code should be self-explanatory. If it needs a comment, try rewriting it or explaining in a docstring.

**Don't do this:**
```python
# Calculate the dot product
result = 0
for i in range(len(v1)):
    result += v1[i] * v2[i]  # Multiply corresponding elements
```

**Do this:**
```python
def dot(vector_1: Vector, vector_2: Vector) -> float:
    n = vector_1.dims
    return sum(vector_1[i] * vector_2[i] for i in range(n))
```

### No Debug/Print Statements
Never commit code with `print()` statements for debugging. Use proper logging if necessary, or remove them entirely.

### Explicit Over Implicit
Be clear about what operations are happening.

```python
# Good: Clear what's happening
row_count = self.rows
col_count = self.cols
result = []
for i in range(row_count):
    new_row = []
    for j in range(col_count):
        new_row.append(self[i][j] + other[i][j])
    result.append(new_row)

# Avoid: Too compact, harder to follow
return Matrix([[self[i][j] + other[i][j] for j in range(self.cols)] for i in range(self.rows)])
```

---

## Testing Requirements

### Test Every Public Function
All public functions and methods must have tests covering:
- Normal operation
- Edge cases
- Error conditions
- Type validation

### Test File Organization
```
tests/
├── test_vector.py           # Vector class tests
├── test_matrix.py           # Matrix class tests
├── test_operations.py       # Operations function tests
└── test_integration.py      # Combined operation tests
```

### Test Naming Convention
```python
class TestVectorAddition:
    def test_add_same_dimension(self):
        # Test normal case
        
    def test_add_different_dimensions(self):
        # Test error condition
        
    def test_add_zero_vector(self):
        # Test edge case
```

### Writing Good Tests
```python
def test_matrix_multiplication_2x2(self):
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    result = m1 * m2
    
    expected = Matrix([[19, 22], [43, 50]])
    assert result == expected
    assert result.shape == (2, 2)
```

### Test Error Messages
```python
def test_add_different_dimensions(self):
    v1 = Vector([1, 2])
    v2 = Vector([1, 2, 3])
    
    with pytest.raises(TypeError) as excinfo:
        result = v1 + v2
    
    assert "dimension" in str(excinfo.value).lower()
```

### Run Tests Before Submitting
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_vector.py

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=panchi
```

---

## Documentation Standards

### Docstring Format
Use NumPy/Google style docstrings for all public functions and classes.

```python
def rotation_matrix_2d(angle: int | float, radians: bool = True) -> Matrix:
    """
    Create a 2D rotation matrix.
    
    Constructs a matrix that rotates points in 2D space by the specified
    angle. The rotation is counterclockwise when using standard coordinate
    orientation.
    
    Parameters
    ----------
    angle : int | float
        Rotation angle. Interpreted as radians if radians=True,
        degrees if radians=False.
    radians : bool, optional
        If True (default), angle is in radians. If False, angle is in degrees.
    
    Returns
    -------
    Matrix
        A 2×2 rotation matrix.
    
    Examples
    --------
    >>> from math import pi
    >>> rot90 = rotation_matrix_2d(pi / 2)
    >>> point = Vector([1, 0])
    >>> rotated = rot90 * point
    >>> print(rotated)
    [0, 1]
    
    See Also
    --------
    rotation_matrix_3d : 3D rotation about an arbitrary axis
    """
    angle_radians = angle if radians else (angle * pi / 180)
    cos_angle = cos(angle_radians)
    sin_angle = sin(angle_radians)
    
    return Matrix([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
```

### Docstring Sections
Include these sections as appropriate:
- **Short summary** (one line)
- **Extended description** (optional, for complex functions)
- **Parameters** (all parameters with types and descriptions)
- **Returns** (return type and description)
- **Raises** (exceptions that can be raised)
- **Examples** (working code examples)
- **See Also** (related functions)
- **Notes** (implementation details, algorithms used)

### Parameter Descriptions
Be specific about constraints:

```python
Parameters
----------
dims : int
    Number of components in the vector. Must be positive.
index : int
    Position of the 1 component (0-indexed). Must be in range [0, dims).
```

### Examples in Docstrings
- Use actual code that can be copied and run
- Show expected output
- Cover common use cases
- Include edge cases when helpful

---

## Contribution Workflow

### 1. Create an Issue (Optional but Recommended)
Before starting work, create an issue describing:
- What you want to add/fix
- Why it's valuable
- How you plan to implement it

This helps avoid duplicate work and ensures alignment with project goals.

### 2. Write Your Code
Follow all code standards and write tests as you go.

### 3. Update Documentation
- Add/update docstrings
- Update API documentation if needed
- Add examples if relevant

### 4. Run Tests
```bash
pytest tests/ -v
```

All tests must pass before submitting.

### 5. Commit Your Changes
Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add determinant calculation for square matrices"
git commit -m "Fix dimension check in matrix multiplication"
git commit -m "Improve error message for incompatible vector addition"

# Avoid vague messages
git commit -m "Update code"
git commit -m "Fix bug"
git commit -m "Changes"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
OR
git push origin bug-fix/bug-name
OR
git push origin documentation/change-name
```

Then create a pull request on GitHub with:
- Clear title summarizing the change
- Description of what changed and why
- Reference to related issues
- Examples of the new functionality

### 7. Respond to Review Feedback
Reviewers may request changes. This is normal and helps maintain quality.

---

## What We're Looking For

### High Priority Contributions

#### 1. Matrix Determinant Calculation
Implement determinant calculation using cofactor expansion or LU decomposition.

**Requirements:**
- Works for any square matrix
- Clear, educational implementation
- Comprehensive tests
- Handles edge cases

#### 2. Matrix Inverse
Implement matrix inversion using Gauss-Jordan elimination.

**Requirements:**
- Checks for invertibility
- Clear error message for singular matrices
- Works for any invertible square matrix
- Tests verify A × A^(-1) = I

#### 3. LU Decomposition
Factor matrices into lower and upper triangular matrices.

**Requirements:**
- Returns L and U matrices
- Includes pivoting for numerical stability
- Educational implementation
- Verification tests (check that L × U = A)

#### 4. QR Decomposition
Implement using Gram-Schmidt process.

**Requirements:**
- Returns Q (orthogonal) and R (upper triangular)
- Clear step-by-step implementation
- Verification tests

#### 5. Visualization Tools
Create functions to visualize vectors and transformations.

**Requirements:**
- Simple, clear visualizations
- Works with existing Vector/Matrix classes
- Matplotlib or Manim (optional dependency) 
- Examples in documentation

### Also Welcome

- **Bug fixes** with test cases
- **Documentation improvements**
- **Additional utility functions** (with justification)
- **Performance improvements** that maintain clarity
- **Educational examples**
- **Better error messages**

### Not Currently Seeking

- Alternative implementations of existing features
- Features requiring external dependencies in core library
- Highly optimized but opaque implementations
- Features without clear educational value

---

## Code Review Process

### What Reviewers Check

1. **Correctness:** Does it work? Are there bugs?
2. **Clarity:** Is the code easy to understand?
3. **Tests:** Are there comprehensive tests?
4. **Documentation:** Are docstrings complete and accurate?
5. **Style:** Does it follow project conventions?
6. **Educational value:** Does it help users learn?

### Approval Requirements

Pull requests need:
- ✅ All tests passing
- ✅ At least one maintainer approval
- ✅ No unresolved conversations
- ✅ Updated documentation

---

## Community Guidelines

### Be Respectful
- Treat all contributors with respect
- Assume good intentions
- Provide constructive feedback
- Welcome newcomers

### Be Collaborative
- Share knowledge
- Help others learn
- Credit contributions
- Build on each other's work

### Be Patient
- Code review takes time
- Discussions may be lengthy
- Learning happens at different paces

### Ask Questions
- No question is too basic
- Clarify requirements before implementing
- Discuss design decisions
- Seek feedback early

---

## Getting Help

- **Questions about implementation:** Open a GitHub issue
- **Clarifying requirements:** Comment on relevant issue/PR
- **General discussion:** Use GitHub Discussions
- **Private concerns:** Email maintainers (coming soon)

---

## Recognition

All contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README (for significant contributions)

---

## License

By contributing to panchi, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to panchi! Your work helps make linear algebra more accessible and understandable for everyone!**
