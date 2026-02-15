# mathrix

**mathrix** is a Python-native linear algebra library designed for learning, experimentation, and visual intuition.

The goal is not performance. The goal is **clarity**.

[![TestCI](https://github.com/Gustavo-Galvao-e-Silva/mathrix/workflows/TestCI/badge.svg)](https://github.com/Gustavo-Galvao-e-Silva/mathrix/actions/workflows/mathrix-test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Why mathrix?

Most linear algebra libraries optimize for speed and abstraction. mathrix optimizes for **understanding**.

mathrix is built for:
- **Students** learning linear algebra who want to see the math happen, not just trust a black box
- **Educators** who need transparent implementations to demonstrate concepts in lectures or workshops
- **Self-learners** building intuition through experimentation and visual feedback
- **Researchers** prototyping algorithms where readability matters more than runtime
- **Developers** who want to understand what's happening under the hood before optimizing

Think of mathrix as a **laboratory notebook**, not a production engine.

---

## Philosophy

1. **Explicit over implicit** – Algorithms are implemented directly, not delegated to opaque backends
2. **Readable over clever** – Code prioritizes clarity and educational value over terse optimizations
3. **Mathematical over computational** – Objects behave like mathematical entities with proper operator overloading
4. **Visual by default** – Visualization is a first-class feature, not an afterthought
5. **Informative errors** – Error messages guide learning by explaining what went wrong and why

---

## Features

### ✅ Currently Available

**Core Primitives**
- `Vector` and `Matrix` classes with native Python data structures
- Natural mathematical syntax through operator overloading (`+`, `-`, `*`, `**`)
- Comprehensive shape checking with educational error messages

**Vector Operations**
- Addition, subtraction, scalar multiplication
- Magnitude calculation and normalization
- Dot product and cross product (3D)
- Unit vectors and zero vectors

**Matrix Operations**
- Addition, subtraction, matrix multiplication
- Scalar multiplication and matrix powers
- Transpose operations
- Matrix-vector transformations
- Trace calculation for square matrices

**Utilities**
- Identity matrices and zero matrices
- Random vectors and matrices with configurable ranges
- 2D and 3D rotation matrices
- Diagonal matrix construction

### 🚧 In Development

- Matrix determinants and inverses
- Matrix decompositions (LU, QR, eigenvalue methods)
- Interactive visualization toolkit
- Animated transformations
- Jupyter notebook integration

### 🔮 Roadmap

- Step-by-step algorithm demonstrations
- Educational machine learning examples (PCA, linear regression)
- Physics simulations using linear operators
- Extended documentation with visual explanations

---

## Installation

mathrix is under active development. To use the latest version:

```bash
git clone https://github.com/Gustavo-Galvao-e-Silva/mathrix.git
cd mathrix
pip install -e .
```

**Requirements:** Python 3.10 or higher

---

## Quick Start

```python
from mathrix.primitives import Vector, Matrix
from mathrix.operations import dot, identity, rotation_matrix_2d
from math import pi

# Vector operations
v = Vector([3, 4])
magnitude = v.magnitude
normalized = v.normalize()

# Matrix transformations
rotation = rotation_matrix_2d(pi / 2)
point = Vector([1, 0])
rotated = rotation * point

# Matrix algebra
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = A * B
trace = C.trace

# Identity and special matrices
I = identity(3)
product = A * A.right_identity
```

---

## Use Cases

### 1. **Learning Linear Algebra**
Work through textbook problems with code that mirrors mathematical notation:
```python
# Verify that (AB)^T = B^T A^T
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

left_side = (A * B).T
right_side = B.T * A.T

assert left_side == right_side
```

### 2. **Teaching and Demonstrations**
Show students exactly how matrix multiplication works:
```python
# Every step is visible and inspectable
m = Matrix([[1, 2], [3, 4]])
v = Vector([5, 6])

result = m * v
```

### 3. **Algorithm Prototyping**
Develop and test linear algebra algorithms before optimization:
```python
def gram_schmidt(vectors):
    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            projection = (dot(v, u) / dot(u, u)) * u
            v = v - projection
        orthogonal.append(v.normalize())
    return orthogonal
```

### 4. **Visual Exploration**
*(Coming soon)* Experiment with transformations and see results immediately:
```python
# Visualize how rotation matrices affect vectors
rotation = rotation_matrix_2d(pi / 4)
vectors = [Vector([1, 0]), Vector([0, 1]), Vector([1, 1])]
transformed = [rotation * v for v in vectors]

plot_transformation(vectors, transformed)
```

---

## Contributing

Contributions are welcome and encouraged! mathrix values clarity and educational merit above all else.

### What We're Looking For

**High Priority**
- Matrix determinant and inverse implementations
- Decomposition algorithms (LU, QR, SVD)
- Visualization utilities for 2D and 3D transformations
- Educational examples and tutorials
- Documentation improvements

**Also Welcome**
- Bug fixes with test cases
- Performance improvements that don't sacrifice readability
- Additional utility functions
- More comprehensive error messages

### Contribution Guidelines

1. **Readability First** – Code should be understandable by someone learning linear algebra
2. **No External Dependencies** – Core functionality uses only Python standard library
3. **Comprehensive Tests** – Include pytest tests demonstrating correctness
4. **Descriptive Docstrings** – Follow NumPy documentation style
5. **Mathematical Accuracy** – Verify implementations against textbook algorithms

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/determinant-calculation`)
3. Write clear, well-documented code with tests
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request with a detailed description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Documentation

Full documentation is available at [mathrix.readthedocs.io](https://mathrix.readthedocs.io) *(coming soon)*

- **API Reference** – Complete function and class documentation
- **Tutorials** – Step-by-step guides for common operations
- **Examples** – Real-world use cases and demonstrations
- **Theory** – Mathematical background and explanations

---

## Project Status

**Current Version:** 0.1.0-alpha (Early Development)

mathrix is actively developed but not yet feature-complete. APIs may change as the library evolves. The project is built deliberately and carefully to maintain clarity and educational value.

---

## Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/Gustavo-Galvao-e-Silva/mathrix/issues)
- **Discussions:** Ask questions in [GitHub Discussions](https://github.com/Gustavo-Galvao-e-Silva/mathrix/discussions)
- **Email:** contact@mathrix-dev.org *(coming soon)*

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Acknowledgments

mathrix is inspired by educational resources that prioritize understanding:
- Gilbert Strang's *Introduction to Linear Algebra*
- 3Blue1Brown's *Essence of Linear Algebra* video series
- The desire to make linear algebra accessible and visual

---

## Final Note

mathrix exists to answer a simple question:

> **"What is linear algebra actually *doing*?"**

If you've ever felt frustrated by libraries that hide the mathematics behind abstractions, or if you want to *see* transformations happen rather than just trust the output, mathrix is for you.

Linear algebra is beautiful. Let's make it visible.
