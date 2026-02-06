# Mathrix

**Mathrix** is a small, Python-native linear algebra library designed for *learning*, *experimentation*, and *visual intuition*.

The goal is not performance. The goal is **clarity**.

Mathrix prioritizes:
- Explicit implementations of linear algebra concepts
- Readable, inspectable code
- Native visualizations and animations of vectors, matrices, and linear transformations
- A foundation for educational simulations in machine learning and physics

This project is meant for students, self-learners, and developers who want to *see* linear algebra happen, not just call NumPy and move on.

---

## Philosophy

Most linear algebra libraries optimize for speed and abstraction. Mathrix optimizes for understanding.

- Algorithms are implemented explicitly, not delegated to optimized backends.
- Objects behave like mathematical objects, not opaque arrays.
- Visualization is treated as a first-class feature, not an afterthought.

Think of Mathrix as a **laboratory notebook**, not a production engine.

---

## Features (Current & Planned)

### Core Linear Algebra
- Vectors and matrices implemented using native Python data structures
- Operator overloading for natural mathematical expressions
- Shape and consistency checks with informative errors

### Educational Focus
- Step-by-step algorithms (e.g. elimination, transformations)
- Clear separation between mathematical logic and visualization
- Designed to be read alongside a linear algebra textbook

### Visualization (In Progress)
- 2D vector and matrix transformation visualizations
- Animated demonstrations of:
  - Vector addition and scaling
  - Linear transformations
  - Basis changes

### Long-Term Goals
- Matrix decompositions (LU, QR, eigen methods)
- Interactive Jupyter support
- Educational machine learning demos (e.g. PCA, linear regression)
- Simple particle and physics simulations built on linear operators

---

## Installation

Currently, Mathrix is under active development and not yet feature-complete.

Clone the repository:

```bash
git clone https://github.com/Gustavo-Galvao-e-Silva/mathrix.git
cd mathrix
```

(Optional) Install in editable mode:

```bash
pip install -e .
```

---

## Python Version Support

- **Runtime compatibility:** Python **3.10+**
- **Development typing features:** Python **3.14 (or future versions)**

Mathrix uses modern type hinting techniques for internal development. Users **running** the library do **not** need Python 3.14.

---

## Example (Conceptual)

```python
from mathrix import Vector, Matrix

v = Vector([1, 2])
w = Vector([3, 1])

u = v + w
```

Visualization utilities allow you to see these operations geometrically.

---

## Project Status

Mathrix is in **early development**.

- APIs may change
- Algorithms are being added incrementally
- Visualization tooling is evolving

This is intentional. The project is built slowly to preserve conceptual clarity.

---

## Contributing

Contributions are welcome, especially in:
- Linear algebra algorithms
- Visualization tooling
- Documentation and examples

If you contribute, prioritize **readability over cleverness**.

---

## License

MIT License.

---

## Final Note

Mathrix exists to answer a simple question:

> “What is linear algebra actually *doing*?”

If that question matters to you, this library is for you.
