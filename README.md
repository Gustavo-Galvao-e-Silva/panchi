<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo/panchi_logo_white.png">
  <img src="docs/assets/logo/panchi_logo_color.png" alt="panchi" width="280">
</picture>

# panchi

**panchi** is a Python-native linear algebra library designed for learning, experimentation, and visual intuition.

The goal is not performance. The goal is **clarity**.

[![TestCI](https://github.com/Gustavo-Galvao-e-Silva/panchi/workflows/TestCI/badge.svg)](https://github.com/Gustavo-Galvao-e-Silva/panchi/actions/workflows/panchi-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/panchi.svg)](https://pypi.org/project/panchi/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Why panchi?

Most linear algebra libraries optimize for speed and abstraction. panchi optimizes for **understanding**.

panchi is built for students who want to see the math happen, educators who need transparent implementations, and anyone who has ever wondered what linear algebra is *actually about*.

Think of it as a **lab**, not a production engine.

---

## Philosophy

1. **Explicit over implicit** – Algorithms are implemented directly, not delegated to opaque backends
2. **Readable over clever** – Code prioritizes clarity and educational value over terse optimizations and pythonisms
3. **Mathematical over computational** – Objects behave like mathematical entities with proper operator overloading
4. **Visual by default** – Visualization is a first-class feature, not an afterthought
5. **Informative errors** – Error messages guide learning by explaining what went wrong and why

---

## Installation

```bash
pip install panchi
```

Requires Python 3.10+. For optional Manim-powered visualizations:

```bash
pip install panchi[manim]
```

---

## A Taste

```python
import panchi as pan
from panchi.algorithms import rref, solve

# Vectors and matrices behave like their mathematical counterparts
A = pan.Matrix([[1, 2, 3], [2, 5, 7], [0, 1, 2]])
b = pan.Vector([1, 0, 0])

# Solve Ax = b — see the status, not just the answer
result = solve(A, b)
print(result.status)    # 'unique'
print(result.solution)  # the solution vector x

# Row reduction shows every step it takes
reduction = rref(A)
print(reduction)        # full step-by-step walkthrough
print(reduction.rank)   # 3
```

---

## Documentation

Full documentation, user guides, and the API reference are available at
**[https://gustavo-galvao-e-silva.github.io/panchi/](https://gustavo-galvao-e-silva.github.io/panchi/)**

---

## Contributing

panchi welcomes contributions that align with its educational mission. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Thanks to all of our contributors, whose names can be found in [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Acknowledgments

panchi is inspired by Gilbert Strang's *Introduction to Linear Algebra* and 3Blue1Brown's *Essence of Linear Algebra* — resources that make the subject visible, not just computable.
