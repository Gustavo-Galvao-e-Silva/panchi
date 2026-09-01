# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Copilot, and others) working in
the panchi repository. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) and the
full [contributing guide](https://gustavo-galvao-e-silva.github.io/panchi/contributing)
— this file is the short, machine-oriented version of the same rules.

## What panchi is

panchi is a Python-native linear algebra library built for **learning, not
performance**. The goal is clarity: every algorithm is implemented directly in
readable Python so a student can see the math happen. Optimize for understanding,
never for speed or cleverness.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Commands

Run all of these from the repository root, inside the virtual environment.

```bash
pytest tests/                 # run the test suite
pytest tests/ --cov=panchi    # with coverage
ruff check panchi tests       # lint
black panchi tests            # format (line length 88)
mypy panchi                   # type check (strict)
```

Config lives in `pyproject.toml`: black and ruff use line length 88; ruff selects
`E, W, F, I, B, C4, UP` (ignoring `E501`, `B008`); mypy runs strict, so **every
public function needs full type hints**.

## Project layout

```
panchi/
├── primitives/       # Vector, Matrix, and factory functions
├── algorithms/       # reductions, decompositions, other algorithms
└── visualizations/   # Animator2D / Animator3D and backends/ (matplotlib, manim)
tests/                # mirrors the package; test_*.py, Test* classes, test_* funcs
docs/                 # mkdocs-material sources
```

## Conventions

- **Clarity over cleverness.** A clear implementation beats a terse one. Avoid
  dense comprehensions and lambdas where an explicit loop reads better.
- **Explicit over implicit.** Spell out what an operation does.
- **NumPy-style docstrings** on every public function and class (summary,
  Parameters, Returns, Raises, Examples, See Also as appropriate).
- **Informative errors.** Error messages should explain what went wrong and why,
  including the offending value/type — they are a teaching tool.
- **Type hints and guards** everywhere; validate inputs (`TypeError`/`ValueError`).
- **Rich display is additive.** A new displayable type gets a `_repr_latex_` (reusing the
  builders in `panchi/_latex.py`) for Jupyter; `__str__`/`__repr__` stay terminal-first and
  unchanged. The core adds no display dependency — the LaTeX only renders when a notebook asks.
- **Collections of vectors are passed as a single `list[Vector]`**, never varargs —
  `plot_vectors([v1, v2])`, `VectorSpace([v1, v2])`, `gram_schmidt([v1, v2])`. This mirrors
  `Vector([...])` / `Matrix([[...]])` and the numpy/pandas sequence idiom. Don't add `*vectors`
  parameters to new public functions.
- **Minimal comments.** If you need many comments to explain code, refactor it
  instead. No debug `print()` statements.
- **Dependencies:** the core library (primitives + algorithms) uses only the
  Python standard library. Visualizations may use matplotlib; manim is an optional
  extra. Do not add core dependencies.
- **Tests are required** for every public function: normal operation, edge cases,
  error conditions, and type validation.

## Before opening a pull request

- `pytest tests/` passes.
- `ruff check panchi tests`, `black panchi tests`, and `mypy panchi` are clean.
- Docstrings added/updated for any changed public API.
- Branch named `feature/…`, `bug-fix/…`, or `documentation/…`.
- Fill in the pull request template and link any related issue.

The [full contributing guide](https://gustavo-galvao-e-silva.github.io/panchi/contributing)
is the source of truth for anything not covered here.
