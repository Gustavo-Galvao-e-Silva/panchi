# Contributing

Thanks for your interest in contributing to panchi!

The full contribution guide is in the [documentation](https://gustavo-galvao-e-silva.github.io/panchi/contributing).

To keep this short and sweet: clarity first, tests required, NumPy-style docstrings, and mathematical correctness over cleverness.

We welcome everyone — please also read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Working with AI agents & quick reference

Using a coding agent (Claude Code, Cursor, Copilot, …)? Point it at
[AGENTS.md](AGENTS.md) — it's the agent-oriented summary of these rules, with the
project layout and exact commands.

```bash
pip install -e ".[dev]"       # setup (inside a virtual environment)
pre-commit install            # optional: run ruff + black on every commit
pytest tests/ -m "not manim"  # fast tests (skips manim rendering)
pytest tests/                 # full test suite
ruff check panchi tests       # lint (CI blocks on this)
black panchi tests            # format (CI blocks on this)
mypy panchi/path/to/file.py   # type check the files you touch
```

The codebase has known mypy errors that are being cleaned up, so `mypy panchi` isn't
clean yet. Just don't add new mypy errors in the files you touch.
