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
pip install -e ".[dev]"     # setup (inside a virtual environment)
pytest tests/               # run tests
ruff check panchi tests     # lint
black panchi tests          # format
mypy panchi                 # type check (strict)
```
