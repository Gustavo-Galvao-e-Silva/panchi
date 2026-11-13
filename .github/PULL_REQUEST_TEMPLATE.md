## Summary

What does this change do, and why?

## Related issue

Closes #

## Checklist

- [ ] `pytest tests/` passes
- [ ] `ruff check panchi tests` and `black --check panchi tests` are clean (CI enforces this)
- [ ] No new mypy errors in the files I touched
- [ ] `mkdocs build --strict` passes (if docs or docstrings changed)
- [ ] New/changed public functions have NumPy-style docstrings
- [ ] Tests cover normal operation, edge cases, error conditions, and type validation
- [ ] Change follows panchi's philosophy — clarity over cleverness, educational value first

## Notes for reviewers

Anything worth calling out — design decisions, trade-offs, follow-ups.
