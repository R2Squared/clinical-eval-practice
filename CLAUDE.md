# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Freshly scaffolded `uv` project (src layout, `uv_build` backend). The only code is a placeholder `main()` in `src/clinical_eval_practice/__init__.py`, and `test_setup.py` is a placeholder (`assert True`). README.md is empty. Update this file as real structure emerges.

## Commands

Package management and running use `uv` (Python 3.12, pinned in `.python-version`; the lockfile is `uv.lock`):

- Install/sync deps: `uv sync`
- Run the entry point: `uv run clinical-eval-practice` (maps to `clinical_eval_practice:main`)
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest test_setup.py::test_setup`
- Add a dependency: `uv add <pkg>` (use `uv add --dev <pkg>` for dev-only); don't hand-edit `uv.
- Run tests with: uv run pytest
- Never add patient-identifiable data.
- Explain every change before making it.
lock`

No linter or formatter is configured yet.

## Notes

- Dependencies are pandas and scikit-learn (data handling and model evaluation metrics); pytest is the dev group.
- `gh` is also listed as a dependency. It is the unrelated PyPI package, not GitHub's CLI, and is likely unintended. Confirm before relying on it.
- `test_setup.py` sits at the repo root, outside `src/`. There is no `tests/` directory yet.
