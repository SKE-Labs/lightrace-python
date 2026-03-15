# Contributing to lightrace-python

Thank you for your interest in contributing!

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

```bash
git clone https://github.com/nichochar/lightrace-python.git
cd lightrace-python
uv sync --extra dev
uv run pre-commit install
```

### Commands

```bash
# Testing
uv run pytest -s -v tests/

# Linting & formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/lightrace

# Pre-commit (all checks)
uv run pre-commit run --all-files
```

## Code Quality

### Pre-commit Hooks

After `pre-commit install`, hooks run automatically on commit:
- **ruff-check**: Linter with auto-fix
- **ruff-format**: Code formatter

### Style Guide

- **Ruff** for linting and formatting (100 char line length)
- **MyPy** with strict mode for type checking
- Type hints required on all function definitions
- Use `from __future__ import annotations` for modern type syntax

### Testing

We use **pytest** for testing. Tests live in `tests/`.

```bash
# Run all tests
pytest -s -v tests/

# Run specific test
pytest -s -v tests/test_trace.py::TestTraceDecorator::test_root_trace
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure `pytest`, `ruff check .`, and `mypy src/lightrace` pass
4. Submit a PR with a clear description

## Release Process

Releases are automated via GitHub Actions. Maintainers trigger the release workflow which handles versioning, building, and PyPI publishing via OIDC.

## License

MIT
