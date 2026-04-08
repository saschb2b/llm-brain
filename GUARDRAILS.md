# Python Guardrails Guide

Your TypeScript toolchain → Python equivalents:

| TypeScript | Python | Purpose |
|------------|--------|---------|
| **Prettier** | **Ruff Format** | Code formatting |
| **ESLint** | **Ruff Lint** | Linting with 20+ rule categories |
| **TypeScript** | **MyPy** | Static type checking |
| **Jest** | **pytest** | Testing framework |
| **Husky** | **pre-commit** | Git hooks |

## Quick Commands

```bash
# Setup
make setup              # Install everything + pre-commit hooks

# Daily development  
make lint               # Auto-fix linting issues
make format             # Format code
make type-check         # Check types
make test               # Run tests
make test-cov           # Run tests with coverage

# CI checks (what GitHub Actions runs)
make check-all          # Run all checks (lint + format + types + test)
```

## The Stack Explained

### 1. Ruff (Replaces Black + Flake8 + isort)

Ultra-fast (10-100x faster than alternatives) linter and formatter written in Rust.

```bash
# Configuration in pyproject.toml
[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle (PEP 8)
    "F",         # Pyflakes
    "I",         # isort (import sorting)
    "N",         # pep8-naming
    "D",         # pydocstyle (docstrings)
    "UP",        # pyupgrade (Python upgrade checks)
    "B",         # flake8-bugbear (bugs/design issues)
    "C4",        # flake8-comprehensions
    "SIM",       # flake8-simplify
    "ARG",       # unused arguments
    "PL",        # Pylint rules
    "PERF",      # Performance issues
    "S",         # Security (bandit)
    "RUF",       # Ruff-specific rules
]
```

### 2. MyPy (TypeScript equivalent)

Static type checker with `strict = true` mode.

```python
# ✅ Good - fully typed
def process_vector(
    vector: np.ndarray,
    config: BrainConfig | None = None
) -> list[Memory]:
    ...

# ❌ Bad - untyped
def process_vector(vector, config=None):
    ...
```

### 3. pytest + pytest-cov

Testing with coverage enforcement (80% minimum).

```bash
pytest tests/ --cov=llm_brain --cov-fail-under=80
```

### 4. pre-commit

Runs all checks before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## CI/CD Pipeline

GitHub Actions runs on every PR:

1. **Lint & Type Check** (fast feedback ~30s)
2. **Tests** (across Python 3.9-3.12, Ubuntu/Windows/macOS)
3. **Coverage** (enforces 80% minimum)
4. **Security Scan** (Bandit + Safety)
5. **Build** (package verification)

## Strictness Level

This setup is **very strict** - comparable to strict TypeScript:

- ✅ All functions must be typed
- ✅ All imports must be sorted
- ✅ Docstrings required for public APIs
- ✅ No unused variables/imports
- ✅ Cyclomatic complexity < 10
- ✅ 80% test coverage minimum
- ✅ Security checks enabled

## VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.codeActionsOnSave": {
    "source.fixAll.ruff": "explicit",
    "source.organizeImports.ruff": "explicit"
  },
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "mypy-type-checker.args": ["--strict"]
}
```

Extensions to install:
- Ruff (charliermarsh.ruff)
- MyPy Type Checker (ms-python.mypy-type-checker)
- Python (ms-python.python)

## Common Commands

```bash
# Full development workflow
make setup              # One-time setup
make lint               # Fix issues while coding
make test               # Test while coding
make check-all          # Before committing

# Individual tools
ruff check src/ --fix   # Auto-fix lint errors
ruff format src/        # Format code
mypy src/llm_brain      # Type check
pytest tests/ -v        # Verbose tests
pytest tests/ --pdb     # Debug failing test

# Pre-commit (runs automatically on git commit)
pre-commit run --all-files    # Run manually
pre-commit run ruff           # Run specific hook
```

## Disable Rules (Use Sparingly)

```python
# noqa: disable specific rule
result = some_function()  # noqa: F841 (unused variable intentionally)

# type: ignore for mypy
value = untyped_lib.call()  # type: ignore

# mypy: ignore entire file
# mypy: ignore-errors
```

## Why This Stack?

| Tool | Why |
|------|-----|
| Ruff | 10-100x faster than alternatives, unified tool |
| MyPy | Industry standard, strict mode catches real bugs |
| pytest | Most popular, great fixtures, plugins |
| pre-commit | Prevents bad commits, automated enforcement |
| Makefile | Simple command interface, IDE agnostic |
