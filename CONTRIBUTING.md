# Contributing to LLM-Brain

Thank you for your interest in contributing! This project follows strict code quality standards.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/saschametz/llm-brain.git
cd llm-brain

# Install in development mode
make setup
# Or manually:
pip install -e ".[dev]"
pre-commit install
```

## Code Quality Standards

We use automated tools to ensure code quality. All checks must pass before merging.

### Linting & Formatting (Ruff)

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting:

```bash
# Auto-fix issues
make lint

# Check formatting
make format

# Check without modifying (CI mode)
make lint-check
make format-check
```

### Type Checking (MyPy)

[MyPy](https://mypy.readthedocs.io/) provides static type checking:

```bash
# Run type checker
make type-check

# Strict mode
make type-check-strict
```

**Important:** All functions must have type annotations:

```python
def process_memory(
    memory: Memory, 
    importance_threshold: float = 0.5
) -> list[Memory]:
    """Process memories above threshold.
    
    Args:
        memory: Memory to process
        importance_threshold: Minimum importance score
        
    Returns:
        List of processed memories
    """
    ...
```

### Testing

```bash
# Run all tests
make test

# With coverage report
make test-cov

# Enforce 80% coverage
make test-cov-fail
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Run manually on all files
make pre-commit

# Skip hooks (emergency only)
git commit -m "message" --no-verify
```

## Pull Request Process

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow the code style guidelines
3. **Run checks**: `make check-all` must pass
4. **Add tests**: Coverage must not decrease
5. **Update docs**: README, docstrings, or type annotations
6. **Submit PR**: Fill out the PR template

## Code Style Guidelines

### Python Style

- **Line length**: 100 characters
- **Quotes**: Double quotes for strings
- **Type annotations**: Required for all functions
- **Docstrings**: Google style, required for public APIs

```python
from typing import Optional

def calculate_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    metric: str = "cosine"
) -> float:
    """Calculate similarity between two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector  
        metric: Similarity metric ("cosine", "euclidean")
        
    Returns:
        Similarity score between 0.0 and 1.0
        
    Raises:
        ValueError: If metric is not supported
        
    Example:
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([0.0, 1.0])
        >>> calculate_similarity(a, b)
        0.0
    """
    if metric not in ("cosine", "euclidean"):
        raise ValueError(f"Unknown metric: {metric}")
    # Implementation...
```

### Import Order

Ruff handles import sorting. General order:

1. Standard library
2. Third-party packages
3. Local modules (llm_brain)

```python
import json
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel

from llm_brain.core.config import BrainConfig
from llm_brain.memory.models import Memory
```

### Naming Conventions

- **Classes**: `PascalCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`
- **Type variables**: `T`, `K`, `V` or `PascalCase`

## Architecture Guidelines

### Adding New Features

1. **Models**: Add Pydantic models in `memory/models.py`
2. **Storage**: Implement CRUD in `memory/storage.py`
3. **API**: Expose via `api/brain_api.py`
4. **Tests**: Add tests in `tests/`

### Database Changes

1. Update `core/schema.sql`
2. Add migration logic if needed
3. Update models
4. Test with fresh and existing databases

## Common Issues

### MyPy Errors

```bash
# Missing return type
def process():  # ❌
    ...

def process() -> Memory:  # ✅
    ...

# Missing argument type
def process(data):  # ❌
    ...

def process(data: dict[str, Any]) -> None:  # ✅
    ...
```

### Ruff Errors

```bash
# Auto-fix most issues
make lint

# Check specific rules
ruff check src/ --select E,W,I
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing code for examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
