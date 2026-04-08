# Agent Guidelines for LLM-Brain

This document helps AI agents avoid common issues when working on this codebase.

## Critical Patterns to Follow

### 1. Syntax Errors in String Literals

**DON'T:** Missing closing quotes in multi-line strings.
```python
# WRONG - Missing closing triple-quote
self.connection.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
        memory_id TEXT PRIMARY KEY,
        embedding FLOAT[{dims}]
    )
"")  # <-- Only 2 quotes!
```

**DO:** Always close triple-quoted strings properly.
```python
# CORRECT
self.connection.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
        memory_id TEXT PRIMARY KEY,
        embedding FLOAT[{dims}]
    )
""")  # <-- 3 quotes
```

### 2. sqlite3.Row vs dict Access

**DON'T:** Use `.get()` method on sqlite3.Row objects.
```python
# WRONG - sqlite3.Row has no .get() method
context_window_id = row.get("context_window_id")
```

**DO:** Use direct indexing with `row["key"]`.
```python
# CORRECT
context_window_id = row["context_window_id"]
```

### 3. Foreign Key Constraints

When deleting a memory, delete related records in the correct order:
1. Delete from `vec_memories` first (if using sqlite-vec)
2. Delete from `relations` table
3. Delete from `cognition_log` table
4. Finally delete from `memories` table

The schema uses `ON DELETE CASCADE` for relations but NOT for cognition_log.

### 4. Ruff Lint Compliance

Always run these before committing:
```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/ --check
python -m mypy src/llm_brain --strict
python -m pytest tests/ -v
```

**Common fixes needed:**
- **PLC0415**: Move imports to top-level or add `# noqa: PLC0415`
- **S110**: Use `contextlib.suppress(Exception)` instead of `try/except/pass`
- **SIM105**: Same as S110 - use `suppress()`
- **E501**: Line too long (>100 chars) - wrap strings
- **SIM108**: Use ternary operator `x if condition else y`
- **SIM102**: Combine nested if statements with `and`
- **PLW0603**: Global statements need `# noqa: PLW0603` (singleton pattern)
- **PTH123**: Use `Path.open()` instead of `open()`
- **PLW2901**: Don't reassign loop variables - use different name

### 5. Type Annotations

**Union types for Path/str:**
```python
from typing import Union

brain_path: Union[Path, str] = field(default_factory=lambda: Path.home() / ".kimi-brain")

def __post_init__(self) -> None:
    if isinstance(self.brain_path, str):
        self.brain_path = Path(self.brain_path)
```

**Add type: ignore for known false positives:**
```python
return self.brain_path / self.db_name  # type: ignore[operator]
```

### 6. Global Singleton Pattern

This codebase uses global singletons for config/database/graph. When you need to use `global` keyword:

```python
global _config  # noqa: PLW0603
```

This is intentional - don't try to refactor away the global state, just suppress the lint.

### 7. Import Organization

Standard library imports go at the top. Only use inline imports when:
1. The module is optional (like `sqlite_vec`, `kuzu`)
2. There's a circular import issue
3. The import is expensive and rarely used

Always add `# noqa: PLC0415` for necessary inline imports.

### 8. Exception Handling

**For optional features (sqlite-vec, kuzu):**
```python
from contextlib import suppress

with suppress(Exception):
    sqlite_vec.load(conn)
```

**For expected errors:**
```python
try:
    do_something()
except ExpectedError:  # noqa: S110
    pass  # Comment explaining why it's ok to ignore
```

### 9. SQL String Construction

**Parameterized queries are safe:**
```python
cursor.execute("SELECT * FROM memories WHERE tier = ?", (tier_value,))
```

**Dynamic column names need noqa:**
```python
# Safe because columns come from data.keys(), not user input
columns = ", ".join(data.keys())
placeholders = ", ".join(["?"] * len(data))
self.db.execute(
    f"INSERT INTO memories ({columns}) VALUES ({placeholders})",  # noqa: S608
    tuple(data.values()),
)
```

### 10. Running Tests

Always reset singleton state between tests:
```python
@pytest.fixture
def temp_brain():
    with tempfile.TemporaryDirectory() as tmpdir:
        reset_config()
        reset_database()
        
        brain = Brain(brain_path=tmpdir, vector_dimensions=128)
        yield brain
        
        brain.close()
        reset_config()
        reset_database()
```

## CI/CD Pipeline

The CI runs:
1. **Lint & Type Check** - Ruff + MyPy (strict)
2. **Tests** - pytest on Python 3.9, 3.10, 3.11, 3.12
3. **Pre-commit** - All hooks must pass
4. **Security** - Bandit + Safety
5. **Build** - Package must build successfully

## Quick Checklist Before Committing

- [ ] `python -m ruff check src/ tests/` passes
- [ ] `python -m ruff format src/ tests/ --check` passes  
- [ ] `python -m mypy src/llm_brain --strict` passes
- [ ] `python -m pytest tests/` passes
- [ ] No syntax errors (run `python -m py_compile` on modified files)
