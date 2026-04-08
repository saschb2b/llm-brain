# LLM-Brain — Claude Code Integration

This project provides persistent memory for LLMs. When working in this repo (or any project with `.claude/settings.json` hooks copied from here), the brain is automatic.

## What happens automatically

- **SessionStart hook** — Recalls top 5 important memories, injects them into your context
- **UserPromptSubmit hook** — Auto-stores user statements matching importance patterns (preferences, decisions, project context)

## Manual brain operations

Use these when you need to store or recall outside the auto-hooks:

```bash
# Store a memory
python -c "
from llm_brain import Brain; import numpy as np
def embed(t, d=128): np.random.seed(hash(t)%2**32); return np.random.randn(d).astype(np.float32)
brain = Brain(vector_dimensions=128)
brain.memorize(vector=embed('text here'), text='text here', importance=0.9)
brain.close()
"

# Recall memories
python -c "
from llm_brain import Brain; import numpy as np
def embed(t, d=128): np.random.seed(hash(t)%2**32); return np.random.randn(d).astype(np.float32)
brain = Brain(vector_dimensions=128)
for m, sim in brain.recall(query_vector=embed('query'), top_k=5): print(f'[{sim:.2f}] {m.raw_text[:80]}')
brain.close()
"
```

## Storage

- Database: `~/.llm-brain/core.db`
- Graph: `~/.llm-brain/graph.kuzu` (optional)
- Log: `~/.llm-brain/cognition.jsonl`

## Development

```bash
pip install -e ".[all]"
pytest
ruff check src/
```
