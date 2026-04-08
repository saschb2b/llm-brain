# LLM-Brain Integration

This project is a **persistent memory system for LLMs**. The core library is LLM-agnostic.

## Quick Start (Claude Code)

Claude Code doesn't have inline Python execution like Kimi — use the Bash tool instead.

### At Session Start

```bash
cd /path/to/llm-brain
python -c "
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

brain = Brain(vector_dimensions=128)
memories = brain.recall_important(top_k=5)
if memories:
    for m in memories:
        print(f'[{m.tier.value}] {m.raw_text[:80]}')
else:
    print('Brain connected (no prior memories)')
brain.close()
"
```

### Store a Memory

```bash
python -c "
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

brain = Brain(vector_dimensions=128)
text = 'User prefers concise responses'
brain.memorize(vector=embed(text), text=text, importance=0.9)
brain.close()
print('Stored:', text)
"
```

### Recall Memories

```bash
python -c "
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

brain = Brain(vector_dimensions=128)
results = brain.recall(query_vector=embed('user preferences'), top_k=5)
for memory, similarity in results:
    print(f'[{similarity:.2f}] {memory.raw_text[:80]}')
brain.close()
"
```

### Health Check

```bash
python -c "from llm_brain.api.brain_api import bootstrap; import json; print(json.dumps(bootstrap(vector_dimensions=128), indent=2, default=str))"
```

## Key Differences from Kimi Integration

| Aspect | Kimi 2.5 | Claude Code |
|--------|----------|-------------|
| Code execution | Inline IPython (direct) | Bash tool (`python -c "..."`) |
| Protocol file | `.kimi/skill/brain/SKILL.md` | This file (`CLAUDE.md`) |
| Session lifecycle | Auto-managed by skill | Manual via bash commands |
| Embedding | Same hash-based dev embed | Same hash-based dev embed |

## Storage

- Database: `~/.llm-brain/core.db` (SQLite with vector storage)
- Graph: `~/.llm-brain/graph.kuzu` (optional, requires kuzu)
- Log: `~/.llm-brain/cognition.jsonl`

## Dashboard

```bash
pip install -e ".[web]"
python -m llm_brain.web.server
# Open http://localhost:8080
```

## Architecture

3-tier memory hierarchy:
1. **Working Memory** - Hot cache, auto-loaded at session start
2. **Episodic Memory** - Time-indexed session history
3. **Semantic Memory** - Long-term consolidated knowledge

## Development

```bash
pip install -e ".[all]"
pytest
ruff check src/
```
