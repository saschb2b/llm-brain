# LLM-Brain — Claude Code Integration

This project provides persistent memory for LLMs via the `llm_brain` Python package. You ARE the memory manager — decide what to store, update, connect, and forget.

## Your role as memory manager

You have a persistent brain available. Use it intelligently:

- **Store** facts worth remembering across sessions: user identity, preferences, decisions, project context, lessons learned
- **Recall** when context would help you give a better answer
- **Update** memories when information changes (forget old, store new)
- **Connect** related memories with relations (e.g., a preference relates to a project decision)
- **Forget** outdated or incorrect memories

Do NOT store: ephemeral task details, things obvious from code/git, or raw conversation logs. Store the *insight*, not the transcript.

## How it works

- **SessionStart hook** automatically loads your top 5 important memories into context
- **You** decide during conversation what's worth storing — no keyword matching, no static rules

## Brain operations

### Store a memory

```bash
python -c "
from llm_brain import Brain; import numpy as np
def embed(t, d=128): np.random.seed(hash(t)%2**32); return np.random.randn(d).astype(np.float32)
brain = Brain(vector_dimensions=128)
brain.memorize(vector=embed('descriptive text'), text='descriptive text', importance=0.8, tier='working')
brain.close()
"
```

Parameters:
- `text`: What to remember (concise, meaningful)
- `importance`: 0.0–1.0 (0.9+ for core identity/preferences, 0.7–0.8 for project context, 0.5–0.6 for nice-to-know)
- `tier`: `working` (hot/recent), `episodic` (interaction history), `semantic` (long-term knowledge)

### Recall memories

```bash
python -c "
from llm_brain import Brain; import numpy as np
def embed(t, d=128): np.random.seed(hash(t)%2**32); return np.random.randn(d).astype(np.float32)
brain = Brain(vector_dimensions=128)
for m, sim in brain.recall(query_vector=embed('query'), top_k=5): print(f'[{sim:.2f}] id={m.memory_id} | {m.raw_text[:100]}')
brain.close()
"
```

### Forget a memory

```bash
python -c "
from llm_brain import Brain
brain = Brain(vector_dimensions=128)
brain.forget('memory-id-here')
brain.close()
"
```

### Connect memories

```bash
python -c "
from llm_brain import Brain
brain = Brain(vector_dimensions=128)
brain.relate('source-id', 'target-id', 'relation_type', weight=0.8)
brain.close()
"
```

Relation types: `related_to`, `contradicts`, `supersedes`, `supports`, `part_of`

### Get brain stats

```bash
python -c "
from llm_brain import Brain; import json
brain = Brain(vector_dimensions=128)
print(json.dumps(brain.stats(), indent=2, default=str))
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
