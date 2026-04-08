# LLM-Brain

**Persistent memory for LLMs.** Give your AI assistant memory that survives across sessions.

## Why

Every time you start a new conversation, your AI starts from scratch. It forgets your preferences, your project context, past decisions. LLM-Brain fixes that — it gives any LLM a persistent memory layer that auto-loads at session start and stores important context as you work.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/saschb2b/llm-brain.git
cd llm-brain
pip install -e ".[vec]"
```

### 2. Add to your project

Copy the `.claude/` directory into your project (for Claude Code), or `.kimi/` (for Kimi 2.5):

```bash
# Claude Code
cp -r .claude/ /path/to/your-project/.claude/

# Kimi 2.5
cp -r .kimi/ /path/to/your-project/.kimi/
```

That's it. The hooks handle everything automatically:
- **Session start** — recalls your top 5 memories
- **During conversation** — auto-stores important statements (preferences, decisions, project context)

### 3. Verify

Start a new Claude Code session in your project and check:

```bash
python -c "
from llm_brain import Brain
brain = Brain(vector_dimensions=128)
print(brain.health_check())
brain.close()
"
```

## How It Works

All LLMs share a single brain stored at `~/.llm-brain/core.db` (SQLite with vector storage).

**Auto-recall:** At session start, the hook loads your most important memories and injects them into the AI's context.

**Auto-store:** When you say something matching importance patterns ("I prefer...", "my project...", "we decided..."), it's stored automatically.

**Manual store/recall:** The AI can also use the Python API directly via bash:

```python
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

brain = Brain(vector_dimensions=128)

# Store
brain.memorize(vector=embed("important fact"), text="important fact", importance=0.9)

# Recall
results = brain.recall(query_vector=embed("query"), top_k=5)

brain.close()
```

## Architecture

3-tier memory hierarchy, modeled after human cognition:

```
WORKING MEMORY    — Hot cache. Auto-loaded at session start.
EPISODIC MEMORY   — Session history. Compressed over time.
SEMANTIC MEMORY   — Long-term knowledge. Consolidated facts.
```

## Supported LLMs

| LLM | Integration | How |
|-----|-------------|-----|
| **Claude Code** | `.claude/settings.json` hooks | Auto-recall on SessionStart, auto-store on UserPromptSubmit |
| **Kimi 2.5** | `.kimi/skill/brain/SKILL.md` | Inline IPython execution via skill protocol |
| **Others** | Python API | Use the `Brain` class directly — it's LLM-agnostic |

## Dashboard (Read-Only)

Watch what your AI remembers in real-time:

```bash
pip install -e ".[web]"
python -m llm_brain.web.server
# Open http://localhost:8080
```

## Configuration

| Setting | Default | Env Variable |
|---------|---------|-------------|
| Brain path | `~/.llm-brain` | `LLM_BRAIN_PATH` |
| Vector dimensions | 128 (hooks) / 3072 (API) | `LLM_BRAIN_DIMENSIONS` |

## Development

```bash
pip install -e ".[all]"
pytest
ruff check src/
```

## Status

Research infrastructure. The brain works, the hooks work, the dashboard works. The embedding is currently hash-based (deterministic but not semantic) — real embedding models are the next step.
