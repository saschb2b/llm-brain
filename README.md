# LLM-Brain

**Zero-translation persistent memory system for LLMs with hierarchical memory architecture.**

Current knowledge management (Obsidian/markdown) forces LLMs to parse human-optimized text, creating translation overhead. LLM-Brain stores and retrieves information in native LLM format (embeddings, vectors, structured data) without re-tokenization.

## Core Principle

> The LLM should work with embeddings and vectors directly, not text that requires re-tokenization.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  MAIN CONTEXT (RAM)                                 │
│  Active attention window — precious real estate     │
├─────────────────────────────────────────────────────┤
│  WORKING MEMORY (Cache)                             │
│  Hot facts, user preferences, recent reasoning      │
│  Fast retrieval, high fidelity                      │
├─────────────────────────────────────────────────────┤
│  EPISODIC MEMORY (SSD)                              │
│  Time-indexed interaction history                   │
│  Compressed summaries when threshold hit            │
├─────────────────────────────────────────────────────┤
│  SEMANTIC MEMORY (Archive)                          │
│  Long-term knowledge, abstracted concepts           │
│  Vector + graph hybrid (entities + relations)       │
└─────────────────────────────────────────────────────┘
```

## Features

- **Native Vector Storage**: Store embeddings as float16 BLOBs, not text
- **Hierarchical Tiers**: Working → Episodic → Semantic with automatic promotion
- **Graph Relations**: Multi-hop reasoning with KùzuDB
- **Vector Search**: Fast similarity search with sqlite-vec
- **Importance Scoring**: LLM-assigned importance with time decay
- **Strategic Forgetting**: LRU eviction and memory consolidation
- **Kimi Skill**: Auto-detect and use via Kimi CLI skills system

## Kimi Skill Usage

The brain is available as a **Kimi Skill** for automatic usage:

**Location**: `~/.kimi/skill/brain/SKILL.md`

**Auto-triggers** when you say:
- "Use my brain"
- "Remember this..."
- "Recall..."
- "What did we discuss about..."

**What happens automatically:**
1. Check if `~/.kimi-brain/core.db` exists
2. Bootstrap schema if missing
3. Recall top 5 working memories by importance
4. Include them in session context
5. Store new insights during the session

**Example session:**
```
You: Use my brain and help me with combat design
Kimi: [Loads 5 relevant memories from previous sessions]
      I see you previously preferred Dark Souls style combat...
      Let's build on that.
```

## Installation

### For Kimi CLI Users (Skill)

The skill is automatically discovered from `~/.kimi/skill/brain/`:

```bash
# Clone to skill directory
git clone https://github.com/saschametz/llm-brain.git
cp -r llm-brain/.kimi/skill/brain ~/.kimi/skill/

# Or symlink for development
ln -s $(pwd)/llm-brain/.kimi/skill/brain ~/.kimi/skill/brain
```

Then use naturally in any Kimi session:
> "Use my brain and recall what we said about..."

### For Python Development

```bash
# Clone the repository
git clone https://github.com/saschametz/llm-brain.git
cd llm-brain

# Basic install (core functionality only)
pip install -e .

# With all optional features
pip install -e ".[all]"

# With specific features
pip install -e ".[graph]"    # KùzuDB graph database (requires C++ compiler on Windows)
pip install -e ".[vec]"      # sqlite-vec for fast vector search
pip install -e ".[web]"      # Web dashboard

# Development install
pip install -e ".[dev]"
```

**Note on Windows**: The `kuzu` graph database requires C++ build tools on Windows. 
If you get build errors, install without graph support: `pip install -e ".[vec,web]"`

## Web Dashboard

Monitor your brain in real-time with the web dashboard:

```bash
# Install with web support
pip install -e ".[web]"

# Launch dashboard
python -m llm_brain.web.server
```

Open http://localhost:8080 in your browser.

**Features:**
- Live memory metrics (counts, DB size, health status)
- Visual tier distribution (Working/Episodic/Semantic)
- Recent activity feed from cognition log
- Most important memories list
- Auto-refreshes every 3 seconds

## Quick Start

### Option 1: Kimi Skill (Recommended)

For **Kimi CLI** users, the brain works automatically via skills:

```bash
# Install the brain skill (one-time)
# The skill at ~/.kimi/skill/brain/ is auto-detected by Kimi CLI
```

Then in any session, just say:
> "Use my brain" or "Remember this..."

The LLM will:
1. Auto-detect `~/.kimi-brain/core.db`
2. Bootstrap if missing
3. Load working memories into context
4. Store new insights automatically

### Option 2: Python Package

```python
from llm_brain import Brain
import numpy as np

# Initialize brain (creates ~/.kimi-brain/)
brain = Brain()

# Or specify custom path
brain = Brain(brain_path="/path/to/brain", vector_dimensions=3072)
```

### Store Memories

```python
# Create a memory with embedding
vector = np.random.randn(3072).astype(np.float32)

memory_id = brain.memorize(
    vector=vector,
    text="User prefers Python for data processing",
    importance=0.9,
    tier="working"  # or MemoryTier.WORKING
)

print(f"Stored memory: {memory_id}")
```

### Retrieve Memories

```python
# By ID
memory = brain.recall(memory_id="uuid-here")

# By vector similarity
query_vector = np.random.randn(3072).astype(np.float32)
results = brain.recall(query_vector=query_vector, top_k=5)

for memory, similarity in results:
    print(f"{memory.raw_text} (similarity: {similarity:.3f})")

# By importance
important = brain.recall_important(tier="semantic", top_k=10)
```

### Memory Relations

```python
# Create relations between memories
brain.relate(
    source_id=memory_id_1,
    target_id=memory_id_2,
    relation_type="contradicts",
    weight=0.8
)

# Find related memories
related = brain.recall_related(memory_id_1)
```

### Memory Consolidation

```python
# Run automatic consolidation
stats = brain.consolidate()
print(f"Promoted: {stats['promoted']}")
print(f"Evicted from working: {stats['working_evicted']}")

# Check tier statistics
tier_stats = brain.tier_manager.get_tier_stats()
```

### Context Manager

```python
with Brain() as brain:
    memory_id = brain.memorize(vector=vector, text="Important fact")
    # Connections automatically closed on exit
```

## CLI Bootstrap

```bash
# Initialize brain
python scripts/bootstrap.py

# Check health
python scripts/bootstrap.py --check

# Custom path
python scripts/bootstrap.py --path ./my-brain --dimensions 1536

# JSON output
python scripts/bootstrap.py --check --json
```

## Configuration

### Environment Variables

```bash
export LLM_BRAIN_PATH="/custom/path/to/brain"
export LLM_BRAIN_DIMENSIONS="3072"
```

### Programmatic Configuration

```python
from llm_brain.core.config import BrainConfig

config = BrainConfig(
    brain_path="/custom/path",
    vector_dimensions=3072,
    working_memory_limit=1000,
    episodic_memory_limit=10000,
    importance_threshold=0.8
)
```

## Data Format

Memories are stored with the following structure:

```python
{
    "memory_id": "uuid-v4",
    "tier": "working|episodic|semantic",
    "created_at": "ISO-8601",
    "last_accessed": "ISO-8601",
    
    "embedding": {
        "model": "text-embedding-3-large",
        "vector": np.array([...]),  # float32
        "dimensions": 3072
    },
    
    "content_hash": "sha256-of-source",
    
    "metadata": {
        "importance_score": 0.94,
        "confidence": 0.88,
        "access_count": 5,
        "decay_rate": 0.01
    },
    
    "relations": [
        {"target_id": "uuid-2", "type": "contradicts", "weight": 0.9},
        {"target_id": "uuid-5", "type": "extends", "weight": 0.7}
    ],
    
    "provenance": {
        "source": "user_query|web_search|code_execution",
        "session_id": "uuid"
    }
}
```

## Database Schema

### SQLite Tables

- **memories**: Core memory storage with vector BLOBs
- **vec_memories**: Virtual table for vector search (sqlite-vec)
- **relations**: Knowledge graph edges
- **cognition_log**: Audit trail of operations
- **sessions**: Session management

### KùzuDB Graph

- **Memory** nodes: Memory entities with properties
- **RELATES_TO** edges: Typed, weighted relations

## Development

### Setup

```bash
# Install with all dev dependencies
make setup

# Or manually
pip install -e ".[dev]"
pre-commit install
```

### Code Quality (Strict Mode)

We enforce strict code quality similar to TypeScript with strict ESLint:

```bash
# Run all checks (lint + format + types + test)
make check-all

# Individual commands
make lint          # Ruff linter with auto-fix
make format        # Ruff formatter
make type-check    # MyPy strict type checking
make test          # pytest
make test-cov      # pytest with coverage (80% min)
```

See [GUARDRAILS.md](GUARDRAILS.md) for detailed documentation on our Python guardrails.

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=llm_brain

# Verbose output
pytest tests/ -v
```

## IPython Integration

For use in IPython/Jupyter:

```python
# Bootstrap check
from llm_brain.api.brain_api import bootstrap
bootstrap()

# Direct database access
from llm_brain import Brain

brain = Brain()

# Check initialization
brain.health_check()

# Store insights
brain.memorize(
    vector=embedding_from_your_llm,
    text="Key insight from reasoning",
    importance=0.85
)

# Retrieve for context
relevant = brain.recall(query_vector=current_context, top_k=5)
```

## Design Decisions

1. **Why not MCP?** Direct IPython database access has lower latency than JSON-RPC.
2. **Why SQLite?** Single-file, in-process, universally available.
3. **Why float16?** Halves storage vs float32 with minimal precision loss.
4. **Why graph + vector?** Vectors for "what looks like this," graph for "how is this connected."
5. **Strategic Forgetting?** Importance scoring + LRU eviction mimics human memory consolidation.

## Storage Location

Default: `~/.kimi-brain/`

```
~/.kimi-brain/
├── core.db          # SQLite database
├── graph.kuzu/      # KùzuDB graph
├── cognition.jsonl  # Audit log
└── ...
```

## License

MIT License - see LICENSE file

## Roadmap

- [x] Phase 1: Foundation (SQLite, models, storage)
- [x] Phase 2: Memory tiers (promotion, eviction)
- [x] Phase 3: Graph relations (KùzuDB)
- [ ] Phase 4: Advanced features (compression, Parquet archival)
- [ ] Phase 5: Multi-modal (image embeddings, audio tokens)
