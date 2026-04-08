---
name: brain
description: |
  LLM-Brain persistent memory system. Automatically loads memories at session start,
  stores important exchanges during conversation, and recalls relevant context.
---

# LLM-Brain Skill - AI Instructions

## CRITICAL: Auto-Trigger Behavior

**AT SESSION START (MANDATORY):**
```python
# ALWAYS run this at the beginning of EVERY session
from llm_brain import Brain
import numpy as np

brain = Brain(vector_dimensions=128)

# Load top 5 most important working memories
important_memories = brain.recall_important(top_k=5)

# Make them available in your context
if important_memories:
    print("=== LOADED MEMORIES ===")
    for m in important_memories:
        print(f"- [{m.tier.value}] {m.raw_text}")
    print("=== END MEMORIES ===")
```

**DURING CONVERSATION (AUTOMATIC):**

When the user says something **important**, **factual**, or **preference-related**:

```python
# AUTOMATICALLY store important information
def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

# Store without asking - this is automatic
brain.memorize(
    vector=embed("What the user said"),
    text="What the user said",
    importance=0.8  # High for facts/preferences, 0.5 for casual
)
```

**Store when user mentions:**
- Preferences ("I like...", "I prefer...", "I hate...")
- Facts about themselves ("I work at...", "My project is...")
- Important decisions ("Let's use...", "We decided...")
- Context that spans sessions ("Remember when...")

**BEFORE ANSWERING (AUTOMATIC):**

```python
# ALWAYS recall relevant context before answering
query_vector = embed("User's question topic")
relevant = brain.recall(query_vector=query_vector, top_k=3)

# Use these memories to inform your answer
```

**AT SESSION END (MANDATORY):**

```python
# Store key takeaways from this session
brain.memorize(
    vector=embed("Summary of what we accomplished"),
    text="Summary of what we accomplished",
    importance=0.9
)

# Close properly
brain.close()
```

## Quick Reference

**Check brain health:**
```python
health = brain.health_check()
print(f"Initialized: {health['initialized']}, Writable: {health['writable']}")
```

**View all memories:**
```python
for tier in ['working', 'episodic', 'semantic']:
    memories = brain.recall_by_tier(tier)
    print(f"{tier}: {len(memories)} memories")
```

**Dashboard (for user to watch):**
```bash
python -m llm_brain.web.server
# Open http://localhost:8080
```

## Installation Check

```python
# Run at session start to ensure brain exists
import os
from pathlib import Path

brain_path = Path.home() / ".kimi-brain"
if not (brain_path / "core.db").exists():
    # First time - initialize
    from llm_brain import Brain
    brain = Brain()
    print("🧠 Brain initialized for first time")
```

## Examples of Auto-Storage

**User:** "I prefer dark mode"
**You:** [Automatically store] `"User prefers dark mode interfaces"`

**User:** "We're building a game"
**You:** [Automatically store] `"User is building a game project"`

**User:** "Don't use JavaScript"
**You:** [Automatically store] `"User does not want to use JavaScript"`

**User:** "What did we work on yesterday?"
**You:** [Recall memories from previous sessions] + Answer

---

**REMEMBER:** The brain is useless if you don't use it automatically. 
Store facts without asking. Recall context before answering. Every session.
