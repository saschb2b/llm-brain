---
name: brain
description: |
  LLM-Brain persistent memory system. This is FOR YOU (the AI), not the user.
  Automatically load memories at session start, store insights during conversation.
---

# LLM-Brain Skill - AI Integration Guide

## ⚠️ CRITICAL: This is YOUR Brain, Not the User's

**The user does NOT interact with the brain directly.**
- They do NOT type "remember this"
- They do NOT query it manually  
- They do NOT store things in it

**YOU (the AI) use it automatically:**
- Load memories at session start
- Store facts without asking
- Recall context before answering

The dashboard (`http://localhost:8080`) is for the user to WATCH what you're doing, not to interact.

---

## Mandatory Session Protocol

### 1. AT SESSION START (ALWAYS DO THIS)

```python
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

# Connect to brain
brain = Brain(vector_dimensions=128)

# LOAD: Get top 5 most important working memories
memories = brain.recall_important(top_k=5)

# INJECT: Add these to your context
if memories:
    print("=== LOADED FROM BRAIN ===")
    for m in memories:
        print(f"  [{m.tier.value}] {m.raw_text}")
    print("========================")
```

### 2. DURING CONVERSATION (AUTOMATIC - NO USER PROMPT)

**When the user says something important, STORE IT:**

```python
# Detect important facts automatically (don't ask):
# - User preferences ("I like...", "I prefer...")
# - Key decisions ("Let's use...", "We decided...")  
# - Personal facts ("I work at...", "My project is...")
# - Technical insights

# Store without mentioning it:
brain.memorize(
    vector=embed("User said: [important thing]"),
    text="User said: [important thing]",
    importance=0.8  # High for facts/preferences
)
```

**Before answering complex questions, RECALL:**

```python
# Search for relevant context
query = embed("User's question topic")
relevant = brain.recall(query_vector=query, top_k=3)

# Use this context to inform your answer
```

### 3. AT SESSION END

```python
# Store session summary
summary = "Key outcomes from this session"
brain.memorize(
    vector=embed(summary),
    text=summary,
    importance=0.7
)

brain.close()
```

---

## What to Store (Automatic Detection)

| Type | Example | Importance |
|------|---------|------------|
| Preferences | "I prefer dark mode" | 0.9 |
| Facts | "I work at company X" | 0.85 |
| Decisions | "We'll use Python" | 0.8 |
| Insights | "The bug was in caching" | 0.75 |
| Context | "This is a React project" | 0.7 |
| Casual chat | "Nice weather" | 0.3 (skip) |

---

## Example Session

**User:** "I'm building a game in Godot"

**You (internally):**
```python
brain.memorize(
    vector=embed("User is building a game in Godot engine"),
    text="User is building a game in Godot engine",
    importance=0.9
)
```

**Later - User:** "What engine should I use?"

**You (internally):**
```python
relevant = brain.recall(query_vector=embed("game engine"))
# Finds: "User is building a game in Godot engine"
```

**You (aloud):** "You're already using Godot for your game project. Stick with it or switch?"

---

## Dashboard (User Monitoring)

If the user has the dashboard open at `http://localhost:8080`, they can see:
- What memories you loaded at start
- When you store new memories  
- Your recall operations
- Memory statistics

**This is READ-ONLY for them.** They watch, you operate.

---

## Installation Check

```python
# Run this at session start to ensure brain exists
from pathlib import Path
from llm_brain import Brain

brain_path = Path.home() / ".kimi-brain"
if not (brain_path / "core.db").exists():
    brain = Brain()  # Creates it
    print("🧠 Brain initialized")
else:
    print("🧠 Brain connected")
```

---

**REMEMBER:** The user doesn't manage the brain. YOU do. Load at start, store automatically, recall before answering. The brain is your infrastructure, not their tool.
