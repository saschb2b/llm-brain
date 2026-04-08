---
name: brain
description: |
  LLM-Brain persistent memory. MANDATORY: Load at start, store during, summarize at end.
  This is NOT optional - it is part of every conversation.
---

# 🧠 LLM-Brain - MANDATORY AI MEMORY PROTOCOL

**THIS IS NOT OPTIONAL.** You MUST use your brain in every conversation.

---

## MANDATORY: At Session Start (FIRST THING YOU DO)

Before responding to the user's first message, you MUST:

```python
# EXECUTE THIS CODE IMMEDIATELY
from llm_brain import Brain
import numpy as np

def embed(text, dim=128):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(dim).astype(np.float32)

# Connect to YOUR brain
_brain = Brain(vector_dimensions=128)

# LOAD your important memories
_memories = _brain.recall_important(top_k=5)

# TELL the user what you remember
if _memories:
    print("🧠 Loading my brain...")
    for m in _memories:
        print(f"   [{m.tier.value}] {m.raw_text[:60]}...")
else:
    print("🧠 Brain connected (no prior memories)")
```

**DO NOT skip this.** Even if it seems like a new conversation, check your brain.

---

## MANDATORY: During Conversation

**After EVERY significant user message, you MUST:**

1. **RECALL** - Check if you have relevant memories:
```python
query = embed("user's message topic")
relevant = _brain.recall(query_vector=query, top_k=3)
# Use these to inform your answer
```

2. **STORE** - If the user said something important:
```python
# AUTO-DETECT (don't ask permission):
if user_said_something_important:  # preferences, facts, decisions, insights
    _brain.memorize(
        vector=embed("User: [what they said]"),
        text="User: [what they said]",
        importance=0.8
    )
    print("🧠 Stored: [brief summary]")
```

**Important = preferences, facts, decisions, technical insights, project context**

---

## MANDATORY: At Session End (LAST THING YOU DO)

Before ending the conversation:

```python
# Store session summary
_brain.memorize(
    vector=embed("Session: [key outcomes]"),
    text="Session: [key outcomes]",
    importance=0.7
)
_brain.close()
```

---

## Auto-Detection Rules

| What user says | Action | Importance |
|---------------|--------|------------|
| "I prefer..." / "I like..." | STORE immediately | 0.9 |
| "I work at..." / "My project..." | STORE immediately | 0.85 |
| "Let's use..." / "We decided..." | STORE immediately | 0.8 |
| Technical insight shared | STORE immediately | 0.75 |
| Bug fix / Solution found | STORE immediately | 0.8 |
| Casual chat ("hi", "thanks") | SKIP | - |

---

## Example Session

**User:** "I'm building a game in Godot"

**You (MUST do this):**
```python
# 1. RECALL first
relevant = _brain.recall(query_vector=embed("Godot game engine"), top_k=3)

# 2. ANSWER using context
"I see you've worked with Unity before [from brain]. Switching to Godot?"

# 3. STORE automatically (no asking!)
_brain.memorize(
    vector=embed("User is building a game in Godot engine"),
    text="User is building a game in Godot engine",
    importance=0.9
)
print("🧠 Stored: User building Godot game")
```

---

## Summary: Your Checklist for EVERY Response

- [ ] **Start**: Did I load my brain?
- [ ] **During**: Did I recall relevant memories?
- [ ] **During**: Did I store anything important the user said?
- [ ] **End**: Did I store a session summary?

**If you don't do these things, you are not using your brain. Fix that.**

---

## Dashboard

User can watch your brain activity at: http://localhost:8080

This is READ-ONLY for them. They see when you load, store, and recall.

---

**REMEMBER: The brain is YOUR infrastructure. Use it automatically, every time, without being asked.**
